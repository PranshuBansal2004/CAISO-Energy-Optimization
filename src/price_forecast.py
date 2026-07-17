"""
price_forecast.py

Trying to predict real time prices using day ahead prices + past week
of history, like professor asked in part C.

Using XGBoost and LightGBM, comparing against a simple baseline of
just using the day ahead price as our guess.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_utils import load_day_ahead, load_real_time, aggregate_real_time_hourly, RESULTS_DIR


def make_features(da, rt_hourly):
    da_small = da[['date', 'hour', 'month', 'LMP']].rename(columns={'LMP': 'price_da'})
    rt_small = rt_hourly[['date', 'hour', 'LMP_RT']].rename(columns={'LMP_RT': 'price_rt'})
    df = da_small.merge(rt_small, on=['date', 'hour']).sort_values(['date', 'hour']).reset_index(drop=True)

    df['date_ts'] = pd.to_datetime(df['date'].astype(str))
    df['dow'] = df['date_ts'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # lag features - price from same hour 1, 2, 7 days ago
    df['rt_lag24'] = df['price_rt'].shift(24)
    df['rt_lag48'] = df['price_rt'].shift(48)
    df['rt_lag168'] = df['price_rt'].shift(168)

    # rolling stats over past week
    df['rt_roll_mean'] = df['price_rt'].shift(1).rolling(168, min_periods=24).mean()
    df['rt_roll_std'] = df['price_rt'].shift(1).rolling(168, min_periods=24).std()

    df = df.dropna().reset_index(drop=True)

    feature_cols = ['price_da', 'hour_sin', 'hour_cos', 'dow',
                     'rt_lag24', 'rt_lag48', 'rt_lag168',
                     'rt_roll_mean', 'rt_roll_std']

    return df, feature_cols


def run():
    print("loading data...")
    da = load_day_ahead()
    rt = load_real_time()
    rt_hourly = aggregate_real_time_hourly(rt)

    df, feature_cols = make_features(da, rt_hourly)
    print(f"got {len(df)} rows for training")

    X = df[feature_cols].values
    y = df['price_rt'].values

    # train test split by time (dont shuffle, its a time series)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    da_test = df['price_da'].values[split:]

    results = {}

    import xgboost as xgb
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    pred_xgb = model.predict(X_test)
    results['xgboost'] = {
        'mae': round(mean_absolute_error(y_test, pred_xgb), 2),
        'rmse': round(np.sqrt(mean_squared_error(y_test, pred_xgb)), 2),
        'r2': round(r2_score(y_test, pred_xgb), 3),
    }
    print(f"xgboost - MAE: {results['xgboost']['mae']}, R2: {results['xgboost']['r2']}")

    import lightgbm as lgb
    model2 = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
    model2.fit(X_train, y_train)
    pred_lgb = model2.predict(X_test)
    results['lightgbm'] = {
        'mae': round(mean_absolute_error(y_test, pred_lgb), 2),
        'rmse': round(np.sqrt(mean_squared_error(y_test, pred_lgb)), 2),
        'r2': round(r2_score(y_test, pred_lgb), 3),
    }
    print(f"lightgbm - MAE: {results['lightgbm']['mae']}, R2: {results['lightgbm']['r2']}")

    # baseline - just guess DA price = RT price
    results['baseline_da'] = {
        'mae': round(mean_absolute_error(y_test, da_test), 2),
        'rmse': round(np.sqrt(mean_squared_error(y_test, da_test)), 2),
        'r2': round(r2_score(y_test, da_test), 3),
    }
    print(f"baseline (DA=RT) - MAE: {results['baseline_da']['mae']}, R2: {results['baseline_da']['r2']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # feature importance from xgboost
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    print("\nfeature importance (xgboost):")
    for feat, imp in importance.items():
        print(f"  {feat}: {imp:.3f}")

    plt.figure(figsize=(9, 6))
    plt.barh(list(importance.keys())[::-1], list(importance.values())[::-1])
    plt.xlabel('importance')
    plt.title('feature importance for RT price prediction')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'forecast_feature_importance.png')
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, pred_xgb, alpha=0.1, s=5)
    lims = [min(y_test.min(), pred_xgb.min()), max(y_test.max(), pred_xgb.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('actual RT price')
    plt.ylabel('predicted RT price')
    plt.title('xgboost predictions vs actual')
    plt.savefig(RESULTS_DIR / 'forecast_scatter.png')
    plt.close()

    with open(RESULTS_DIR / 'forecast_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nsaved forecast_results.json and charts")
    return results


if __name__ == "__main__":
    run()
