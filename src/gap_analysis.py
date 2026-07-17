"""
gap_analysis.py

Trying to figure out where the 17% gap between day-ahead and perfect
foresight is actually coming from. Professor Kash asked if its from
certain times of day, days of week, months, or seasons.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_utils import load_day_ahead, load_real_time, aggregate_real_time_hourly, RESULTS_DIR


def analyze_gap():
    da = load_day_ahead()
    rt = load_real_time()
    rt_hourly = aggregate_real_time_hourly(rt)

    # merge day ahead and real time prices on date + hour
    da_small = da[['date', 'hour', 'month', 'LMP']].rename(columns={'LMP': 'price_da'})
    rt_small = rt_hourly[['date', 'hour', 'LMP_RT']].rename(columns={'LMP_RT': 'price_rt'})
    df = da_small.merge(rt_small, on=['date', 'hour'])

    df['gap'] = df['price_da'] - df['price_rt']
    df['abs_gap'] = df['gap'].abs()
    df['gap_sq'] = df['gap'] ** 2

    df['date_ts'] = pd.to_datetime(df['date'].astype(str))
    df['dow'] = df['date_ts'].dt.day_name()

    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}
    df['season'] = df['month'].map(season_map)

    total_sq = df['gap_sq'].sum()

    # ---- by hour ----
    by_hour = df.groupby('hour').agg(mean_abs_gap=('abs_gap', 'mean'))
    by_hour['contribution_pct'] = df.groupby('hour')['gap_sq'].sum() / total_sq * 100

    print("GAP BY HOUR")
    for h in range(24):
        row = by_hour.loc[h]
        print(f"hour {h:02d}: avg gap ${row['mean_abs_gap']:.2f}, contributes {row['contribution_pct']:.1f}% of total")

    plt.figure(figsize=(10, 5))
    plt.bar(range(24), by_hour['contribution_pct'], color='steelblue')
    plt.xlabel('Hour of day')
    plt.ylabel('Contribution to gap (%)')
    plt.title('Which hours cause the DA vs RT gap?')
    plt.xticks(range(0, 24, 2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / 'gap_by_hour.png')
    plt.close()

    # ---- by month ----
    by_month = df.groupby('month').agg(mean_abs_gap=('abs_gap', 'mean'))
    by_month['contribution_pct'] = df.groupby('month')['gap_sq'].sum() / total_sq * 100
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    print("\nGAP BY MONTH")
    for m in range(1, 13):
        row = by_month.loc[m]
        print(f"{months[m-1]}: contributes {row['contribution_pct']:.1f}%")

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 13), by_month['contribution_pct'], color='coral')
    plt.xticks(range(1, 13), months)
    plt.ylabel('Contribution to gap (%)')
    plt.title('Which months cause the DA vs RT gap?')
    plt.savefig(RESULTS_DIR / 'gap_by_month.png')
    plt.close()

    # ---- by season ----
    by_season = df.groupby('season').agg(mean_abs_gap=('abs_gap', 'mean'))
    by_season['contribution_pct'] = df.groupby('season')['gap_sq'].sum() / total_sq * 100
    by_season = by_season.reindex(['Winter', 'Spring', 'Summer', 'Fall'])

    print("\nGAP BY SEASON")
    for s in ['Winter', 'Spring', 'Summer', 'Fall']:
        print(f"{s}: contributes {by_season.loc[s, 'contribution_pct']:.1f}%")

    # ---- by day of week ----
    by_dow = df.groupby('dow').agg(mean_abs_gap=('abs_gap', 'mean'))
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    by_dow = by_dow.reindex(days_order)

    print("\nGAP BY DAY OF WEEK")
    for d in days_order:
        print(f"{d}: avg gap ${by_dow.loc[d, 'mean_abs_gap']:.2f}")

    # save everything to json so we can reference it later
    results = {
        'by_hour': by_hour.round(2).to_dict('index'),
        'by_month': by_month.round(2).to_dict('index'),
        'by_season': by_season.round(2).to_dict('index'),
        'by_day_of_week': by_dow.round(2).to_dict('index'),
    }
    with open(RESULTS_DIR / 'gap_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nsaved gap_analysis_results.json and charts")
    return results


if __name__ == "__main__":
    analyze_gap()
