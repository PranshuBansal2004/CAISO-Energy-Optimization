from __future__ import annotations
"""Compare workload scheduling savings across seasons."""

import json

import numpy as np

from src.data_utils import RESULTS_DIR, load_optimization_data
from src.optimize import (
    CAPACITY,
    TOTAL_DAILY_WORK,
    run_scenario1,
    run_scenario2,
    run_scenario4,
)

FLEX_RATIO = 0.30
SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall": [9, 10, 11],
}


def run_seasonal() -> dict:
    day_ahead, real_time_hourly, complete_days = load_optimization_data()
    flex_work = TOTAL_DAILY_WORK * FLEX_RATIO
    rigid_per_hour = (TOTAL_DAILY_WORK - flex_work) / 24.0
    seasonal_results: dict[str, dict] = {}

    for season_name, months in SEASONS.items():
        season_days = [day for day in complete_days if day.month in months]
        baseline_costs: list[float] = []
        day_ahead_costs: list[float] = []
        perfect_costs: list[float] = []
        price_spreads: list[float] = []
        negative_hours = 0
        total_hours = 0

        for day in season_days:
            da_day = day_ahead[day_ahead["date"] == day].sort_values("Time_UTC")
            rt_day = real_time_hourly[real_time_hourly["date"] == day].sort_values("Time_UTC")
            prices_da = da_day["LMP"].to_numpy(dtype=float)
            prices_rt = rt_day["LMP_RT"].to_numpy(dtype=float)

            baseline = run_scenario1(prices_da, TOTAL_DAILY_WORK)
            day_ahead_cost = run_scenario2(
                prices_da, rigid_per_hour, flex_work, CAPACITY
            )
            perfect_cost = run_scenario4(
                prices_da, prices_rt, rigid_per_hour, flex_work, CAPACITY
            )

            baseline_costs.append(baseline)
            day_ahead_costs.append(baseline if day_ahead_cost is None else day_ahead_cost)
            perfect_costs.append(baseline if perfect_cost is None else perfect_cost)
            price_spreads.append(float(np.max(prices_da) - np.min(prices_da)))
            negative_hours += int(np.sum(prices_da < 0))
            total_hours += len(prices_da)

        avg_baseline = float(np.mean(baseline_costs))
        avg_day_ahead = float(np.mean(day_ahead_costs))
        avg_perfect = float(np.mean(perfect_costs))
        savings_da = (avg_baseline - avg_day_ahead) / avg_baseline * 100.0
        savings_perfect = (avg_baseline - avg_perfect) / avg_baseline * 100.0

        seasonal_results[season_name] = {
            "days": len(season_days),
            "avg_daily_cost_dumb": round(avg_baseline, 2),
            "avg_daily_cost_da": round(avg_day_ahead, 2),
            "avg_daily_cost_perfect": round(avg_perfect, 2),
            "savings_da_pct": round(savings_da, 2),
            "savings_perfect_pct": round(savings_perfect, 2),
            "gap_pct": round(savings_perfect - savings_da, 2),
            "avg_price_spread": round(float(np.mean(price_spreads)), 2),
            "negative_price_pct": round(negative_hours / total_hours * 100.0, 2),
        }

        result = seasonal_results[season_name]
        print(
            f"{season_name:<6} | {result['days']:>3} days | "
            f"DA savings {result['savings_da_pct']:>6.2f}% | "
            f"Perfect savings {result['savings_perfect_pct']:>6.2f}%"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "seasonal_results.json"
    path.write_text(json.dumps(seasonal_results, indent=2), encoding="utf-8")
    print(f"Saved {path.relative_to(RESULTS_DIR.parent)}")
    return seasonal_results


if __name__ == "__main__":
    run_seasonal()
