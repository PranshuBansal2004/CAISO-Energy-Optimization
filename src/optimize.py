from __future__ import annotations
"""Main workload scheduling optimization."""

import json

import cvxpy as cp
import numpy as np
import pandas as pd

from src.data_utils import RESULTS_DIR, load_optimization_data

TOTAL_DAILY_WORK = 48.0  # MWh per day, approximately 2 MW average
CAPACITY = 4.0  # MW maximum in any hour
FLEX_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ACCEPTED_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


def _solve(problem: cp.Problem) -> float | None:
    """Solve with the first available supported solver."""
    installed = set(cp.installed_solvers())
    for solver in ("ECOS", "CLARABEL", "SCS"):
        if solver not in installed:
            continue
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in ACCEPTED_STATUSES and problem.value is not None:
                return float(problem.value)
        except cp.error.SolverError:
            continue
    return None


def run_scenario1(prices_da: np.ndarray, total_work: float) -> float:
    """Scenario 1: distribute all work evenly over 24 hours."""
    flat_load = total_work / 24.0
    return float(np.sum(prices_da * flat_load))


def run_scenario2(
    prices_da: np.ndarray,
    rigid_per_hour: float,
    flex_work: float,
    capacity: float,
) -> float | None:
    """Scenario 2: schedule flexible work using day-ahead prices only."""
    flexible_load = cp.Variable(24)
    total_load = rigid_per_hour + flexible_load

    constraints = [
        cp.sum(flexible_load) == flex_work,
        flexible_load >= 0,
        total_load <= capacity,
    ]
    problem = cp.Problem(cp.Minimize(prices_da @ total_load), constraints)
    return _solve(problem)


def run_scenario4(
    prices_da: np.ndarray,
    prices_rt: np.ndarray,
    rigid_per_hour: float,
    flex_work: float,
    capacity: float,
) -> float | None:
    """Scenario 4: perfect foresight across day-ahead and real-time markets."""
    day_ahead_purchase = cp.Variable(24)
    real_time_purchase = cp.Variable(24)
    flexible_load = cp.Variable(24)
    total_load = rigid_per_hour + flexible_load

    constraints = [
        total_load == day_ahead_purchase + real_time_purchase,
        cp.sum(flexible_load) == flex_work,
        flexible_load >= 0,
        day_ahead_purchase >= 0,
        real_time_purchase >= 0,
        total_load <= capacity,
    ]
    cost = prices_da @ day_ahead_purchase + prices_rt @ real_time_purchase
    problem = cp.Problem(cp.Minimize(cost), constraints)
    return _solve(problem)


def calculate_results(
    flex_ratios: list[float] | None = None,
    total_daily_work: float = TOTAL_DAILY_WORK,
    capacity: float = CAPACITY,
) -> list[dict]:
    """Calculate optimization results for each workload flexibility ratio."""
    ratios = FLEX_RATIOS if flex_ratios is None else flex_ratios
    day_ahead, real_time_hourly, complete_days = load_optimization_data()

    print(f"Loaded {len(day_ahead):,} day-ahead rows.")
    print(f"Using {len(complete_days)} complete 24-hour Pacific-time days.")

    all_results: list[dict] = []

    for flex_ratio in ratios:
        flex_work = total_daily_work * flex_ratio
        rigid_per_hour = (total_daily_work - flex_work) / 24.0
        baseline_costs: list[float] = []
        day_ahead_costs: list[float] = []
        perfect_costs: list[float] = []

        for day in complete_days:
            da_day = day_ahead[day_ahead["date"] == day].sort_values("Time_UTC")
            rt_day = real_time_hourly[real_time_hourly["date"] == day].sort_values("Time_UTC")
            prices_da = da_day["LMP"].to_numpy(dtype=float)
            prices_rt = rt_day["LMP_RT"].to_numpy(dtype=float)

            baseline = run_scenario1(prices_da, total_daily_work)
            day_ahead_cost = run_scenario2(prices_da, rigid_per_hour, flex_work, capacity)
            perfect_cost = run_scenario4(
                prices_da, prices_rt, rigid_per_hour, flex_work, capacity
            )

            baseline_costs.append(baseline)
            day_ahead_costs.append(baseline if day_ahead_cost is None else day_ahead_cost)
            perfect_costs.append(baseline if perfect_cost is None else perfect_cost)

        avg_baseline = float(np.mean(baseline_costs))
        avg_day_ahead = float(np.mean(day_ahead_costs))
        avg_perfect = float(np.mean(perfect_costs))
        savings_da = (avg_baseline - avg_day_ahead) / avg_baseline * 100.0
        savings_perfect = (avg_baseline - avg_perfect) / avg_baseline * 100.0

        result = {
            "flex": int(round(flex_ratio * 100)),
            "days_analyzed": len(complete_days),
            "cost_dumb": round(avg_baseline, 2),
            "cost_da": round(avg_day_ahead, 2),
            "cost_perfect": round(avg_perfect, 2),
            "savings_da_pct": round(savings_da, 2),
            "savings_perfect_pct": round(savings_perfect, 2),
            "gap_pct": round(savings_perfect - savings_da, 2),
            "annual_savings_da": round((avg_baseline - avg_day_ahead) * 365, 2),
            "annual_savings_perfect": round((avg_baseline - avg_perfect) * 365, 2),
        }
        all_results.append(result)

        print(
            f"Flex {result['flex']:>2}% | Baseline ${result['cost_dumb']:>8.2f} | "
            f"DA ${result['cost_da']:>8.2f} ({result['savings_da_pct']:>6.2f}%) | "
            f"Perfect ${result['cost_perfect']:>8.2f} "
            f"({result['savings_perfect_pct']:>6.2f}%)"
        )

    return all_results


def save_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "optimization_results.json"
    csv_path = RESULTS_DIR / "optimization_results.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Saved {json_path.relative_to(RESULTS_DIR.parent)}")
    print(f"Saved {csv_path.relative_to(RESULTS_DIR.parent)}")


def run_all() -> list[dict]:
    results = calculate_results()
    save_results(results)
    return results


if __name__ == "__main__":
    run_all()
