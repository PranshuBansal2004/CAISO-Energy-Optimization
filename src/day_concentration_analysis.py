"""
day_concentration_analysis.py

Professor asked: is the 17% gain coming from a few crazy days, or is it
spread out evenly over the year? This script ranks every day by how
much the real time market helped that day and checks the concentration.

Also checking if theres a consistent pattern of moving work from
afternoon into other hours (thats a separate question he asked).
"""

import json
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_utils import load_optimization_data, RESULTS_DIR
from src.optimize import TOTAL_DAILY_WORK, CAPACITY, run_scenario1, run_scenario2, run_scenario4

FLEX_RATIO = 0.30


def solve_flex_schedule(prices, rigid_per_hour, flex_work, capacity):
    f = cp.Variable(24)
    x = rigid_per_hour + f
    constraints = [cp.sum(f) == flex_work, f >= 0, x <= capacity, x >= 0]
    prob = cp.Problem(cp.Minimize(prices @ x), constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    if prob.status == 'optimal':
        return f.value
    return None


def run():
    da, rt_hourly, days = load_optimization_data()
    flex_work = TOTAL_DAILY_WORK * FLEX_RATIO
    rigid_per_hour = (TOTAL_DAILY_WORK - flex_work) / 24

    print(f"checking {len(days)} days at 30% flexibility")

    rows = []
    schedule_sum = np.zeros(24)
    n_days = 0

    for day in days:
        d_da = da[da['date'] == day].sort_values('Time_UTC')
        d_rt = rt_hourly[rt_hourly['date'] == day].sort_values('Time_UTC')
        if len(d_rt) < 24:
            continue

        pda = d_da['LMP'].to_numpy(dtype=float)
        prt = d_rt['LMP_RT'].to_numpy(dtype=float)

        c1 = run_scenario1(pda, TOTAL_DAILY_WORK)
        c2 = run_scenario2(pda, rigid_per_hour, flex_work, CAPACITY)
        c4 = run_scenario4(pda, prt, rigid_per_hour, flex_work, CAPACITY)
        c2 = c2 if c2 is not None else c1
        c4 = c4 if c4 is not None else c1

        gain = c2 - c4  # how much RT access saved that specific day

        sched = solve_flex_schedule(pda, rigid_per_hour, flex_work, CAPACITY)
        if sched is not None:
            schedule_sum += sched
            n_days += 1

        rows.append({'date': str(day), 'gain': gain, 'max_price': pda.max(), 'min_price': pda.min()})

    df = pd.DataFrame(rows).sort_values('gain', ascending=False).reset_index(drop=True)
    total_gain = df['gain'].sum()

    # how much do the top days contribute?
    print("\nHOW CONCENTRATED IS THE GAIN?")
    for n in [10, 20, 50, 100]:
        pct = df.iloc[:n]['gain'].sum() / total_gain * 100
        print(f"top {n} days = {pct:.1f}% of the total yearly gain")

    print(f"\nmean gain per day: ${df['gain'].mean():.2f}")
    print(f"median gain per day: ${df['gain'].median():.2f}")
    print(f"biggest single day: {df.iloc[0]['date']} with ${df.iloc[0]['gain']:.2f}")

    top20_pct = df.iloc[:20]['gain'].sum() / total_gain * 100
    if top20_pct > 40:
        print("\n=> looks like the gain IS concentrated in a smaller set of days")
    else:
        print("\n=> looks like the gain is spread pretty evenly, not just a few crazy days")

    # top 10 days
    print("\nTOP 10 DAYS:")
    for i in range(10):
        r = df.iloc[i]
        print(f"{r['date']}: ${r['gain']:.2f} gain")

    # workload shift pattern - where does the optimizer put the flexible work?
    avg_schedule = schedule_sum / n_days
    flat = flex_work / 24
    print("\nWHERE DOES THE OPTIMIZER PUT FLEXIBLE WORK (avg MWh per hour)")
    for h in range(24):
        diff = avg_schedule[h] - flat
        note = "more" if diff > 0.3 else ("less" if diff < -0.3 else "")
        print(f"hour {h:02d}: {avg_schedule[h]:.2f} MWh (flat baseline = {flat:.2f}) {note}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # plot: cumulative gain curve
    plt.figure(figsize=(9, 6))
    n = len(df)
    df['cum_pct'] = df['gain'].cumsum() / total_gain * 100
    plt.plot(np.arange(1, n+1)/n*100, df['cum_pct'], label='actual')
    plt.plot([0, 100], [0, 100], 'k--', label='if it were even')
    plt.xlabel('% of days (ranked highest gain first)')
    plt.ylabel('cumulative % of total gain')
    plt.title('Is the RT market benefit concentrated or spread out?')
    plt.legend()
    plt.savefig(RESULTS_DIR / 'rt_gain_concentration_curve.png')
    plt.close()

    # plot: workload shift
    plt.figure(figsize=(10, 5))
    plt.bar(range(24), avg_schedule, color='steelblue', label='optimized')
    plt.axhline(flat, color='red', linestyle='--', label='flat/no optimization')
    plt.xlabel('hour of day')
    plt.ylabel('avg flexible MWh placed')
    plt.title('Where the optimizer moves flexible work to')
    plt.legend()
    plt.savefig(RESULTS_DIR / 'workload_shift_pattern.png')
    plt.close()

    results = {
        'top_20_pct': round(top20_pct, 2),
        'top_50_pct': round(df.iloc[:50]['gain'].sum() / total_gain * 100, 2),
        'top_100_pct': round(df.iloc[:100]['gain'].sum() / total_gain * 100, 2),
        'mean_gain': round(df['gain'].mean(), 2),
        'median_gain': round(df['gain'].median(), 2),
        'top_10_days': df.head(10)[['date', 'gain']].to_dict('records'),
        'avg_hourly_schedule': {str(h): round(avg_schedule[h], 3) for h in range(24)},
    }
    with open(RESULTS_DIR / 'day_concentration_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    df.to_csv(RESULTS_DIR / 'day_level_gain_ranked.csv', index=False)

    print("\nsaved day_concentration_results.json and charts")
    return results


if __name__ == "__main__":
    run()
