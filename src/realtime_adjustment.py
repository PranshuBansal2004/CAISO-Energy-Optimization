"""
realtime_adjustment.py

This is Scenario 3 - the real time adjustment algorithm.

Basic idea from talking with professor:
- We already locked in a day ahead plan (Scenario 2), that money is spent
  no matter what happens later
- If real time price is higher than what we paid, we can sell back power
  we dont need (defer that flexible work to later) and make a profit
- If real time price is lower, we can buy extra power, but we only get
  the cheap price on the EXTRA units, not our whole hourly usage
- We cant sell back more than what we already bought in day ahead

Two versions here:
1. "ideal" - solves the whole day at once assuming we know all of todays
   real time prices (this isolates the value of the buy/sell mechanism)
2. "rolling" - the realistic version, only knows the current hour's real
   price, assumes future hours will match day ahead price (best guess)
"""

import json
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_utils import load_optimization_data, RESULTS_DIR
from src.optimize import TOTAL_DAILY_WORK, CAPACITY, FLEX_RATIOS, run_scenario1, run_scenario2, run_scenario4


def solve_safe(prob):
    # ECOS sometimes fails to converge at higher flex levels, so we
    # fall back to other solvers if it errors out
    for solver in [cp.ECOS, cp.CLARABEL, cp.SCS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ('optimal', 'optimal_inaccurate') and prob.value is not None:
                return True
        except Exception:
            continue
    return False


def get_da_plan(prices_da, rigid_per_hour, flex_work, capacity):
    # this is just scenario 2, gives us the locked in purchase per hour
    f = cp.Variable(24)
    x = rigid_per_hour + f
    constraints = [cp.sum(f) == flex_work, f >= 0, x <= capacity, x >= 0]
    prob = cp.Problem(cp.Minimize(prices_da @ x), constraints)
    if not solve_safe(prob):
        return None
    return rigid_per_hour + f.value  # this is the committed purchase per hour


def scenario3_ideal(prices_da, prices_rt, rigid_per_hour, flex_work, capacity):
    da_commit = get_da_plan(prices_da, rigid_per_hour, flex_work, capacity)
    if da_commit is None:
        return None

    f_actual = cp.Variable(24)
    x_actual = rigid_per_hour + f_actual
    rt_purchase = x_actual - da_commit  # + means buying extra, - means selling back

    constraints = [
        cp.sum(f_actual) == flex_work,
        f_actual >= 0,
        x_actual <= capacity,
        x_actual >= 0,
        rt_purchase >= -da_commit,  # cant sell back more than we bought
    ]

    prob = cp.Problem(cp.Minimize(prices_rt @ rt_purchase), constraints)
    if not solve_safe(prob):
        return None

    da_cost = float(prices_da @ da_commit)
    rt_cost = float(prices_rt @ rt_purchase.value)
    return da_cost + rt_cost


def scenario3_rolling(prices_da, prices_rt, rigid_per_hour, flex_work, capacity):
    da_commit = get_da_plan(prices_da, rigid_per_hour, flex_work, capacity)
    if da_commit is None:
        return None

    actual_flex = np.zeros(24)
    remaining = flex_work

    for t in range(24):
        left = 24 - t
        if left == 1:
            actual_flex[t] = max(0, min(remaining, capacity - rigid_per_hour))
            remaining -= actual_flex[t]
            continue

        f_rem = cp.Variable(left)
        x_rem = rigid_per_hour + f_rem

        # current hour: we know the real price. future hours: assume = day ahead
        rt_guess = np.zeros(left)
        rt_guess[0] = prices_rt[t]
        rt_guess[1:] = prices_da[t+1:]

        da_commit_remaining = da_commit[t:]
        rt_purchase = x_rem - da_commit_remaining

        constraints = [
            cp.sum(f_rem) == remaining,
            f_rem >= 0,
            x_rem <= capacity,
            x_rem >= 0,
            rt_purchase >= -da_commit_remaining,
        ]

        prob = cp.Problem(cp.Minimize(rt_guess @ rt_purchase), constraints)

        if solve_safe(prob):
            actual_flex[t] = max(0, f_rem.value[0])
        else:
            actual_flex[t] = max(0, min(da_commit[t] - rigid_per_hour, remaining))

        remaining = max(0, remaining - actual_flex[t])

    actual_power = rigid_per_hour + actual_flex
    rt_purchase = np.maximum(actual_power - da_commit, -da_commit)

    total_cost = float(np.sum(prices_da * da_commit) + np.sum(prices_rt * rt_purchase))
    return total_cost


def run():
    da, rt_hourly, days = load_optimization_data()
    print(f"running scenario 3 on {len(days)} days")

    all_results = []

    for flex_ratio in FLEX_RATIOS:
        flex_work = TOTAL_DAILY_WORK * flex_ratio
        rigid_per_hour = (TOTAL_DAILY_WORK - flex_work) / 24

        s1_list, s2_list, s3i_list, s3r_list, s4_list = [], [], [], [], []

        for day in days:
            d_da = da[da['date'] == day].sort_values('Time_UTC')
            d_rt = rt_hourly[rt_hourly['date'] == day].sort_values('Time_UTC')
            if len(d_rt) < 24:
                continue

            pda = d_da['LMP'].to_numpy(dtype=float)
            prt = d_rt['LMP_RT'].to_numpy(dtype=float)

            c1 = run_scenario1(pda, TOTAL_DAILY_WORK)
            c2 = run_scenario2(pda, rigid_per_hour, flex_work, CAPACITY)
            c3i = scenario3_ideal(pda, prt, rigid_per_hour, flex_work, CAPACITY)
            c3r = scenario3_rolling(pda, prt, rigid_per_hour, flex_work, CAPACITY)
            c4 = run_scenario4(pda, prt, rigid_per_hour, flex_work, CAPACITY)

            s1_list.append(c1)
            s2_list.append(c2 if c2 is not None else c1)
            s3i_list.append(c3i if c3i is not None else c1)
            s3r_list.append(c3r if c3r is not None else c1)
            s4_list.append(c4 if c4 is not None else c1)

        a1 = np.mean(s1_list)
        a2 = np.mean(s2_list)
        a3i = np.mean(s3i_list)
        a3r = np.mean(s3r_list)
        a4 = np.mean(s4_list)

        result = {
            'flex': int(flex_ratio * 100),
            'savings_da_pct': round((a1 - a2) / a1 * 100, 2),
            'savings_s3_ideal_pct': round((a1 - a3i) / a1 * 100, 2),
            'savings_s3_rolling_pct': round((a1 - a3r) / a1 * 100, 2),
            'savings_perfect_pct': round((a1 - a4) / a1 * 100, 2),
        }
        all_results.append(result)
        print(f"flex {result['flex']}%: DA={result['savings_da_pct']}% "
              f"s3ideal={result['savings_s3_ideal_pct']}% "
              f"s3rolling={result['savings_s3_rolling_pct']}% "
              f"perfect={result['savings_perfect_pct']}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'scenario3_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    flexes = [r['flex'] for r in all_results]
    plt.figure(figsize=(10, 6))
    plt.plot(flexes, [0]*len(flexes), 'k--', label='scenario 1 baseline')
    plt.plot(flexes, [r['savings_da_pct'] for r in all_results], 'b-o', label='scenario 2: day ahead only')
    plt.plot(flexes, [r['savings_s3_rolling_pct'] for r in all_results], 'g-^', label='scenario 3: rolling (realistic)')
    plt.plot(flexes, [r['savings_perfect_pct'] for r in all_results], 'r-s', label='scenario 4: perfect foresight')
    plt.xlabel('flexible workload %')
    plt.ylabel('savings %')
    plt.title('all scenarios compared')
    plt.legend()
    plt.savefig(RESULTS_DIR / 'all_scenarios_corrected.png')
    plt.close()

    print("saved scenario3_results.json and chart")
    return all_results


if __name__ == "__main__":
    run()
