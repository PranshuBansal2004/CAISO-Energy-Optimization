# runs everything in order
# takes a while especially scenario 3 (~15-20 min) because it solves
# a lot of LPs

from src.optimize import run_all as run_optimization
from src.seasonal_analysis import run_seasonal
from src.negative_price_analysis import analyze_negative_prices
from src.hourly_profile import analyze_hourly_profile
from src.gap_analysis import analyze_gap
from src.day_concentration_analysis import run as run_day_concentration
from src.realtime_adjustment import run as run_scenario3
from src.price_forecast import run as run_forecast
from src.generate_plots import generate_all_plots


def main():
    print("step 1: main optimization (scenario 1, 2, 4)")
    run_optimization()

    print("\nstep 2: seasonal analysis")
    run_seasonal()

    print("\nstep 3: negative price analysis")
    analyze_negative_prices()

    print("\nstep 4: hourly price profile")
    analyze_hourly_profile()

    print("\nstep 5: gap analysis - where does the 17% come from")
    analyze_gap()

    print("\nstep 6: day concentration - few extreme days or spread out?")
    run_day_concentration()

    print("\nstep 7: scenario 3 - real time adjustment algorithm")
    run_scenario3()

    print("\nstep 8: price forecasting models")
    run_forecast()

    print("\nstep 9: generating all the charts")
    generate_all_plots()

    print("\ndone, check results/ folder")


if __name__ == "__main__":
    main()
