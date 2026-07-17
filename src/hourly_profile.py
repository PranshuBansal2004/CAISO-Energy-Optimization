from __future__ import annotations
"""Calculate average local-hour electricity price profiles."""

import json

from src.data_utils import RESULTS_DIR, load_day_ahead

SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall": [9, 10, 11],
}


def analyze_hourly_profile() -> dict:
    day_ahead = load_day_ahead()
    overall = day_ahead.groupby("hour")["LMP"].agg(["mean", "std", "min", "max"])

    results: dict[str, dict] = {
        "timezone": "America/Los_Angeles",
        "overall": {
            statistic: {str(k): float(v) for k, v in values.round(2).items()}
            for statistic, values in overall.items()
        },
    }

    for season_name, months in SEASONS.items():
        profile = (
            day_ahead[day_ahead["month"].isin(months)]
            .groupby("hour")["LMP"]
            .mean()
            .round(2)
        )
        results[season_name] = {str(k): float(v) for k, v in profile.items()}

    means = {int(k): v for k, v in results["overall"]["mean"].items()}
    cheapest_hour = min(means, key=means.get)
    expensive_hour = max(means, key=means.get)
    results["summary"] = {
        "cheapest_hour": cheapest_hour,
        "cheapest_price": means[cheapest_hour],
        "most_expensive_hour": expensive_hour,
        "most_expensive_price": means[expensive_hour],
        "average_hourly_spread": round(means[expensive_hour] - means[cheapest_hour], 2),
    }

    print(
        f"Cheapest average hour: {cheapest_hour:02d}:00 "
        f"(${means[cheapest_hour]:.2f}/MWh)"
    )
    print(
        f"Most expensive average hour: {expensive_hour:02d}:00 "
        f"(${means[expensive_hour]:.2f}/MWh)"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "hourly_profile_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {path.relative_to(RESULTS_DIR.parent)}")
    return results


if __name__ == "__main__":
    analyze_hourly_profile()
