from __future__ import annotations
"""Analyze the frequency and timing of negative electricity prices."""

import json

from src.data_utils import RESULTS_DIR, load_day_ahead, load_real_time


def analyze_negative_prices() -> dict:
    day_ahead = load_day_ahead()
    real_time = load_real_time()
    da_negative = day_ahead[day_ahead["LMP"] < 0]
    rt_negative = real_time[real_time["LMP"] < 0]

    negative_by_hour = (
        da_negative.groupby("hour").size()
        .div(day_ahead.groupby("hour").size())
        .mul(100)
        .reindex(range(24), fill_value=0)
        .round(2)
    )
    negative_by_month = (
        da_negative.groupby("month").size()
        .div(day_ahead.groupby("month").size())
        .mul(100)
        .reindex(range(1, 13), fill_value=0)
        .round(2)
    )

    revenue_per_mw = float(-da_negative["LMP"].sum())
    results = {
        "timezone": "America/Los_Angeles",
        "day_ahead": {
            "total_hours": len(day_ahead),
            "negative_hours": len(da_negative),
            "negative_pct": round(len(da_negative) / len(day_ahead) * 100.0, 2),
            "avg_negative_price": round(float(da_negative["LMP"].mean()), 2),
            "min_price": round(float(day_ahead["LMP"].min()), 2),
            "revenue_at_1mw": round(revenue_per_mw, 2),
        },
        "real_time": {
            "total_intervals": len(real_time),
            "negative_intervals": len(rt_negative),
            "negative_pct": round(len(rt_negative) / len(real_time) * 100.0, 2),
            "avg_negative_price": round(float(rt_negative["LMP"].mean()), 2),
            "min_price": round(float(real_time["LMP"].min()), 2),
        },
        "negative_by_hour": {str(k): float(v) for k, v in negative_by_hour.items()},
        "negative_by_month": {str(k): float(v) for k, v in negative_by_month.items()},
        "free_computation": {
            "hours_available": len(da_negative),
            "mwh_at_1mw": len(da_negative),
            "revenue_at_1mw": round(revenue_per_mw, 2),
        },
    }

    print("Negative price analysis")
    print(
        f"Day-ahead: {len(da_negative):,}/{len(day_ahead):,} hours "
        f"({results['day_ahead']['negative_pct']}%)"
    )
    print(
        f"Real-time: {len(rt_negative):,}/{len(real_time):,} intervals "
        f"({results['real_time']['negative_pct']}%)"
    )
    print(
        f"At 1 MW, all negative day-ahead hours represent "
        f"{len(da_negative):,} MWh and ${revenue_per_mw:,.2f} in negative-price revenue."
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "negative_price_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {path.relative_to(RESULTS_DIR.parent)}")
    return results


if __name__ == "__main__":
    analyze_negative_prices()
