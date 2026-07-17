from __future__ import annotations
"""Generate all project figures in results/."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_utils import (
    RESULTS_DIR,
    aggregate_real_time_hourly,
    load_day_ahead,
    load_real_time,
)

SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall": [9, 10, 11],
}


def _save(name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.relative_to(RESULTS_DIR.parent)}")


def plot_savings_curve() -> None:
    path = RESULTS_DIR / "optimization_results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    flexes = [row["flex"] for row in results]
    savings_da = [row["savings_da_pct"] for row in results]
    savings_perfect = [row["savings_perfect_pct"] for row in results]

    plt.figure(figsize=(10, 6))
    plt.plot(flexes, [0] * len(flexes), "k--", linewidth=1.5, label="Scenario 1: Baseline")
    plt.plot(flexes, savings_da, "o-", linewidth=2.5, label="Scenario 2: Day Ahead")
    plt.plot(flexes, savings_perfect, "s-", linewidth=2.5, label="Scenario 4: Perfect Foresight")
    plt.fill_between(flexes, savings_da, alpha=0.15)
    plt.fill_between(flexes, savings_da, savings_perfect, alpha=0.15)
    plt.xlabel("Flexible workload (%)")
    plt.ylabel("Electricity cost savings (%)")
    plt.title("Cost Savings vs Workload Flexibility\nCAISO DLAP_SDGE, 2024")
    plt.xticks(flexes, [f"{value}%" for value in flexes])
    plt.grid(alpha=0.3)
    plt.legend()
    _save("savings_curve.png")


def plot_duck_curve() -> None:
    day_ahead = load_day_ahead()
    real_time_hourly = aggregate_real_time_hourly(load_real_time())
    july_days = sorted(day_ahead.loc[day_ahead["month"] == 7, "date"].unique())
    sample_day = july_days[10]
    da_day = day_ahead[day_ahead["date"] == sample_day].sort_values("Time_UTC")
    rt_day = real_time_hourly[real_time_hourly["date"] == sample_day].sort_values("Time_UTC")

    plt.figure(figsize=(10, 5))
    plt.plot(range(24), da_day["LMP"].to_numpy(), "o-", linewidth=2, label="Day Ahead")
    plt.plot(range(24), rt_day["LMP_RT"].to_numpy(), "s--", linewidth=2, label="Real Time")
    plt.axhline(0, linewidth=1, alpha=0.4)
    plt.xlabel("Pacific local hour")
    plt.ylabel("Price ($/MWh)")
    plt.title(f"Day-Ahead and Real-Time Prices — {sample_day}")
    plt.xticks(range(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.legend()
    _save("duck_curve_sample.png")


def plot_monthly_prices() -> None:
    day_ahead = load_day_ahead()
    monthly = day_ahead.groupby("month")["LMP"].mean()
    plt.figure(figsize=(10, 5))
    plt.bar(monthly.index, monthly.values, alpha=0.8)
    plt.xlabel("Month")
    plt.ylabel("Average day-ahead price ($/MWh)")
    plt.title("Monthly Average Day-Ahead Prices — DLAP_SDGE 2024")
    plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.grid(alpha=0.3, axis="y")
    _save("monthly_prices.png")


def plot_seasonal_savings() -> None:
    results = json.loads((RESULTS_DIR / "seasonal_results.json").read_text(encoding="utf-8"))
    seasons = list(SEASONS)
    savings_da = [results[name]["savings_da_pct"] for name in seasons]
    savings_perfect = [results[name]["savings_perfect_pct"] for name in seasons]
    x = np.arange(len(seasons))
    width = 0.36

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, savings_da, width, label="Day Ahead")
    plt.bar(x + width / 2, savings_perfect, width, label="Perfect Foresight")
    plt.xlabel("Season")
    plt.ylabel("Electricity cost savings (%)")
    plt.title("Seasonal Savings at 30% Workload Flexibility")
    plt.xticks(x, seasons)
    plt.grid(alpha=0.3, axis="y")
    plt.legend()
    _save("seasonal_savings.png")


def plot_hourly_profile_by_season() -> None:
    day_ahead = load_day_ahead()
    plt.figure(figsize=(10, 6))
    for season_name, months in SEASONS.items():
        hourly = day_ahead[day_ahead["month"].isin(months)].groupby("hour")["LMP"].mean()
        plt.plot(hourly.index, hourly.values, "o-", linewidth=2, label=season_name)
    plt.axhline(0, linewidth=1, alpha=0.4)
    plt.xlabel("Pacific local hour")
    plt.ylabel("Average day-ahead price ($/MWh)")
    plt.title("Average Hourly Price Profile by Season")
    plt.xticks(range(0, 24, 2))
    plt.grid(alpha=0.3)
    plt.legend()
    _save("hourly_profile_seasonal.png")


def plot_negative_price_hours() -> None:
    day_ahead = load_day_ahead()
    negative = day_ahead[day_ahead["LMP"] < 0]
    percentages = (
        negative.groupby("hour").size()
        .div(day_ahead.groupby("hour").size())
        .mul(100)
        .reindex(range(24), fill_value=0)
    )
    plt.figure(figsize=(10, 5))
    plt.bar(range(24), percentages.values, alpha=0.8)
    plt.xlabel("Pacific local hour")
    plt.ylabel("Negative-price observations (%)")
    plt.title("When Do Negative Day-Ahead Prices Occur?")
    plt.xticks(range(24))
    plt.grid(alpha=0.3, axis="y")
    _save("negative_price_hours.png")


def plot_price_distribution() -> None:
    day_ahead = load_day_ahead()
    negative = day_ahead.loc[day_ahead["LMP"] < 0, "LMP"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(day_ahead["LMP"], bins=100, alpha=0.8)
    axes[0].axvline(0, linestyle="--", alpha=0.6)
    axes[0].set_xlabel("Price ($/MWh)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Day-Ahead Price Distribution")
    axes[0].set_xlim(-80, 200)
    axes[1].hist(negative, bins=50, alpha=0.8)
    axes[1].set_xlabel("Price ($/MWh)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Negative Prices")
    figure.suptitle("DLAP_SDGE Day-Ahead Prices, 2024")
    _save("price_distribution.png")


def generate_all_plots() -> None:
    plot_savings_curve()
    plot_duck_curve()
    plot_monthly_prices()
    plot_seasonal_savings()
    plot_hourly_profile_by_season()
    plot_negative_price_hours()
    plot_price_distribution()


if __name__ == "__main__":
    generate_all_plots()
