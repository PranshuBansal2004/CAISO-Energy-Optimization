from __future__ import annotations
"""Shared data loading and time-zone handling utilities."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PACIFIC_TIMEZONE = "America/Los_Angeles"

DAY_AHEAD_FILE = DATA_DIR / "sdge_day_ahead_2024_full.csv"
REAL_TIME_FILE = DATA_DIR / "sdge_realtime_2024_full.csv"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            "Use the CSV files included in data/ or run: python src/pull_data.py"
        )

    data = pd.read_csv(path)
    required_columns = {"Time", "LMP"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    data["Time_UTC"] = pd.to_datetime(data["Time"], utc=True)
    data["Time"] = data["Time_UTC"].dt.tz_convert(PACIFIC_TIMEZONE)
    data["date"] = data["Time"].dt.date
    data["hour"] = data["Time"].dt.hour
    data["month"] = data["Time"].dt.month
    return data


def load_day_ahead() -> pd.DataFrame:
    """Load hourly day-ahead prices in Pacific local time."""
    return _load_csv(DAY_AHEAD_FILE)


def load_real_time() -> pd.DataFrame:
    """Load 15-minute real-time prices in Pacific local time."""
    return _load_csv(REAL_TIME_FILE)


def aggregate_real_time_hourly(real_time: pd.DataFrame) -> pd.DataFrame:
    """Aggregate four 15-minute real-time intervals into unique hourly values."""
    hourly = real_time.copy()
    hourly["hour_start_utc"] = hourly["Time_UTC"].dt.floor("h")
    hourly = hourly.groupby("hour_start_utc", as_index=False)["LMP"].mean()
    hourly = hourly.rename(columns={"LMP": "LMP_RT", "hour_start_utc": "Time_UTC"})
    hourly["Time"] = hourly["Time_UTC"].dt.tz_convert(PACIFIC_TIMEZONE)
    hourly["date"] = hourly["Time"].dt.date
    hourly["hour"] = hourly["Time"].dt.hour
    hourly["month"] = hourly["Time"].dt.month
    return hourly


def load_optimization_data():
    """Load both markets and return only dates containing 24 hourly observations."""
    day_ahead = load_day_ahead()
    real_time = load_real_time()
    real_time_hourly = aggregate_real_time_hourly(real_time)

    da_counts = day_ahead.groupby("date").size()
    rt_counts = real_time_hourly.groupby("date").size()

    complete_da = set(da_counts[da_counts == 24].index)
    complete_rt = set(rt_counts[rt_counts == 24].index)
    complete_days = sorted(complete_da.intersection(complete_rt))

    return day_ahead, real_time_hourly, complete_days
