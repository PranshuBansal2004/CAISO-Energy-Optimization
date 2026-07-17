from __future__ import annotations
"""Download CAISO DLAP_SDGE day-ahead and real-time price data."""

from pathlib import Path

import gridstatus
import pandas as pd

from src.data_utils import DATA_DIR

LOCATION = "DLAP_SDGE-APND"


def _monthly_ranges(year: int):
    for month in range(1, 13):
        start = pd.Timestamp(year=year, month=month, day=1)
        end = pd.Timestamp(year + 1, 1, 1) if month == 12 else pd.Timestamp(year, month + 1, 1)
        yield start, end


def _pull_market(year: int, market: str, output_name: str) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    caiso = gridstatus.CAISO()
    months = []

    for start, end in _monthly_ranges(year):
        try:
            data = caiso.get_lmp(
                start=start,
                end=end,
                market=market,
                locations=[LOCATION],
                sleep=3,
            )
            months.append(data)
            print(f"{market} {start:%Y-%m}: {len(data):,} rows")
        except Exception as error:
            print(f"{market} {start:%Y-%m}: failed — {error}")

    if not months:
        raise RuntimeError(f"No data was downloaded for {market}.")

    combined = pd.concat(months, ignore_index=True)
    path = DATA_DIR / output_name
    combined.to_csv(path, index=False)
    print(f"Saved {len(combined):,} rows to {path}")
    return combined


def pull_day_ahead(year: int = 2024) -> pd.DataFrame:
    return _pull_market(year, "DAY_AHEAD_HOURLY", f"sdge_day_ahead_{year}_full.csv")


def pull_real_time(year: int = 2024) -> pd.DataFrame:
    return _pull_market(year, "REAL_TIME_15_MIN", f"sdge_realtime_{year}_full.csv")


if __name__ == "__main__":
    pull_day_ahead(2024)
    pull_real_time(2024)
