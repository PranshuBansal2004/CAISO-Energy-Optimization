# Adaptive Computation Scheduling for Data Centers

Research project for UIC CS with Professor Ian Kash.

## What this is about

Electricity prices change every hour in California, cheap when solar is
out, expensive in the evening. Data centers use a lot of power but some
of their work (like batch jobs, training, backups) can wait a few hours.
So the question is: how much money can you save by shifting flexible
work to cheaper hours? And is it worth building a real time price
response system, or is just planning the day ahead good enough?

## Data

Real CAISO electricity price data for the DLAP_SDGE node (SDG&E area,
covers San Diego where SDSC is). Full year 2024.

- day ahead prices: hourly, 8784 rows
- real time prices: every 15 min, 35136 rows

## Setup

```
git clone https://github.com/PranshuBansal2004/CAISO-Energy-Optimization.git
cd CAISO-Energy-Optimization
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run everything:
```
python run_all.py
```

Note: this takes like 15-20 minutes because scenario 3 solves a ton of
small LPs (24 per day x 364 days x 7 flexibility levels).

Or run pieces individually:
```
python -m src.optimize                     # main scenarios 1, 2, 4
python -m src.seasonal_analysis
python -m src.negative_price_analysis
python -m src.hourly_profile
python -m src.gap_analysis                  # where the 17% gap comes from
python -m src.day_concentration_analysis    # concentrated vs spread out
python -m src.realtime_adjustment           # scenario 3
python -m src.price_forecast                # ML models
python -m src.generate_plots
```

All results go into `results/`.

## Folder structure

```
data/                 the two CSV files
src/
  data_utils.py        shared loading + timezone stuff
  optimize.py           scenarios 1, 2, 4
  seasonal_analysis.py
  negative_price_analysis.py
  hourly_profile.py
  gap_analysis.py         where does the 17% gap come from (hour/month/season)
  day_concentration_analysis.py   is it a few big days or spread out
  realtime_adjustment.py    scenario 3, the real time adjustment algo
  price_forecast.py         ML forecasting for real time prices
  generate_plots.py
results/              generated charts and json files
```

## The 4 scenarios

1. baseline - just spread work evenly, dont care about prices
2. day ahead only - plan using tomorrows known prices, stick to the plan
3. real time adjustment - start with the day ahead plan but adjust as
   real time prices come in (can sell back power we dont need, buy more
   if its cheap)
4. perfect foresight - pretend we know everything in advance, gives us
   the theoretical upper bound

## What we found so far

- at 30% flexibility, day ahead planning alone saves about 22%, perfect
  foresight saves about 39%
- the ~17% gap between those two stays roughly the same no matter how
  much flexibility you have
- the gap is mostly coming from summer evenings (6-7pm), hour 19 alone
  is about 35% of the total gap
- it's NOT just a few crazy days driving everything - top 20 days only
  account for ~22% of the total yearly gain, so it's fairly spread out
- the optimizer consistently shifts flexible work into the 9am-2pm
  window (solar hours) and avoids 4pm-11pm
- for forecasting real time prices, just using the day ahead price as
  your guess is honestly a pretty strong baseline, XGBoost and LightGBM
  only did slightly better

