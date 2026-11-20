import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
SPACE = ROOT / "hf_space"
MODEL_REPO = ROOT / "hf_model"

# Hugging Face retired the built-in Streamlit SDK on 2025-04-30, so the Space runs Streamlit inside Docker.
SPACE_README = """---
title: TransitPulse
emoji: 🚌
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
tags:
  - streamlit
pinned: false
---

# TransitPulse

Predict hourly bus ridership for 24 MTA routes in New York (January to September 2025).
Pick a route, a date and an hour; the app shows the predicted riders and the predicted profile for the whole day.

Model card: see the model repository. Course project for CT-378 Machine Learning.
"""

SPACE_REQUIREMENTS = """streamlit==1.67.0
pandas==3.1.1
numpy==2.5.4
scikit-learn==1.9.1
joblib==1.6.0
holidays==0.108
"""

# Python 3.14 matches the local .venv the app was tested with (numpy 2.5 needs Python 3.12 or newer).
# Spaces run the container as user 1000, so the app gets its own home folder (Streamlit writes ~/.streamlit).
SPACE_DOCKERFILE = """FROM python:3.14-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
"""

MODEL_CARD = """---
library_name: sklearn
tags:
  - tabular-regression
  - time-series
  - transportation
license: other
---

# TransitPulse planning model

Predicts **hourly bus ridership for one route** (riders per route-hour) from information known in advance:
route, date, hour, schedule and weather. Built for transport planners as a CT-378 Machine Learning course project.

## Intended use
- What-if planning for the 24 MTA bus routes it was trained on, for dates from 2025-01-06 to 2025-09-28.
- **Not** for other routes, other cities or dates outside that period (see Limitations).

## Model
`sklearn.pipeline.Pipeline`: one-hot encoding of `bus_route` and `borough`, then `HistGradientBoostingRegressor`
(learning_rate 0.05, max_iter 500, max_leaf_nodes 63). Selected by validation MAE among models whose saved file is under 50 MB;
a Random Forest was about as accurate but about 110 MB.
Input columns: bus_route, borough, gtfs_n_stops, gtfs_max_stops_per_trip, sched_trips_hr, sched_headway_min, sched_trips_day,
temperature_2m, precipitation, snowfall, wind_speed_10m, hour, day_of_week, is_weekend, hour_sin, hour_cos, is_am_peak, is_pm_peak,
is_holiday, is_school_day.

## Data (three sources plus calendar)
1. **MTA Bus Hourly Ridership: Beginning 2025**, NY Open Data (dataset id `gxb3-akrn`), summed to one row per route per hour.
   Licence: the dataset page gives no licence information; it is covered by NY Open Data's terms of use.
2. **MTA GTFS static bus schedules** (mta.info/developers). A 2026 snapshot, not the 2025 service that produced the ridership.
3. **Open-Meteo historical weather** (New York City coordinates, one value per hour for the whole city).
4. US federal holidays (Python `holidays` package) and a hand-typed, approximate NYC school calendar.

Eight two-week windows between January and September 2025, 24 routes, 64,512 route-hour rows. Missing hours are filled with 0 riders.
Data are from New York because Karachi has no public passenger-count data.

## Evaluation
Time-ordered split by whole days: train 2025-01-13 to 07-14, validation 07-15 to 09-04, test 09-05 to 09-28 (17 days, scored once).
The final model is refit on train plus validation before testing.

| | MAE (riders) | RMSE | R2 |
|---|---|---|---|
| Planning model (this model) | 65.4 | 113.3 | 0.92 |
| Baseline: training average for the route, hour and weekday/weekend | 78.0 | 129.4 | 0.89 |

The average route-hour in the test period has 450 riders, so the error is about 14.5%.
A separate history model (uses recent ridership) reaches MAE 39.0 but needs recent data, so it is not deployed.
Other checks: leave-one-window-out MAE 53.0 vs baseline 68.9; unseen routes MAE 111 vs baseline 132.

## Limitations
- Only 24 routes and eight 2-week windows in one year; the test set is only 17 days, so small differences are not meaningful.
- Some routes changed level during the year (Q58, Q10, Q65 about 35% lower; Q25 about 32% higher from summer). The model does not know about service changes; its error on these routes is about three times larger.
- Weather helps a little, but partly because it hints at the season. It is not evidence that weather causes ridership changes.
- Weather was one city-wide value, and the test used recorded (not forecast) weather, so real use will have higher error.
- Schedule data is a 2026 snapshot; the school calendar is approximate; licence wording was not independently checked.
- New York data only; no claim is made about Karachi.

## How to load
```python
import joblib
model = joblib.load("planning_model.joblib")  # needs scikit-learn 1.9.1
```
The input row must contain the columns listed above (see `features.py` in the project repository).
"""

for folder in (SPACE, MODEL_REPO):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()

shutil.copy(APP / "streamlit_app.py", SPACE / "app.py")
shutil.copy(APP / "features.py", SPACE / "features.py")
shutil.copytree(APP / "lookup", SPACE / "lookup")
shutil.copy(ROOT / "models" / "planning_model.joblib", SPACE / "planning_model.joblib")
(SPACE / "README.md").write_text(SPACE_README, encoding="utf-8")
(SPACE / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
(SPACE / "Dockerfile").write_text(SPACE_DOCKERFILE, encoding="utf-8")

shutil.copy(ROOT / "models" / "planning_model.joblib", MODEL_REPO / "planning_model.joblib")
(MODEL_REPO / "README.md").write_text(MODEL_CARD, encoding="utf-8")

print("built", SPACE, "and", MODEL_REPO)
