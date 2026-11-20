"""Build the Hugging Face files: hf_space/ (the Space) and hf_model/ (model card and model file).

Hugging Face now needs a paid plan for Spaces that run Python (Docker or Gradio); static Spaces are free.
So the Space is a static web page that runs the same model in the visitor's browser:
the 500 trees of models/planning_model.joblib are exported to model.json, the lookup tables to data.json,
and app/static_space/predict.js adds up the trees like scikit-learn does.
Run app/verify_static_space.py afterwards to check that the browser predictions match the Python model.
"""
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
SPACE = ROOT / "hf_space"
MODEL_REPO = ROOT / "hf_model"
MODEL_FILE = ROOT / "models" / "planning_model.joblib"

sys.path.insert(0, str(APP))
import features  # noqa: E402  (shared with the Streamlit and FastAPI apps)

SPACE_README = """---
title: TransitPulse
emoji: 🚌
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: Hourly bus ridership predictions for 24 NYC bus routes
---

# TransitPulse

Predict hourly bus ridership for 24 MTA routes in New York. Pick a route, a date and an hour; the page shows the
predicted riders with a ± range and the predicted profile for the whole day. 2025 dates use recorded weather; later
dates use Open-Meteo's forecast (16 days ahead) or weather you enter. Weather data by Open-Meteo.com (CC BY 4.0).

This is a static Space: the Gradient Boosting model runs in your browser. Its 500 trees were exported from
scikit-learn to `model.json`, and `predict.js` adds them up exactly like scikit-learn; the predictions were
checked to be identical to the Python model.

Model card: https://huggingface.co/ayanjawed/transitpulse-planning-model
Code: https://github.com/ayanmirza99/ML-Project
Course project for CT-378 Machine Learning.
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

- Live demo (Hugging Face Space, runs the model in the browser): https://huggingface.co/spaces/ayanjawed/TransitPulse
- Live demo (Streamlit): https://ml-project-wuwhhjv3aqsnuxh5x5tdwi.streamlit.app/
- Code, data builder and notebooks: https://github.com/ayanmirza99/ML-Project

## Intended use
- What-if planning for the 24 MTA bus routes it was trained on. It was trained on January to September 2025;
  the apps also accept later dates (up to 2027-06-30) with Open-Meteo forecast weather, but those are rough guesses,
  especially for March and October to December, which the model never saw.
- **Not** for other routes or other cities (see Limitations).
- The apps show each prediction with a ± 56 riders range (the validation MAE).

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
3. **Open-Meteo historical weather** (New York City coordinates, one value per hour for the whole city). Licence: CC BY 4.0. Weather data by Open-Meteo.com.
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
The input row must contain the columns listed above (see `app/features.py` in the project repository).
"""


def export_model(model):
    """The pipeline as plain numbers: one-hot categories, numeric column order and every tree node."""
    prep, booster = model.named_steps["prep"], model.named_steps["model"]
    (_, onehot, cat_columns), (_, _, num_columns) = prep.transformers_[:2]
    # The browser code assumes a single regression output and no native categorical splits; stop if that changes.
    assert booster.n_trees_per_iteration_ == 1 and booster.is_categorical_ is None
    assert list(prep.get_feature_names_out())[-len(num_columns):] == [f"num__{c}" for c in num_columns]

    trees = []
    # _predictors and _baseline_prediction are scikit-learn internals (checked with scikit-learn 1.9.1).
    for (predictor,) in booster._predictors:
        nodes = predictor.nodes
        leaf = nodes["is_leaf"].astype(bool)
        trees.append({
            "feature": np.where(leaf, -1, nodes["feature_idx"]).tolist(),
            "threshold": np.where(leaf, 0.0, nodes["num_threshold"]).tolist(),
            "left": nodes["left"].tolist(),
            "right": nodes["right"].tolist(),
            "missing_left": nodes["missing_go_to_left"].tolist(),
            "value": np.where(leaf, nodes["value"], 0.0).tolist(),
        })
    return {
        "baseline": float(booster._baseline_prediction[0][0]),
        "onehot": [[column, [str(c) for c in cats]] for column, cats in zip(cat_columns, onehot.categories_)],
        "numeric": list(num_columns),
        "trees": trees,
    }


def export_data():
    """Lookup tables for the browser, taken from app/features.py so both apps use the same values."""
    route_info = features.routes.reset_index()
    schedule_columns = list(features.schedule.columns)
    first_route = features.ROUTES[0]
    # calendar for every accepted date; weather only for the saved 2025 dates (later dates are fetched from Open-Meteo)
    calendar = {str(day.date()): features.calendar_info(day) for day in pd.date_range(features.FIRST_DATE, features.LAST_DATE)}
    weather = {}
    for day in pd.date_range(features.FIRST_DATE, features.LAST_RECORDED_DATE):
        rows = features.day_features(first_route, day)
        weather[str(day.date())] = {column: rows[column].astype(float).tolist() for column in features.WEATHER_LIMITS}
    hour_sin, hour_cos = rows["hour_sin"].tolist(), rows["hour_cos"].tolist()
    schedule = {
        route: {day_type: features.schedule.loc[route, day_type].sort_index()[schedule_columns].to_numpy().tolist()
                for day_type in ("weekday", "saturday", "sunday")}
        for route in features.ROUTES
    }
    return {
        "route_names": features.ROUTES,
        "routes": {r["bus_route"]: {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if k != "bus_route"}
                   for r in route_info.to_dict("records")},
        "schedule_columns": schedule_columns,
        "schedule": schedule,
        "calendar": calendar,
        "weather": weather,
        "weather_limits": {k: list(v) for k, v in features.WEATHER_LIMITS.items()},
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "first_date": str(features.FIRST_DATE.date()),
        "last_date": str(features.LAST_DATE.date()),
        "last_recorded_date": str(features.LAST_RECORDED_DATE.date()),
        "trained_months": sorted(features.TRAINED_MONTHS),
        "typical_error": features.TYPICAL_ERROR,
        "open_meteo": {"latitude": features.LATITUDE, "longitude": features.LONGITUDE,
                       "forecast_url": features.FORECAST_URL, "archive_url": features.ARCHIVE_URL,
                       "forecast_days": features.FORECAST_DAYS, "archive_delay_days": features.ARCHIVE_DELAY_DAYS},
    }


def write_json(path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    for folder in (SPACE, MODEL_REPO):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()

    model = joblib.load(MODEL_FILE)
    shutil.copy(APP / "static_space" / "index.html", SPACE / "index.html")
    shutil.copy(APP / "static_space" / "predict.js", SPACE / "predict.js")
    write_json(SPACE / "model.json", export_model(model))
    write_json(SPACE / "data.json", export_data())
    (SPACE / "README.md").write_text(SPACE_README, encoding="utf-8")

    shutil.copy(MODEL_FILE, MODEL_REPO / "planning_model.joblib")
    (MODEL_REPO / "README.md").write_text(MODEL_CARD, encoding="utf-8")

    print("built", SPACE, "and", MODEL_REPO)
