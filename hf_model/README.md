---
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

- Live demo (Hugging Face Space): https://huggingface.co/spaces/ayanjawed/TransitPulse
- Code, data builder and notebooks: https://github.com/ayanmirza99/ML-Project

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
