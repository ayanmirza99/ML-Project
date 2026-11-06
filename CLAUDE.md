# CLAUDE.md: TransitPulse project knowledge

Read this first. It summarises the course rules, the project decisions, the data pipeline, known issues and the plan, so a new session can continue without re-deriving anything.

## 1. Project in one paragraph
**TransitPulse** predicts **hourly bus ridership per route** (regression) for transport planners. Course: **CT-378 Machine Learning**, Fall 2026, semester project (10 sessional marks). Group of 2-3. Data is from **New York (MTA)** because Karachi has no public passenger-count data; this must be stated as a limitation in the report.

## 2. Course guidelines (CT-378 Semester Project)
- Milestones: proposal in **Week 12** (half page: problem, data source, target variable, success metric); final demo, report and viva in **Week 15**.
- Problem: classification, regression, clustering or time-series with a clear user. **No duplicate topics across groups; first-come approval.** Banned: Iris, Titanic, MNIST, Boston Housing, or any already-used dataset.
- Data: collect it ourselves **or combine at least two public sources**. **Minimum 5,000 rows and 15 features.** Document source, collection method, licence and date for every dataset. Remove personal identifiers.
- ML lifecycle (all required):
  1. EDA: distributions, missing values, outliers, correlations, class balance; **at least 5 plots**.
  2. Pre-processing (imputation, encoding, scaling, feature engineering) **inside a scikit-learn Pipeline fitted on training data only** (no leakage).
  3. Train/validation/test split (e.g. 70/15/15); **time-ordered split for time series**.
  4. A simple **baseline plus at least three algorithms** from the course (KNN, Linear/Logistic Regression, SVM, Random Forest, Boosting, ...).
  5. Tuning: **5-fold cross-validation** with GridSearchCV or RandomizedSearchCV.
  6. Evaluation with task-appropriate metrics (regression: MAE, RMSE, R^2), **reported once on the held-out test set**; discuss over/under-fitting and errors.
  7. Save the final pipeline with **joblib**; pin versions in `requirements.txt`.
- Deployment: the table says "choose any one" (Streamlit on Community Cloud with a public URL; Hugging Face model card + Space; or local Flask/FastAPI with a `/predict` endpoint, HTML front end, input validation and error messages). **But the overview and marking scheme say all three must work live, and a deployment that fails live scores zero for that platform. This contradiction is unresolved; ask the instructor, and plan for all three.**
- Deliverables: GitHub repo (`data/` or download script, `notebooks/`, `app/`, `models/`, `requirements.txt`, README with setup steps and all three deployment links); PDF report, 3-4 pages (Abstract, Problem and motivation, Data collection, EDA, Methodology, Results with comparison table and plots, Deployment screenshots, Limitations and future work, References, Individual contribution table); 10-minute demo (3 min problem and data, 4 min live demo, 3 min results) then viva.
- Marking: Demonstration 4 (lifecycle quality 2, all three deployments live 2); Report 3; Viva 3 (individual, every member must explain any part of the code). Copied code/reports/datasets without citation = zero for the whole project. Late submission loses 1 mark per day.

## 3. Topics already taken by other groups (avoid)
Tormenta (geomagnetic Kp from solar wind); KaarWala (used-car prices Pakistan); Student academic performance/dropout; Weather-based rainfall prediction; SunPulse (solar power generation); HeatWave AI (Karachi heat warnings); HunarLens (youth unemployment/skills gap); Grid Reality (Formula 1 outcomes); Karachi and Lahore PM2.5 forecaster; PSX stock direction predictor; Climate-resilient low-water crop selection; CineScore (IMDb rating before release).

## 4. Data sources (three, plus calendar tables)
**Source 1: MTA Bus Hourly Ridership: Beginning 2025** (NY Open Data, Socrata dataset id `gxb3-akrn`).
- API: `https://data.ny.gov/resource/gxb3-akrn.csv` with `$select/$where/$group/$order/$limit`; never download the full CSV (tens of GB).
- Fields used: `bus_route`, `transit_timestamp`, `ridership` (summed over payment method and fare class to one row per route per hour).
- Licence: the dataset page says no licence information was provided and that it is covered by different Terms of Use than Data.gov; cite the NY Open Data terms.
- Other related ids (not used): bus 2020-2024 `kv7t-n8in`; subway hourly `5wq4-mkjj` and `wujg-7c2s`; bus stop level `fvdm-uavx`.

**Source 2: MTA GTFS static bus schedule** (mta.info/developers; six feeds: Bronx, Brooklyn, Manhattan, Queens, Staten Island, MTA Bus Co). Zips are stored in `gtfs_zips/` and reused on later runs (route counts parsed: 46, 68, 41, 40, 58, 92).
- Gives scheduled trips per route per hour (weekday / Saturday / Sunday), headway, number of stops, a route-length proxy and directions.
- Services are resolved for **exact dates** (calendar.txt + calendar_dates.txt exceptions) over 8 sample dates per day type; the **median** date is used. Trips are counted at first-stop departure, **both directions together**; `sched_headway_min` is per direction (capped at 120 min when no trips).
- Caveat: the feed is a **snapshot of the schedule valid at download (2026)**, not the 2025 service that produced the ridership.

**Source 3: Open-Meteo historical weather** (archive API, New York City coordinates 40.7128, -74.0060, timezone America/New_York). 10 variables: temperature, apparent temperature, relative humidity, dew point, precipitation, snowfall, pressure, cloud cover, wind speed, wind gusts. Units: degC, %, mm, cm, hPa, km/h. One city-wide value per hour (simplification). Check Open-Meteo's terms for the licence wording.

**Calendar features:** US/NY holidays from the Python `holidays` package; `is_school_day` from a **hand-typed, approximate NYC school calendar** in the script (verify at schools.nyc.gov).

## 5. The dataset builder: `build_transit_dataset.py`
Run: `pip install pandas numpy requests holidays`, then `python build_transit_dataset.py` (options: `--no-gtfs`, `--ref-date YYYY-MM-DD`). Output folder `data/`:
`transit_dataset.csv` (the modelling table), `feature_groups.json` (columns per group for ablations), `SOURCES.md` (provenance for the report), `raw/mta_ridership_raw.csv`, `raw/weather_raw.csv`, `raw/gtfs_supply_table.csv`.

Design choices:
- **Four 2-week windows** (2025-01-13 to 01-26 winter, 04-07 to 04-20 spring, 07-07 to 07-20 summer, 09-15 to 09-28 autumn) so weather, holidays and school days vary. Each is fetched with **7 extra warm-up days** used only for lag features, then dropped.
- `N_ROUTES = 6` busiest routes, at most `MAX_PER_BOROUGH = 2` per borough, only routes that match a GTFS route; variants of the same corridor (`M15` / `M15+`) are not both chosen. Ranking uses the spring window.
- Hours with no API row are filled with 0 riders (assumption; state in the report).
- Lag/rolling features are computed **per route per window using only past values** (shift before rolling). Verified: `lag_24h` equals ridership 24 hours earlier.
- Times are NYC local, naive timestamps; weather is joined on the exact hour.

Last real run (before the corridor de-duplication): 8,064 rows, 36 columns (33 features + target `ridership`, plus `transit_timestamp` and `window_id`), no missing values. Routes then: M15+, Q58, Q65, B6, M15, BX12+.

## 6. Feature groups (for the source-contribution analysis)
| Group | Columns |
|---|---|
| route | `bus_route`, `borough`, `gtfs_n_stops`, `gtfs_max_stops_per_trip`, `gtfs_n_directions` |
| supply (GTFS) | `sched_trips_hr`, `sched_headway_min`, `sched_trips_day` |
| weather | the 10 Open-Meteo variables |
| calendar | `hour`, `day_of_week`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `is_am_peak`, `is_pm_peak`, `is_holiday`, `is_school_day` |
| history | `lag_1h`, `lag_24h`, `lag_168h`, `roll_24h_mean`, `roll_168h_mean` |

Target: `ridership`. Not model inputs: `transit_timestamp`, `window_id`.

## 7. Known issues and decisions from review
- **Drop `month` from the model inputs.** Only months 1, 4, 7, 9 exist, so it acts as a window ID, and September never appears in training under a time split. Keep it for EDA only.
- **Drop `gtfs_n_directions`**: it is 2 for every route.
- **Route-level GTFS features are redundant with `bus_route` when there are only 6 routes** (they take 6 distinct values), so a "remove the schedule source" ablation would show little effect. Only `sched_trips_hr` / `sched_headway_min` vary within a route. **Recommended fix: `N_ROUTES = 16`, `MAX_PER_BOROUGH = 4` (about 21,500 rows)**; this also allows a "new, unseen route" group split. Not yet decided.
- **Test set = September only** under a global time-ordered split, and the only holiday (Jan 20) falls in training, so `is_holiday` carries no test signal. Use the strict time split as the main result and a **blocked split inside each window** (about 10 days train, 2 validation, 2 test) as a robustness check; state the single-season limitation.
- Earlier suspicion that GTFS trips were inflated was partly a misread: counts are both directions combined, so "1.6 min" means about 3 min per direction. Numbers for M15+ and B6 look plausible but were **not checked against published timetables**.
- Only 3 raw columns come from the MTA and 6-10 from weather; most of the 33 features are engineered. **Ask whether engineered features count toward the 15-feature minimum.**
- Environment note: the team runs pandas 3.x (uses `day_of_week`, not `dayofweek`).

## 8. Modelling plan (next step)
1. **EDA (>= 5 plots):** ridership by hour and weekday, by route, vs temperature/precipitation, correlation heatmap, distribution/outliers, missing-value check, schedule trips vs ridership.
2. **Two model variants:** (a) *with history*, which uses lags and rolling means and is the most accurate; (b) *planning model*, which has no history features, uses only things known in advance (route, calendar, schedule, weather forecast), and is what the deployed app should use because users will not know recent ridership.
3. **Pipeline:** `ColumnTransformer` (one-hot for `bus_route`/`borough`, scaling for numeric where needed, none for trees) + model, fitted on training data only.
4. **Split:** time-ordered 70/15/15 (oldest train, next validation, newest test).
5. **Models:** baseline (same hour last week, or mean by route-hour) + Linear Regression, KNN Regressor, Random Forest, Gradient Boosting.
6. **Tuning:** GridSearchCV/RandomizedSearchCV with **`TimeSeriesSplit(n_splits=5)`** (not shuffled K-fold, which leaks the future; explain this in the report and viva).
7. **Evaluation:** MAE, RMSE, R^2 on the test set, once. Add per-route and per-hour error analysis and an over/under-fitting discussion.
8. **Source-contribution ablation:** retrain with each group removed (use `feature_groups.json`) and report the change in MAE; this is the project's main analytic story.
9. **Save:** `joblib.dump(pipeline, "models/....joblib")`; pin versions in `requirements.txt`.

## 9. Deployment plan
Inputs for the app (planning model): route, date, hour, weather (or "use typical weather"). Output: predicted riders plus a chart (for example the predicted daily profile for that route). Build all three: Streamlit Community Cloud (public URL), Hugging Face Space + model card, and a local Flask/FastAPI app with `/predict`, input validation and clear error messages. Test each live before the demo.

Suggested repo layout: `data/` (or download script `build_transit_dataset.py`), `notebooks/`, `app/`, `models/`, `requirements.txt`, `README.md`, `CLAUDE.md`.

## 10. Open items
- [ ] Instructor approval of the topic (first-come; message them soon).
- [ ] Ask: do engineered features count toward the 15 features?
- [ ] Ask: one deployment or all three?
- [x] Decide 6 vs 16 routes: **16 chosen** (2026-11-06), dataset rebuilt, 21,504 rows.
- [ ] Verify the school-calendar dates and the licence wording for each source.
- [ ] EDA, modelling, ablation, apps, report, viva preparation (every member must be able to explain every part).

## 11. Working agreements for Claude
- Teammates include non-technical people: explain in simple English when asked.
- Be honest about what is verified versus recalled from memory; never claim a live service or number was checked when it was not.
- Preserve the leakage rules: fit preprocessing on training data only; lags use past values only; time-ordered splits.
- Cite all external code, data and papers; copied work without citation scores zero.

## 12. Repo status (as of 2026-11-06)
- **Done:** dataset rebuilt with `N_ROUTES = 16`, `MAX_PER_BOROUGH = 4` (21,504 rows, 36 columns, no missing values; routes M15+, Q58, Q65, B6, Q27, Q25, M101, BX12+, B41, M86+, B35, B1, M4, BX9, BX19, BX36). `.venv` created with Python 3.14 (`.venv/bin/python`, git-ignored). `requirements.txt` pinned. `notebooks/01_eda.ipynb` executed, with 8 plots saved to `figures/`.
- **EDA findings:** weekday double peak (7-8 am sharp, 5 pm broad), weekends one low midday hump; routes differ a lot in size; raw weather plot is misleading (hot/rainy hours are afternoons), and after dividing by the typical ridership for the same route-hour-daytype, rain is about -3% and freezing weather about -6% (small); Jan 20 holiday behaved like a weekend (133k vs about 200-218k on weekdays); weather columns correlate 0.95-0.99 with each other; `gtfs_n_directions` is constant.
- **Modelling done (notebooks/02_modelling.ipynb, executed):** day-based time split 70/15/15 (train Jan 13-Jul 17, val Jul 18-Sep 19, test Sep 20-28). Baseline + Linear Regression, KNN, Random Forest, Gradient Boosting in Pipelines, GridSearchCV with TimeSeriesSplit(5), chosen on validation MAE, refit on train+val, tested once. **Test results:** planning model Random Forest MAE 53.6 / RMSE 90.2 / R2 0.94 (baseline route-hour average MAE 74.5); history model Random Forest MAE 48.0 / R2 0.95 (baseline last week MAE 66.0). Final pipelines saved to `models/planning_model.joblib` and `models/history_model.joblib`. History features help only about 10%; planning model was tested with actual (not forecast) weather. Test set is only 9 days, so small differences are noise.
- **Ablation done (notebooks/03_ablation.ipynb):** same Random Forest, 5-fold TimeSeriesSplit CV on train+val only (test untouched). Full planning model CV MAE 69.4. Removing calendar +32 (biggest); weather +5.8; GTFS schedule +3.0; route name +0.07 (GTFS route facts also identify the route, so removing route name AND all GTFS gives +66); history features -17.5. Build-up: calendar only 134, + route name 76, + GTFS 75, + weather 69. Story: MTA ridership + calendar do most of the work; GTFS and weather are small extras. `feature_groups.json` mixes route name and GTFS route facts in group `route`, so the notebook splits them.
- **Robustness done (notebooks/04_robustness.ipynb):** blocked split (train first 10 days of each window, test last 4 days): planning 47.3 vs baseline 63.3, history 39.3. **Leave-one-window-out: planning only about 8% better than baseline (72.6 vs 78.9; January no better), history 49.6.** **Unseen routes: planning 133 vs baseline 133 (no better).** So the 28% gain in notebook 02 depends on having seen similar seasons; the app must only offer the 16 trained routes. Option to improve: more windows (e.g. 8) in the builder.
- `notebooks/utils.py` holds shared helpers for notebooks 03-04 (notebook 02 keeps its own copies).
- **Git:** original script, data and `gtfs_zips/` are committed and pushed. Everything from 2026-11-06 (rebuilt data, notebooks, figures, models, `requirements.txt`) is **not committed yet**; the user said not to commit until they ask.
- **Not done:** apps, README, report, all other open items in section 10.
