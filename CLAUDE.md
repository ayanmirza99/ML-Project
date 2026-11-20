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

**Calendar features:** US **federal** holidays from the Python `holidays` package (NY state-only observances such as Lincoln's Birthday are excluded: ridership on those days was normal); `is_school_day` from a **hand-typed, approximate NYC school calendar** in the script (verify at schools.nyc.gov).

## 5. The dataset builder: `build_transit_dataset.py`
Run: `pip install pandas numpy requests holidays`, then `python build_transit_dataset.py` (options: `--no-gtfs`, `--ref-date YYYY-MM-DD`). Output folder `data/`:
`transit_dataset.csv` (the modelling table), `feature_groups.json` (columns per group for ablations), `SOURCES.md` (provenance for the report), `raw/mta_ridership_raw.csv`, `raw/weather_raw.csv`, `raw/gtfs_supply_table.csv`.

Design choices:
- **Eight 2-week windows** in 2025 (Jan 13-26, Feb 10-23, Apr 7-20, May 19-Jun 1, Jun 9-22, Jul 7-20, Aug 25-Sep 7, Sep 15-28), chosen to include MLK Day, Presidents Day, Memorial Day, Juneteenth and Labor Day and to stay inside the typed school calendar (Jan-Sep). Each is fetched with **7 extra warm-up days** used only for lag features, then dropped.
- `N_ROUTES = 24` busiest routes, at most `MAX_PER_BOROUGH = 6` per borough, only routes that match a GTFS route; variants of the same corridor (`M15` / `M15+`) are not both chosen. Ranking uses the April window (`ROUTE_PICK_WINDOW = 2`). Chosen: M15+, Q58, Q65, B6, Q27, Q25, M101, Q44+, BX12+, B41, M86+, Q10, B35, B1, M4, M14D+, B38, BX9, BX19, B8, M96, BX36, S79+, BX7.
- Hours with no API row are filled with 0 riders (assumption; state in the report).
- Lag/rolling features are computed **per route per window using only past values** (shift before rolling). Verified: `lag_24h` equals ridership 24 hours earlier.
- Times are NYC local, naive timestamps; weather is joined on the exact hour.

Current dataset (rebuilt 2026-11-06): **64,512 rows, 36 columns** (33 features + target `ridership`, plus `transit_timestamp` and `window_id`), 24 routes x 8 windows, no missing values, no duplicate route-hours, `lag_24h` verified equal to ridership 24 h earlier.

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
- [x] Decide route count: first 16, then **24 routes and 8 windows** (2026-11-06), 64,512 rows.
- [ ] Verify the school-calendar dates and the licence wording for each source.
- [x] EDA, modelling, ablation, robustness checks and local apps are done (see section 12).
- [ ] README, report, live deployments, demo and viva preparation (every member must be able to explain every part).

## 11. Working agreements for Claude
- Teammates include non-technical people: explain in simple English when asked.
- Be honest about what is verified versus recalled from memory; never claim a live service or number was checked when it was not.
- Preserve the leakage rules: fit preprocessing on training data only; lags use past values only; time-ordered splits.
- Cite all external code, data and papers; copied work without citation scores zero.

## 12. Repo status (as of 2026-11-20)
- **Git:** the 2026-11-06 work (data, notebooks, figures, apps, README) was pushed to `github.com/ayanmirza99/ML-Project` without `models/`. On 2026-11-20 `models/`, the Docker-based Space files and these notes were committed on request. Commit only when the user asks.
- **Environment:** `.venv` (Python 3.14, git-ignored; recreated 2026-11-20 on the user's Mac with `/opt/homebrew/bin/python3.14`, because the default `python3` there is 3.15), `requirements.txt` pinned (fastapi, holidays, joblib, matplotlib, numpy, pandas, requests, scikit-learn, seaborn, streamlit, uvicorn). Flask is not used. numpy 2.5.4 needs Python >= 3.12; the pinned set installs on Linux for 3.12, 3.13 and 3.14. First import after creating a venv is slow (about 30 s).
- **Data:** rebuilt with 8 windows, 24 routes, federal holidays (see section 5), committed.
- **Notebooks (all executed), shared helpers in `notebooks/utils.py`:**
  - `01_eda.ipynb`: 9 sections, 9+ plots in `figures/`. Hour profile (weekday peaks 7-8 am and 4-5 pm, weekend hump at 2 pm); raw weather plot is misleading (hot/rainy hours are afternoons), adjusted for route-hour-daytype rain is about -1%, freezing about -7%; all 5 holidays have fewer riders than any normal weekday; **Q10, Q65, Q58 fell 32-39% and Q25 rose 32% between June and late August, 16 of 24 routes stayed within 5%** (cause not verified).
  - `02_modelling.ipynb`: day-based split (train Jan 13-Jul 14, val Jul 15-Sep 4, test Sep 5-28 = 17 days), baselines, Linear, KNN, Random Forest, HistGradientBoosting and an RF+GB average, GridSearchCV with TimeSeriesSplit(5), choose on validation among models **under 50 MB** (RF is about 110 MB compressed, over GitHub's 100 MB limit), refit on train+val, test once. **Test (planning, deployed): Gradient Boosting MAE 65.4, RMSE 113.3, R2 0.92 vs baseline 78.0 (16% better; 14.5% of the average 450 riders). History model: MAE 39.0, R2 0.97 vs baseline 54.1.** RF 64.0 and average 63.9 are within noise. Planning error is concentrated in four Queens routes (MAE 158 vs 47 elsewhere).
  - `03_ablation.ipynb` (tuned boosting, 5-fold TimeSeriesSplit CV on train+val only; full planning CV MAE 55.0): minus calendar +47.3; minus route name +0.8 (GTFS also identifies the route); minus route name and all GTFS +74; minus schedule +2.8; minus weather +2.9; history model -13.6. Build-up: calendar only 126, + route name 60, + GTFS 58, + weather 55. Story: ridership data and calendar do most of the work; GTFS and weather add a few percent.
  - `04_robustness.ipynb`: blocked split planning 49.2 vs baseline 59.6; leave-one-window-out planning 53.0 vs 68.9 (beats baseline in 7 of 8 windows, history 36.6); unseen routes planning 111 vs baseline 132 (about 24% of average ridership).
  - `05_improvements.ipynb`: leave-one-window-out with Random Forest. Fewer weather columns (keep temperature, precipitation, snowfall, wind speed) adopted (52.7 vs 53.9; better in 7 of 8 windows). Typical-ridership feature rejected (worse). Dropped columns are in `utils.WEATHER_DROPPED`.
- **Models:** `models/planning_model.joblib` and `models/history_model.joblib` (Gradient Boosting, 1.5 MB each, saved with compress=3). The originals were never committed; on 2026-11-20 the planning model was restored from `hf_model/` (same file the Space build copied) and the history model was regenerated by re-running `02_modelling.ipynb` (all printed results identical to the committed notebook; the re-run planning model gave identical predictions on all 64,512 rows).
- **Apps (built and tested locally 2026-11-20; none deployed yet):**
  - `app/features.py` shared core (validation, builds the model's input row for a route/date/weather; verified to reproduce dataset rows exactly), `app/prepare_lookups.py` builds `app/lookup/*.csv`.
  - `app/api.py` + `app/index.html`: local FastAPI with HTML front end, `POST /predict`, `GET /routes`, `GET /health`, input validation with clear 400 messages (tested with 10 bad-input cases). Run: `cd app && ../.venv/bin/uvicorn api:app --port 8000`.
  - `app/streamlit_app.py`: Streamlit app (tested headless with `AppTest` and with a real `streamlit run` server). Streamlit Community Cloud is **not set up** (needs the user's browser login; choose Python 3.14 in Advanced settings).
  - `app/build_hf_space.py` builds `hf_space/` (Space files, `app.py`, pinned requirements, Space README, **Dockerfile**) and `hf_model/` (model card + model file). Hugging Face retired the built-in Streamlit SDK on 2025-04-30, so the Space uses `sdk: docker`, `app_port: 8501`, `python:3.14-slim`, user 1000. The Docker image was **not built locally** (no Docker on the Mac). **Nothing has been pushed to Hugging Face** (user not logged in).
  - Supported dates: 2025-01-06 to 2025-09-28 only (model and weather coverage); the 24 trained routes only.
- **Known app caveat:** entering extreme weather on a date from another season changes the prediction a lot because weather also hints at the season (e.g. -10 C on a September date raised Q58 by about 22%).
- **Not done:** README live links, report (3-4 pages), demo and viva preparation, deploying the three apps live (Streamlit Cloud public URL, Hugging Face Space and model repo, local FastAPI demo), instructor questions, verifying school calendar and licences, spot-check of GTFS counts against published timetables.
