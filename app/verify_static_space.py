"""Check that the browser version of the model (hf_space/) gives the same answers as the Python model.

Needs Node.js. Run after app/build_hf_space.py:  .venv/bin/python app/verify_static_space.py
Checks:
  1. the exported trees give the same raw prediction as scikit-learn for every row of data/transit_dataset.csv;
  2. the full app path (lookups, inputs, rounding) gives the same 24 hourly predictions as app/features.py
     for every route and every supported date with recorded weather, plus random typed-in weather;
  3. bad inputs get the same error messages.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import features  # noqa: E402

NODE_CHECK = r"""
const fs = require("fs");
const tp = require(process.argv[2] + "/predict.js");
const model = JSON.parse(fs.readFileSync(process.argv[2] + "/model.json"));
const data = JSON.parse(fs.readFileSync(process.argv[2] + "/data.json"));
const cases = JSON.parse(fs.readFileSync(process.argv[3]));
const out = {};
out.raw = cases.matrix.map((x) => tp.rawPredict(model, x));
out.days = cases.days.map((c) => tp.predictDay(model, data, c.route, c.date, c.weather).map((p) => p.predicted_riders));
out.errors = cases.errors.map((c) => {
  try { tp.predictHour(model, data, c.route, c.date, c.hour, c.weather); return null; }
  catch (e) { return e instanceof tp.InputError ? e.message : "NOT AN InputError: " + e.message; }
});
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""


def python_error(case):
    try:
        features.predict_hour(model, case["route"], case["date"], case["hour"], case["weather"])
        return None
    except features.InputError as error:
        return str(error)


model = joblib.load(ROOT / "models" / "planning_model.joblib")
prep, booster = model.named_steps["prep"], model.named_steps["model"]

# 1. every dataset row, raw (unrounded) predictions
dataset = pd.read_csv(ROOT / "data" / "transit_dataset.csv")
matrix = prep.transform(dataset[list(model.feature_names_in_)])
expected_raw = booster.predict(matrix)

# 2. every route x every date (recorded weather) and 300 random typed-in weather days
days = [{"route": r, "date": str(d.date()), "weather": None}
        for r in features.ROUTES for d in pd.date_range(features.FIRST_DATE, features.LAST_DATE)]
rng = np.random.default_rng(0)
all_dates = pd.date_range(features.FIRST_DATE, features.LAST_DATE)
for _ in range(300):
    weather = {k: round(float(rng.uniform(lo, hi)), 1) for k, (lo, hi) in features.WEATHER_LIMITS.items()}
    days.append({"route": str(rng.choice(features.ROUTES)), "date": str(all_dates[rng.integers(len(all_dates))].date()), "weather": weather})
expected_days = [features.predict_day(model, c["route"], c["date"], c["weather"])["predicted_riders"].tolist() for c in days]

# 3. bad inputs
ok = {"route": "M15+", "date": "2025-09-23", "hour": 8, "weather": None}
errors = [dict(ok, route="ZZZ"), dict(ok, date="2025-13-01"), dict(ok, date="2030-01-01"), dict(ok, date="2025-01-05"),
          dict(ok, hour=24), dict(ok, hour=-1), dict(ok, weather={"temperature_2m": 99.0}),
          dict(ok, weather={"precipitation": -1.0}), dict(ok, weather={"humidity": 50.0}), ok]
expected_errors = [python_error(c) for c in errors]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "check.js").write_text(NODE_CHECK)
    (tmp / "cases.json").write_text(json.dumps({"matrix": matrix.tolist(), "days": days, "errors": errors}))
    subprocess.run(["node", str(tmp / "check.js"), str(ROOT / "hf_space"), str(tmp / "cases.json"), str(tmp / "out.json")], check=True)
    got = json.loads((tmp / "out.json").read_text())

raw_diff = np.abs(np.array(got["raw"]) - expected_raw)
day_mismatch = sum(a != b for a, b in zip(got["days"], expected_days))
error_mismatch = [(e, g) for e, g in zip(expected_errors, got["errors"]) if e != g]

print(f"1. dataset rows: {len(expected_raw):,}; largest raw difference: {raw_diff.max():.3g} riders; exactly equal: {(raw_diff == 0).mean():.2%}")
print(f"2. days checked: {len(days):,} ({len(days) * 24:,} hourly predictions); days with any different hour: {day_mismatch}")
print(f"3. error cases: {len(errors)}; different messages: {len(error_mismatch)}")
for expected, got_message in error_mismatch:
    print("   python:", expected, "| browser:", got_message)
passed = raw_diff.max() < 1e-6 and day_mismatch == 0 and not error_mismatch
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
