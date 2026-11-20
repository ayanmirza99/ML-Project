"""Check that the browser version of the model (hf_space/) gives the same answers as the Python model.

Needs Node.js. Run after app/build_hf_space.py:  .venv/bin/python app/verify_static_space.py
Checks (all offline: dates use recorded or typed-in weather, so Open-Meteo is not called):
  0. app/features.py rebuilds exactly the model inputs stored in data/transit_dataset.csv for every dataset row;
  1. the exported trees give the same raw prediction as scikit-learn for every dataset row;
  2. the full app path gives the same answer as features.predict (24 hourly predictions, ± range, warnings,
     weather source) for every route and every recorded 2025 date, plus 400 random dates up to 2027-06-30 with typed-in weather;
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
(async () => {
  const out = { raw: cases.matrix.map((x) => tp.rawPredict(model, x)), results: [], errors: [] };
  for (const c of cases.results) out.results.push(await tp.predict(model, data, c.route, c.date, c.hour, c.weather));
  for (const c of cases.errors) {
    try { await tp.predict(model, data, c.route, c.date, c.hour, c.weather); out.errors.push(null); }
    catch (e) { out.errors.push(e instanceof tp.InputError ? e.message : "NOT AN InputError: " + e.message); }
  }
  fs.writeFileSync(process.argv[4], JSON.stringify(out));
})();
"""


def python_error(case):
    try:
        features.predict(model, case["route"], case["date"], case["hour"], case["weather"])
        return None
    except features.InputError as error:
        return str(error)


model = joblib.load(ROOT / "models" / "planning_model.joblib")
prep, booster = model.named_steps["prep"], model.named_steps["model"]
inputs = list(model.feature_names_in_)

# 0. the app's inputs equal the dataset's inputs
dataset = pd.read_csv(ROOT / "data" / "transit_dataset.csv", parse_dates=["transit_timestamp"])
dataset["day"] = dataset["transit_timestamp"].dt.normalize()
input_mismatch = 0
for (route, day), part in dataset.groupby(["bus_route", "day"]):
    rows = features.day_features(route, day).set_index("transit_timestamp")[inputs]
    expected = part.set_index("transit_timestamp")[inputs]
    numeric = [c for c in inputs if c not in ("bus_route", "borough")]
    same = rows.loc[expected.index, ["bus_route", "borough"]].equals(expected[["bus_route", "borough"]]) and \
        np.allclose(rows.loc[expected.index, numeric].to_numpy(float), expected[numeric].to_numpy(float), rtol=0, atol=1e-9)
    input_mismatch += not same

# 1. every dataset row, raw (unrounded) predictions
matrix = prep.transform(dataset[inputs])
expected_raw = booster.predict(matrix)

# 2. full predictions with the ± range, warnings and weather source
results = [{"route": r, "date": str(d.date()), "hour": int(h), "weather": None}
           for r in features.ROUTES for d, h in zip(pd.date_range(features.FIRST_DATE, features.LAST_RECORDED_DATE),
                                                  np.resize(np.arange(24), 10_000))]
rng = np.random.default_rng(0)
all_dates = pd.date_range(features.FIRST_DATE, features.LAST_DATE)
for _ in range(400):
    weather = {k: round(float(rng.uniform(lo, hi)), 1) for k, (lo, hi) in features.WEATHER_LIMITS.items()}
    results.append({"route": str(rng.choice(features.ROUTES)), "date": str(all_dates[rng.integers(len(all_dates))].date()),
                    "hour": int(rng.integers(24)), "weather": weather})
results.append({"route": "B6", "date": "2025-04-10", "hour": 17, "weather": {"precipitation": 2.0}})  # partly typed weather
expected_results = [features.predict(model, c["route"], c["date"], c["hour"], c["weather"]) for c in results]

# 3. bad inputs
ok = {"route": "M15+", "date": "2025-09-23", "hour": 8, "weather": None}
far = str((features.today_in_new_york() + pd.Timedelta(days=40)).date())
errors = [dict(ok, route="ZZZ"), dict(ok, date="2025-13-01"), dict(ok, date="2030-01-01"), dict(ok, date="2025-01-05"),
          dict(ok, date="2027-07-01"), dict(ok, hour=24), dict(ok, hour=-1), dict(ok, weather={"temperature_2m": 99.0}),
          dict(ok, weather={"precipitation": -1.0}), dict(ok, weather={"humidity": 50.0}),
          dict(ok, date=far), dict(ok, date=far, weather={"temperature_2m": 5.0}), ok]
expected_errors = [python_error(c) for c in errors]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "check.js").write_text(NODE_CHECK)
    (tmp / "cases.json").write_text(json.dumps({"matrix": matrix.tolist(), "results": results, "errors": errors}))
    subprocess.run(["node", str(tmp / "check.js"), str(ROOT / "hf_space"), str(tmp / "cases.json"), str(tmp / "out.json")], check=True)
    got = json.loads((tmp / "out.json").read_text())

raw_diff = np.abs(np.array(got["raw"]) - expected_raw)
result_mismatch = [(e, g) for e, g in zip(expected_results, got["results"]) if e != g]
error_mismatch = [(e, g) for e, g in zip(expected_errors, got["errors"]) if e != g]

print(f"0. app inputs vs dataset: {dataset.groupby(['bus_route', 'day']).ngroups:,} route-days; different: {input_mismatch}")
print(f"1. dataset rows: {len(expected_raw):,}; largest raw difference: {raw_diff.max():.3g} riders; exactly equal: {(raw_diff == 0).mean():.2%}")
print(f"2. full predictions: {len(results):,} ({len(results) * 24:,} hourly values, with range, warnings, weather source); different: {len(result_mismatch)}")
for expected, got_result in result_mismatch[:3]:
    print("   python:", expected, "\n   browser:", got_result)
print(f"3. error cases: {len(errors)}; different messages: {len(error_mismatch)}")
for expected, got_message in error_mismatch:
    print("   python:", expected, "| browser:", got_message)
passed = input_mismatch == 0 and raw_diff.max() < 1e-6 and not result_mismatch and not error_mismatch
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
