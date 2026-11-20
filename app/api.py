from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

import features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "planning_model.joblib"
PAGE = (Path(__file__).resolve().parent / "index.html").read_text(encoding="utf-8")

model = joblib.load(MODEL_PATH)
app = FastAPI(title="TransitPulse")


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/routes")
def routes():
    return {"routes": features.ROUTES, "first_date": str(features.FIRST_DATE.date()), "last_date": str(features.LAST_DATE.date()),
            "default_date": str(features.default_date().date()), "typical_error": features.TYPICAL_ERROR}


@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "Request body must be valid JSON."}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Request body must be a JSON object."}, status_code=400)

    missing = [name for name in ("route", "date", "hour") if name not in body]
    if missing:
        return JSONResponse({"error": f"Missing field(s): {', '.join(missing)}."}, status_code=400)

    try:
        # predicted_riders, likely_range (± typical_error), daily_profile, warnings, weather_source
        return features.predict(model, body["route"], body["date"], body["hour"], body.get("weather"))
    except features.InputError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
