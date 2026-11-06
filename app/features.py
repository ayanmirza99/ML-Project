from pathlib import Path

import holidays
import numpy as np
import pandas as pd

LOOKUP = Path(__file__).resolve().parent / "lookup"

FIRST_DATE = pd.Timestamp("2025-01-06")
LAST_DATE = pd.Timestamp("2025-09-28")
DEFAULT_DATE = pd.Timestamp("2025-09-23")

WEATHER_LIMITS = {
    "temperature_2m": (-30.0, 45.0),
    "precipitation": (0.0, 100.0),
    "snowfall": (0.0, 30.0),
    "wind_speed_10m": (0.0, 150.0),
}

# Same school calendar as build_transit_dataset.py (hand-typed, approximate)
SCHOOL_CLOSED_RANGES = [("2024-12-24", "2025-01-01"), ("2025-02-17", "2025-02-21"),
                        ("2025-04-14", "2025-04-18"), ("2025-06-27", "2025-09-03")]
SCHOOL_CLOSED_DAYS = ["2025-01-20", "2025-05-26", "2025-06-19", "2025-09-23", "2025-09-24"]

routes = pd.read_csv(LOOKUP / "routes.csv").set_index("bus_route")
schedule = pd.read_csv(LOOKUP / "schedule.csv").set_index(["bus_route", "day_type", "hour"])
recorded_weather = pd.read_csv(LOOKUP / "weather_2025.csv", parse_dates=["transit_timestamp"]).set_index("transit_timestamp")

holiday_dates = set(holidays.country_holidays("US", years=2025).keys())
school_closed = set(pd.to_datetime(SCHOOL_CLOSED_DAYS).date)
for start, end in SCHOOL_CLOSED_RANGES:
    school_closed.update(d.date() for d in pd.date_range(start, end))

ROUTES = sorted(routes.index)


class InputError(ValueError):
    pass


def check_route(route):
    if route not in routes.index:
        raise InputError(f"Unknown route '{route}'. Choose one of: {', '.join(ROUTES)}.")


def check_date(date):
    try:
        day = pd.Timestamp(date).normalize()
    except (ValueError, TypeError):
        raise InputError(f"Date '{date}' is not valid. Use the format YYYY-MM-DD.")
    if not FIRST_DATE <= day <= LAST_DATE:
        raise InputError(f"Date must be between {FIRST_DATE.date()} and {LAST_DATE.date()}, "
                         "the period the model was trained on.")
    return day


def check_hour(hour):
    if isinstance(hour, bool) or not isinstance(hour, (int, np.integer)) or not 0 <= hour <= 23:
        raise InputError("Hour must be a whole number from 0 to 23.")
    return int(hour)


def check_weather(weather):
    weather = {k: v for k, v in (weather or {}).items() if v is not None}
    for name, value in weather.items():
        if name not in WEATHER_LIMITS:
            raise InputError(f"Unknown weather field '{name}'. Allowed: {', '.join(WEATHER_LIMITS)}.")
        low, high = WEATHER_LIMITS[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise InputError(f"{name} must be a number between {low} and {high}.")
    return weather


def day_features(route, date, weather=None):
    check_route(route)
    day = check_date(date)
    weather = check_weather(weather)

    hours = pd.date_range(day, periods=24, freq="h")
    rows = pd.DataFrame({"transit_timestamp": hours})
    rows["bus_route"] = route
    rows["hour"] = rows["transit_timestamp"].dt.hour
    rows["day_of_week"] = day.day_of_week
    rows["is_weekend"] = int(day.day_of_week >= 5)
    rows["hour_sin"] = np.sin(2 * np.pi * rows["hour"] / 24)
    rows["hour_cos"] = np.cos(2 * np.pi * rows["hour"] / 24)
    rows["is_am_peak"] = rows["hour"].between(7, 9).astype(int)
    rows["is_pm_peak"] = rows["hour"].between(16, 19).astype(int)
    rows["is_holiday"] = int(day.date() in holiday_dates)
    rows["is_school_day"] = int(day.day_of_week < 5 and day.date() not in school_closed)

    for column, value in routes.loc[route].items():
        rows[column] = value

    day_type = "weekday" if day.day_of_week < 5 else "saturday" if day.day_of_week == 5 else "sunday"
    sched = schedule.loc[route, day_type].reindex(rows["hour"])
    for column in sched.columns:
        rows[column] = sched[column].to_numpy()

    recorded = recorded_weather.loc[day:day + pd.Timedelta(hours=23)]
    for column in WEATHER_LIMITS:
        rows[column] = weather[column] if column in weather else recorded[column].to_numpy()
    return rows


def predict_day(model, route, date, weather=None):
    rows = day_features(route, date, weather)
    predicted = np.clip(model.predict(rows[list(model.feature_names_in_)]), 0, None)
    return pd.DataFrame({"hour": rows["hour"], "predicted_riders": np.round(predicted).astype(int)})


def predict_hour(model, route, date, hour, weather=None):
    hour = check_hour(hour)
    day = predict_day(model, route, date, weather)
    return int(day.loc[day["hour"] == hour, "predicted_riders"].iloc[0])
