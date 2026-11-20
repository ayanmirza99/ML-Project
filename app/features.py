import time
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import requests

LOOKUP = Path(__file__).resolve().parent / "lookup"

FIRST_DATE = pd.Timestamp("2025-01-06")
LAST_RECORDED_DATE = pd.Timestamp("2025-09-28")  # end of the saved weather file and of the training data
LAST_DATE = pd.Timestamp("2027-06-30")           # end of the typed school calendar (2026-27 school year)
TRAINED_MONTHS = {1, 2, 4, 5, 6, 7, 8, 9}        # months that appear in the training windows

# Shown as "± riders" next to every prediction: the validation MAE of the deployed model
# (Gradient Boosting, 55.7 riders per route-hour, notebooks/02_modelling.ipynb), as promised in the proposal.
TYPICAL_ERROR = 56

WEATHER_LIMITS = {
    "temperature_2m": (-30.0, 45.0),
    "precipitation": (0.0, 100.0),
    "snowfall": (0.0, 30.0),
    "wind_speed_10m": (0.0, 150.0),
}

# Weather for dates after the saved 2025 file comes from Open-Meteo (CC BY 4.0), same place and units as the training data.
LATITUDE, LONGITUDE = 40.7128, -74.0060
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_DAYS = 16        # today plus the next 15 days
ARCHIVE_DELAY_DAYS = 5    # the archive lags a few days behind; newer days come from the forecast service

# NYC public school calendar (hand-typed, approximate). Up to 2025-09-28 it is the same as build_transit_dataset.py.
# Later dates: 2025-26 and 2026-27 calendars from schools.nyc.gov/calendar. Days when students do not attend in
# person count as closed (including the 2026-11-03 remote Election Day, because students do not travel).
SCHOOL_CLOSED_RANGES = [("2024-12-24", "2025-01-01"), ("2025-02-17", "2025-02-21"),
                        ("2025-04-14", "2025-04-18"), ("2025-06-27", "2025-09-03"),
                        # 2025-26
                        ("2025-11-27", "2025-11-28"), ("2025-12-24", "2026-01-02"), ("2026-02-16", "2026-02-20"),
                        ("2026-04-02", "2026-04-10"), ("2026-06-27", "2026-09-09"),
                        # 2026-27
                        ("2026-11-26", "2026-11-27"), ("2026-12-24", "2027-01-01"), ("2027-02-15", "2027-02-19"),
                        ("2027-04-22", "2027-04-30"), ("2027-06-29", "2027-09-30")]
SCHOOL_CLOSED_DAYS = ["2025-01-20", "2025-05-26", "2025-06-19", "2025-09-23", "2025-09-24",
                      # 2025-26
                      "2025-10-02", "2025-10-13", "2025-10-20", "2025-11-04", "2025-11-11", "2026-01-19",
                      "2026-03-20", "2026-05-25", "2026-05-27", "2026-06-04", "2026-06-19",
                      # 2026-27
                      "2026-09-21", "2026-10-12", "2026-11-03", "2026-11-11", "2027-01-18", "2027-03-09",
                      "2027-03-26", "2027-05-17", "2027-05-31", "2027-06-10"]

routes = pd.read_csv(LOOKUP / "routes.csv").set_index("bus_route")
schedule = pd.read_csv(LOOKUP / "schedule.csv").set_index(["bus_route", "day_type", "hour"])
recorded_weather = pd.read_csv(LOOKUP / "weather_2025.csv", parse_dates=["transit_timestamp"]).set_index("transit_timestamp")

holiday_dates = set(holidays.country_holidays("US", years=range(FIRST_DATE.year, LAST_DATE.year + 1)).keys())
school_closed = set(pd.to_datetime(SCHOOL_CLOSED_DAYS).date)
for start, end in SCHOOL_CLOSED_RANGES:
    school_closed.update(d.date() for d in pd.date_range(start, end))

ROUTES = sorted(routes.index)
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]


class InputError(ValueError):
    pass


def today_in_new_york():
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()


def default_date():
    return today_in_new_york() + pd.Timedelta(days=1)


def check_route(route):
    if route not in routes.index:
        raise InputError(f"Unknown route '{route}'. Choose one of: {', '.join(ROUTES)}.")


def check_date(date):
    try:
        day = pd.Timestamp(date).normalize()
    except (ValueError, TypeError):
        raise InputError(f"Date '{date}' is not valid. Use the format YYYY-MM-DD.")
    if not FIRST_DATE <= day <= LAST_DATE:
        raise InputError(f"Date must be between {FIRST_DATE.date()} and {LAST_DATE.date()}.")
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


def calendar_info(day):
    """Calendar inputs for one day."""
    return {
        "day_of_week": int(day.day_of_week),
        "is_holiday": int(day.date() in holiday_dates),
        "is_school_day": int(day.day_of_week < 5 and day.date() not in school_closed),
    }


_weather_cache = {}


def fetch_weather(day):
    """Hourly weather for one day after the saved file: Open-Meteo archive for older days, forecast for recent and future days."""
    key = str(day.date())
    cached = _weather_cache.get(key)
    if cached and time.time() - cached[0] < 3600:
        return cached[1]
    today = today_in_new_york()
    if day > today + pd.Timedelta(days=FORECAST_DAYS - 1):
        raise InputError(f"No weather forecast exists yet for {key} (forecasts reach {FORECAST_DAYS} days ahead). "
                         "Enter the weather yourself.")
    url = ARCHIVE_URL if day < today - pd.Timedelta(days=ARCHIVE_DELAY_DAYS) else FORECAST_URL
    params = {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": ",".join(WEATHER_LIMITS),
              "timezone": "America/New_York", "start_date": key, "end_date": key}
    problem = InputError(f"Could not get the weather for {key} from Open-Meteo. Try again, or enter the weather yourself.")
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        hourly = response.json()["hourly"]
    except (requests.RequestException, ValueError, KeyError) as error:
        raise problem from error
    values = {}
    for column in WEATHER_LIMITS:
        by_hour = {}
        for stamp, value in zip(hourly["time"], hourly.get(column, [])):
            hour = int(stamp[11:13])
            if stamp.startswith(key) and hour not in by_hour and value is not None:
                by_hour[hour] = float(value)
        if not by_hour:
            raise problem
        # a missing hour (for example when clocks change) takes the value of the hour before it, or after it at midnight
        series = [by_hour.get(hour) for hour in range(24)]
        for hour in range(24):
            if series[hour] is None:
                series[hour] = series[hour - 1] if hour > 0 else next(v for v in series if v is not None)
        values[column] = series
    _weather_cache[key] = (time.time(), {"values": values, "source": "archive" if url == ARCHIVE_URL else "forecast"})
    return _weather_cache[key][1]


def day_weather(day, weather):
    """24 hourly values per weather column, and where they came from."""
    if day <= LAST_RECORDED_DATE:
        recorded = recorded_weather.loc[day:day + pd.Timedelta(hours=23)]
        base, source = {c: recorded[c].to_numpy() for c in WEATHER_LIMITS}, "recorded weather (Open-Meteo)"
    elif all(c in weather for c in WEATHER_LIMITS):
        base, source = None, None
    else:
        fetched = fetch_weather(day)
        base = fetched["values"]
        source = "Open-Meteo historical weather" if fetched["source"] == "archive" else "Open-Meteo forecast"
    columns = {c: (np.full(24, float(weather[c])) if c in weather else np.asarray(base[c], dtype=float)) for c in WEATHER_LIMITS}
    if len(weather) == len(WEATHER_LIMITS):
        source = "your weather"
    elif weather:
        source += ", with your values where given"
    return columns, source


def date_warnings(day):
    warnings = []
    if day.month not in TRAINED_MONTHS:
        warnings.append(f"The model never saw {MONTH_NAMES[day.month - 1]} in training (it learned from January, February "
                        "and April to September 2025), so treat this as a rough guess.")
    if day > LAST_RECORDED_DATE:
        warnings.append("This date is after the training data (January to September 2025); bus service may have changed since then.")
    return warnings


def day_features(route, date, weather=None, with_source=False):
    check_route(route)
    day = check_date(date)
    weather = check_weather(weather)

    hours = pd.date_range(day, periods=24, freq="h")
    rows = pd.DataFrame({"transit_timestamp": hours})
    rows["bus_route"] = route
    rows["hour"] = rows["transit_timestamp"].dt.hour
    calendar = calendar_info(day)
    rows["day_of_week"] = calendar["day_of_week"]
    rows["is_weekend"] = int(day.day_of_week >= 5)
    rows["hour_sin"] = np.sin(2 * np.pi * rows["hour"] / 24)
    rows["hour_cos"] = np.cos(2 * np.pi * rows["hour"] / 24)
    rows["is_am_peak"] = rows["hour"].between(7, 9).astype(int)
    rows["is_pm_peak"] = rows["hour"].between(16, 19).astype(int)
    rows["is_holiday"] = calendar["is_holiday"]
    rows["is_school_day"] = calendar["is_school_day"]

    for column, value in routes.loc[route].items():
        rows[column] = value

    day_type = "weekday" if day.day_of_week < 5 else "saturday" if day.day_of_week == 5 else "sunday"
    sched = schedule.loc[route, day_type].reindex(rows["hour"])
    for column in sched.columns:
        rows[column] = sched[column].to_numpy()

    columns, source = day_weather(day, weather)
    for column, values in columns.items():
        rows[column] = values
    return (rows, source) if with_source else rows


def predict_day(model, route, date, weather=None):
    rows = day_features(route, date, weather)
    predicted = np.clip(model.predict(rows[list(model.feature_names_in_)]), 0, None)
    return pd.DataFrame({"hour": rows["hour"], "predicted_riders": np.round(predicted).astype(int)})


def predict_hour(model, route, date, hour, weather=None):
    hour = check_hour(hour)
    day = predict_day(model, route, date, weather)
    return int(day.loc[day["hour"] == hour, "predicted_riders"].iloc[0])


def predict(model, route, date, hour, weather=None):
    """Everything the apps show: the prediction, its ± range, the day profile, warnings and the weather source."""
    hour = check_hour(hour)
    rows, source = day_features(route, date, weather, with_source=True)
    predicted = np.round(np.clip(model.predict(rows[list(model.feature_names_in_)]), 0, None)).astype(int)
    value = int(predicted[hour])
    return {
        "route": route,
        "date": str(rows["transit_timestamp"].iloc[0].date()),
        "hour": hour,
        "predicted_riders": value,
        "typical_error": TYPICAL_ERROR,
        "likely_range": [max(0, value - TYPICAL_ERROR), value + TYPICAL_ERROR],
        "daily_profile": [{"hour": h, "predicted_riders": int(p)} for h, p in enumerate(predicted)],
        "warnings": date_warnings(rows["transit_timestamp"].iloc[0]),
        "weather_source": source,
    }
