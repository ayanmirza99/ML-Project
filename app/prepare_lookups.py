from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "lookup"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "transit_dataset.csv")
df["day_type"] = np.select([df["day_of_week"] < 5, df["day_of_week"] == 5], ["weekday", "saturday"], "sunday")

route_cols = ["borough", "gtfs_n_stops", "gtfs_max_stops_per_trip"]
schedule_cols = ["sched_trips_hr", "sched_headway_min", "sched_trips_day"]

df.groupby("bus_route")[route_cols].first().reset_index().to_csv(OUT / "routes.csv", index=False)
df.groupby(["bus_route", "day_type", "hour"])[schedule_cols].first().reset_index().to_csv(OUT / "schedule.csv", index=False)

weather_cols = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]
weather = pd.read_csv(ROOT / "data" / "raw" / "weather_raw.csv")
weather[["transit_timestamp"] + weather_cols].to_csv(OUT / "weather_2025.csv", index=False)

print("saved lookup tables to", OUT)
