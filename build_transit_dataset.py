#!/usr/bin/env python3
"""
build_transit_dataset.py  --  TransitPulse dataset builder (three sources)

Predict hourly bus ridership per route from:
  SOURCE 1  MTA Bus Hourly Ridership (data.ny.gov)      -> target + route/time + past demand
  SOURCE 2  MTA GTFS static schedule (service supply)   -> scheduled trips/hour, headway, stops
  SOURCE 3  Open-Meteo historical weather               -> context
  plus      US/NY holidays (python 'holidays' package) and a hand-typed NYC school calendar

Install:   pip install pandas numpy requests holidays
Run:       python build_transit_dataset.py
Options:   --no-gtfs            skip the schedule source (only if the GTFS download fails)
           --ref-date YYYY-MM-DD   date used to pick the active GTFS service period (default: today)

If the GTFS download fails (blocked network), download the bus feeds by hand from
https://www.mta.info/developers  (Bronx, Brooklyn, Manhattan, Queens, Staten Island, MTA Bus Co),
put the .zip files in a folder called  gtfs_zips/  next to this script, and run again.

Outputs (folder ./data):
  transit_dataset.csv            <- THE DATASET to use for EDA / modelling
  feature_groups.json            <- column names per source group (for ablation experiments)
  SOURCES.md                     <- provenance (source, method, date) for your report
  raw/mta_ridership_raw.csv      <- raw API pull
  raw/weather_raw.csv            <- raw weather pull
  raw/gtfs_supply_table.csv      <- route x day-type x hour schedule table

IMPORTANT: this was written without access to the live services, so URLs, column names and
API parameters come from memory. Every step prints what it did and stops with a readable
message if something does not match. Verify the school-calendar dates against schools.nyc.gov.
"""
import argparse
import datetime as dt
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import holidays
except ImportError:
    sys.exit("Missing package. Run: pip install pandas numpy requests holidays")

# =============================== CONFIG ======================================
OUT_DIR = Path("data")
RAW_DIR = OUT_DIR / "raw"
GTFS_DIR = Path("gtfs_zips")

MTA_URL = "https://data.ny.gov/resource/gxb3-akrn.csv"   # MTA Bus Hourly Ridership: Beginning 2025
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

COL_ROUTE = "bus_route"
COL_TIME = "transit_timestamp"
COL_RIDERS = "ridership"

# Four 2-week windows spread over the year (winter / spring / summer / autumn) so that
# weather, holidays and school days actually vary. Each window is fetched with 7 extra
# days BEFORE it; those warm-up days are used only to compute lag features, then dropped.
WINDOWS = [
    ("2025-01-13", "2025-01-26"),   # winter, MLK Day
    ("2025-04-07", "2025-04-20"),   # spring, school spring recess
    ("2025-07-07", "2025-07-20"),   # summer, schools closed
    ("2025-09-15", "2025-09-28"),   # autumn, Rosh Hashanah closure
]
WARMUP_DAYS = 7
N_ROUTES = 6              # 6 routes x 14 days x 24 h x 4 windows = 8,064 final rows
MAX_PER_BOROUGH = 2       # spread the chosen routes over several boroughs
ROUTE_PICK_WINDOW = 1     # which window is used to rank routes by ridership
N_CANDIDATES = 60         # how many top routes to consider

# Hourly weather variables (Open-Meteo names). If the full list is rejected, the core list is used.
WEATHER_FULL = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "dew_point_2m",
                "precipitation", "snowfall", "pressure_msl", "cloud_cover",
                "wind_speed_10m", "wind_gusts_10m"]
WEATHER_CORE = ["temperature_2m", "relative_humidity_2m", "precipitation", "snowfall",
                "wind_speed_10m", "cloud_cover"]
LAT, LON = 40.7128, -74.0060   # New York City

HEADWAY_CAP_MIN = 120     # headway used when no trip is scheduled in an hour

# ---- NYC school calendar (HAND-TYPED, approximate: VERIFY at schools.nyc.gov) ----------
SCHOOL_CLOSED_RANGES = [
    ("2024-12-24", "2025-01-01"),   # winter recess
    ("2025-02-17", "2025-02-21"),   # mid-winter recess
    ("2025-04-14", "2025-04-18"),   # spring recess
    ("2025-06-27", "2025-09-03"),   # summer break
]
SCHOOL_CLOSED_DAYS = [
    "2025-01-20",   # Martin Luther King Jr. Day
    "2025-05-26",   # Memorial Day
    "2025-06-19",   # Juneteenth
    "2025-09-23",   # Rosh Hashanah
    "2025-09-24",   # Rosh Hashanah
]

# MTA GTFS static feeds (bus). Newer S3 hosting first, older mta.info path second.
GTFS_CANDIDATES = {
    "bronx": ["https://rrgtfsfeeds.s3.amazonaws.com/gtfs_bx.zip",
              "http://web.mta.info/developers/data/nyct/bus/google_transit_bronx.zip"],
    "brooklyn": ["https://rrgtfsfeeds.s3.amazonaws.com/gtfs_b.zip",
                 "http://web.mta.info/developers/data/nyct/bus/google_transit_brooklyn.zip"],
    "manhattan": ["https://rrgtfsfeeds.s3.amazonaws.com/gtfs_m.zip",
                  "http://web.mta.info/developers/data/nyct/bus/google_transit_manhattan.zip"],
    "queens": ["https://rrgtfsfeeds.s3.amazonaws.com/gtfs_q.zip",
               "http://web.mta.info/developers/data/nyct/bus/google_transit_queens.zip"],
    "staten_island": ["https://rrgtfsfeeds.s3.amazonaws.com/gtfs_si.zip",
                      "http://web.mta.info/developers/data/nyct/bus/google_transit_staten_island.zip"],
    "mta_bus_co": ["https://rrgtfsfeeds.s3.amazonaws.com/gtfs_busco.zip",
                   "http://web.mta.info/developers/data/busco/google_transit.zip"],
}
# =============================================================================

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# ------------------------------ helpers --------------------------------------
def get(url, params=None, tries=3, headers=None, timeout=180):
    """GET with retries; exits with a readable message when all tries fail."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}: {r.text[:300]}"
        except requests.RequestException as e:
            last = str(e)
        print(f"    attempt {attempt} failed ({last[:200]}); retrying...")
        time.sleep(3 * attempt)
    sys.exit(f"Request failed after {tries} tries.\nURL: {url}\nParams: {params}\nLast error: {last}")


def try_get(url, tries=2, timeout=180):
    """Like get() but returns None instead of exiting."""
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def norm(x):
    return str(x).strip().upper()


def borough_of(route):
    r = norm(route)
    if r.startswith(("BXM", "BM", "QM", "SIM", "X")):
        return "Express"
    if r.startswith("BX"):
        return "Bronx"
    if r.startswith("B"):
        return "Brooklyn"
    if r.startswith("M"):
        return "Manhattan"
    if r.startswith("Q"):
        return "Queens"
    if r.startswith("S"):
        return "Staten Island"
    return "Other"


def socrata_csv(params):
    headers = {}
    tok = os.environ.get("SOCRATA_APP_TOKEN")
    if tok:
        headers["X-App-Token"] = tok
    r = get(MTA_URL, params=params, headers=headers or None)
    return pd.read_csv(io.StringIO(r.text))


def ts(x):
    return pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%S")


# ------------------------------ GTFS (source 2) ------------------------------
def read_txt(z, name, cols):
    with z.open(name) as f:
        return pd.read_csv(f, dtype=str, usecols=lambda c: c in cols, keep_default_na=False)


def hhmmss_to_secs(s):
    p = s.astype(str).str.strip().str.split(":", expand=True).reindex(columns=[0, 1, 2])
    p = p.apply(pd.to_numeric, errors="coerce")
    return p[0] * 3600 + p[1] * 60 + p[2]


def services_on_date(cal, cdates, d):
    """Exact GTFS rule for one date: base calendar (weekday flag + date range),
    then calendar_dates additions (type 1) and removals (type 2)."""
    svcs = set()
    if cal is not None:
        m = cal[(cal["s"] <= d) & (cal["e"] >= d) & (cal[DAYS[d.weekday()]] > 0)]
        svcs = set(m["service_id"])
    if cdates is not None:
        x = cdates[cdates["d"] == d]
        svcs |= set(x.loc[x["exception_type"] == "1", "service_id"])
        svcs -= set(x.loc[x["exception_type"] == "2", "service_id"])
    return svcs


def services_by_daytype(z, names, ref, trips, n_samples=8):
    """Pick which service_ids represent a typical weekday / Saturday / Sunday.

    For each day type we resolve the services that really run on each of the next
    `n_samples` Wednesdays / Saturdays / Sundays from `ref` (calendar + exceptions),
    count their trips, and take the MEDIAN date, so holidays and superseded schedules
    do not inflate the numbers."""
    cal = cdates = None
    if "calendar.txt" in names:
        with z.open("calendar.txt") as f:
            cal = pd.read_csv(f, dtype=str, keep_default_na=False)
        for d in DAYS:
            cal[d] = pd.to_numeric(cal[d], errors="coerce").fillna(0).astype(int)
        cal["s"] = pd.to_datetime(cal["start_date"], format="%Y%m%d", errors="coerce")
        cal["e"] = pd.to_datetime(cal["end_date"], format="%Y%m%d", errors="coerce")
    if "calendar_dates.txt" in names:
        with z.open("calendar_dates.txt") as f:
            cdates = pd.read_csv(f, dtype=str, keep_default_na=False)
        cdates["d"] = pd.to_datetime(cdates["date"], format="%Y%m%d", errors="coerce")
    if cal is None and cdates is None:
        raise RuntimeError("neither calendar.txt nor calendar_dates.txt found")

    trips_per_service = trips.groupby("service_id").size()
    ref = pd.Timestamp(ref).normalize()
    out, chosen = {}, {}
    for dtype, dow in (("weekday", 2), ("saturday", 5), ("sunday", 6)):
        first = ref + pd.Timedelta(days=(dow - ref.weekday()) % 7)
        cands = []
        for k in range(n_samples):
            d = first + pd.Timedelta(days=7 * k)
            sv = services_on_date(cal, cdates, d)
            if sv:
                cands.append((int(trips_per_service.reindex(list(sv)).fillna(0).sum()), d, sv))
        if not cands:  # feed expired/not started: look backwards from ref instead
            for k in range(1, n_samples + 1):
                d = first - pd.Timedelta(days=7 * k)
                sv = services_on_date(cal, cdates, d)
                if sv:
                    cands.append((int(trips_per_service.reindex(list(sv)).fillna(0).sum()), d, sv))
        if not cands:
            raise RuntimeError(f"no services found for a typical {dtype}")
        cands.sort(key=lambda t: t[0])
        total, d, sv = cands[len(cands) // 2]
        out[dtype] = sv
        chosen[dtype] = (d.date().isoformat(), len(sv), total)
    print("      service day used (date, #services, trips/day all routes): "
          + "; ".join(f"{k}={v}" for k, v in chosen.items()))
    return out


def parse_feed(zip_path, ref_date):
    """Return (route_info, hourly_trips) for one GTFS zip."""
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        missing = {"routes.txt", "trips.txt", "stop_times.txt"} - names
        if missing:
            raise RuntimeError(f"missing files: {sorted(missing)}")

        routes = read_txt(z, "routes.txt", {"route_id", "route_short_name"})
        if "route_short_name" not in routes:
            routes["route_short_name"] = ""
        key = np.where(routes["route_short_name"].str.strip() != "",
                       routes["route_short_name"], routes["route_id"])
        routes["route_key"] = pd.Series(key, index=routes.index).map(norm)

        trips = read_txt(z, "trips.txt", {"route_id", "service_id", "trip_id", "direction_id"})
        if "direction_id" not in trips:
            trips["direction_id"] = "0"
        trips = trips.merge(routes[["route_id", "route_key"]], on="route_id", how="inner")
        trip2route = trips.drop_duplicates("trip_id").set_index("trip_id")["route_key"]

        svc_map = services_by_daytype(z, names, ref_date, trips)

        firsts, counts = [], []
        stops_per_route = {}
        with z.open("stop_times.txt") as f:
            for chunk in pd.read_csv(f, dtype=str, keep_default_na=False, chunksize=1_000_000,
                                     usecols=lambda c: c in {"trip_id", "departure_time",
                                                             "stop_sequence", "stop_id"}):
                chunk["seq"] = pd.to_numeric(chunk["stop_sequence"], errors="coerce")
                chunk["secs"] = hhmmss_to_secs(chunk["departure_time"])
                chunk["route_key"] = chunk["trip_id"].map(trip2route)
                chunk = chunk.dropna(subset=["route_key", "seq", "secs"])
                if chunk.empty:
                    continue
                for rk, st in chunk.groupby("route_key")["stop_id"].unique().items():
                    stops_per_route.setdefault(rk, set()).update(st.tolist())
                idx = chunk.groupby("trip_id")["seq"].idxmin()
                firsts.append(chunk.loc[idx, ["trip_id", "seq", "secs"]])
                counts.append(chunk.groupby("trip_id").size().rename("n"))
        if not firsts:
            raise RuntimeError("stop_times.txt gave no usable rows")

        first = pd.concat(firsts).sort_values("seq").drop_duplicates("trip_id")
        cnt = pd.concat(counts).groupby(level=0).sum().rename("n_stops_trip").reset_index()
        cnt.columns = ["trip_id", "n_stops_trip"]

        tr = first.merge(trips[["trip_id", "route_key", "service_id", "direction_id"]],
                         on="trip_id", how="inner").merge(cnt, on="trip_id", how="left")
        tr["hour"] = ((tr["secs"] // 3600) % 24).astype(int)

        hourly = []
        for dtype, svcs in svc_map.items():
            sub = tr[tr["service_id"].isin(svcs)]
            g = sub.groupby(["route_key", "hour"]).size().rename("trips").reset_index()
            g["day_type"] = dtype
            hourly.append(g)
        hourly = pd.concat(hourly, ignore_index=True)

        info = tr.groupby("route_key").agg(
            gtfs_max_stops_per_trip=("n_stops_trip", "max"),
            gtfs_n_directions=("direction_id", "nunique")).reset_index()
        info["gtfs_n_stops"] = info["route_key"].map(lambda k: len(stops_per_route.get(k, ())))
    return info, hourly


def get_gtfs_zips():
    GTFS_DIR.mkdir(exist_ok=True)
    local = sorted(GTFS_DIR.glob("*.zip"))
    if local:
        print(f"      using {len(local)} GTFS zip(s) already in {GTFS_DIR}/")
        return local
    got = []
    for name, urls in GTFS_CANDIDATES.items():
        ok = False
        for u in urls:
            r = try_get(u)
            if r is not None and r.content[:2] == b"PK":
                path = GTFS_DIR / f"{name}.zip"
                path.write_bytes(r.content)
                print(f"      downloaded {name} ({len(r.content) / 1e6:.1f} MB) from {u}")
                got.append(path)
                ok = True
                break
        if not ok:
            print(f"      could not download GTFS feed '{name}'")
    return got


def build_supply(ref_date):
    zips = get_gtfs_zips()
    if not zips:
        return None
    infos, hourlies = [], []
    for z in zips:
        try:
            info, hourly = parse_feed(z, ref_date)
            infos.append(info)
            hourlies.append(hourly)
            print(f"      parsed {z.name}: {info['route_key'].nunique()} routes")
        except Exception as e:  # keep going with the other feeds
            print(f"      WARNING: skipped {z.name}: {e}")
    if not infos:
        return None
    info = (pd.concat(infos).groupby("route_key", as_index=False)
            .agg(gtfs_max_stops_per_trip=("gtfs_max_stops_per_trip", "max"),
                 gtfs_n_directions=("gtfs_n_directions", "max"),
                 gtfs_n_stops=("gtfs_n_stops", "max")))
    all_keys = pd.concat([i["route_key"] for i in infos])
    dup = sorted(set(all_keys[all_keys.duplicated()]))
    if dup:
        print(f"      WARNING: {len(dup)} route(s) appear in more than one GTFS feed ({dup[:6]}). "
              "Using the larger count instead of adding them, in case the same feed was added twice.")
    # max (not sum) across feeds: protects against the same feed being present twice
    hourly = pd.concat(hourlies).groupby(["route_key", "day_type", "hour"], as_index=False)["trips"].max()

    # complete grid: every route x day type x hour (no trips -> 0)
    grid = pd.MultiIndex.from_product(
        [info["route_key"], ["weekday", "saturday", "sunday"], range(24)],
        names=["route_key", "day_type", "hour"]).to_frame(index=False)
    sup = grid.merge(hourly, on=["route_key", "day_type", "hour"], how="left")
    sup["sched_trips_hr"] = sup["trips"].fillna(0).astype(int)
    # trips are counted for BOTH directions together, so the headway per direction is
    # 60 / (trips / n_directions)
    sup = sup.merge(info[["route_key", "gtfs_n_directions"]], on="route_key", how="left")
    dirs = pd.to_numeric(sup["gtfs_n_directions"], errors="coerce").fillna(1).clip(1, 2)
    per_dir = sup["sched_trips_hr"] / dirs
    sup["sched_headway_min"] = np.where(per_dir > 0, 60.0 / per_dir.clip(lower=1e-9), HEADWAY_CAP_MIN)
    sup["sched_headway_min"] = sup["sched_headway_min"].clip(upper=HEADWAY_CAP_MIN)
    sup = sup.drop(columns="gtfs_n_directions")
    sup["sched_trips_day"] = sup.groupby(["route_key", "day_type"])["sched_trips_hr"].transform("sum")
    sup = sup.drop(columns="trips")
    sup.to_csv(RAW_DIR / "gtfs_supply_table.csv", index=False)
    return info, sup


def match_route(r, keys):
    """Map a ridership route name (e.g. 'M15+') to a GTFS route key."""
    r = norm(r)
    for cand in (r, r.replace("+", "-SBS"), r.replace("+", ""), r.replace("-SBS", "+")):
        if cand in keys:
            return cand
    return None


# ------------------------------ MTA ridership (source 1) ---------------------
def pick_routes(gtfs_keys):
    w0, w1 = WINDOWS[ROUTE_PICK_WINDOW]
    print(f"[2/6] Ranking routes by ridership ({w0} to {w1}) ...")
    top = socrata_csv({
        "$select": f"{COL_ROUTE},sum({COL_RIDERS}) as total",
        "$where": f"{COL_TIME} >= '{ts(w0)}' AND {COL_TIME} < '{ts(pd.Timestamp(w1) + pd.Timedelta(days=1))}'",
        "$group": COL_ROUTE, "$order": "total DESC", "$limit": N_CANDIDATES})
    if COL_ROUTE not in top.columns:
        sys.exit(f"Column '{COL_ROUTE}' not found. Got: {list(top.columns)}. "
                 f"Open {MTA_URL}?$limit=5 in a browser and fix COL_* in the CONFIG section.")
    cands = top[COL_ROUTE].astype(str).tolist()

    def base(r):  # 'M15+' and 'M15-SBS' are the same corridor as 'M15'
        return re.sub(r"(\+|-SBS)$", "", norm(r))

    chosen, per_b, unmatched, seen = {}, {}, [], set()
    for r in cands:
        key = match_route(r, gtfs_keys) if gtfs_keys is not None else norm(r)
        if key is None:
            unmatched.append(r)
            continue
        b = borough_of(r)
        if per_b.get(b, 0) >= MAX_PER_BOROUGH or base(r) in seen:
            continue
        chosen[r] = key
        seen.add(base(r))
        per_b[b] = per_b.get(b, 0) + 1
        if len(chosen) == N_ROUTES:
            break
    if len(chosen) < N_ROUTES:  # relax the borough cap (still no duplicate corridors)
        for r in cands:
            key = match_route(r, gtfs_keys) if gtfs_keys is not None else norm(r)
            if key is not None and r not in chosen and base(r) not in seen:
                chosen[r] = key
                seen.add(base(r))
            if len(chosen) == N_ROUTES:
                break
    if not chosen:
        sample = sorted(gtfs_keys)[:15] if gtfs_keys else []
        sys.exit("No ridership route matched a GTFS route.\n"
                 f"  ridership routes: {cands[:10]}\n  GTFS route keys : {sample}\n"
                 "Fix the naming (see match_route) or rerun with --no-gtfs.")
    if unmatched:
        print(f"      note: {len(unmatched)} busy routes had no GTFS match and were skipped: {unmatched[:8]}")
    print("      chosen routes:", {r: borough_of(r) for r in chosen})
    return chosen


def fetch_ridership(chosen):
    print(f"[3/6] Fetching hourly ridership for {len(chosen)} routes x {len(WINDOWS)} windows ...")
    route_list = ",".join(f"'{r}'" for r in chosen)
    frames = []
    for i, (w0, w1) in enumerate(WINDOWS):
        start = pd.Timestamp(w0) - pd.Timedelta(days=WARMUP_DAYS)
        end = pd.Timestamp(w1) + pd.Timedelta(days=1)
        df = socrata_csv({
            "$select": f"{COL_ROUTE},{COL_TIME},sum({COL_RIDERS}) as {COL_RIDERS}",
            "$where": f"{COL_ROUTE} in({route_list}) AND {COL_TIME} >= '{ts(start)}' AND {COL_TIME} < '{ts(end)}'",
            "$group": f"{COL_ROUTE},{COL_TIME}",
            "$order": f"{COL_TIME},{COL_ROUTE}",
            "$limit": 50000})
        for c in (COL_ROUTE, COL_TIME, COL_RIDERS):
            if c not in df.columns:
                sys.exit(f"Column '{c}' not found. Got: {list(df.columns)}")
        df[COL_TIME] = pd.to_datetime(df[COL_TIME])
        df[COL_RIDERS] = pd.to_numeric(df[COL_RIDERS], errors="coerce")
        df = df.dropna(subset=[COL_RIDERS])
        df["window_id"] = i
        df["win_start"] = pd.Timestamp(w0)
        df["grid_start"] = start
        df["grid_end"] = end - pd.Timedelta(hours=1)
        print(f"      window {i}: {w0} to {w1}: {len(df)} rows fetched (incl. {WARMUP_DAYS}-day warm-up)")
        if len(df) >= 50000:
            print("      WARNING: hit the 50,000-row cap; shorten the window or use fewer routes.")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw.drop(columns=["win_start", "grid_start", "grid_end"]).to_csv(RAW_DIR / "mta_ridership_raw.csv", index=False)
    return raw


# ------------------------------ weather (source 3) ---------------------------
def fetch_weather(start, end):
    print("[4/6] Fetching hourly weather from Open-Meteo ...")
    for variables in (WEATHER_FULL, WEATHER_CORE):
        try:
            r = requests.get(WEATHER_URL, params={
                "latitude": LAT, "longitude": LON,
                "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(end).strftime("%Y-%m-%d"),
                "hourly": ",".join(variables), "timezone": "America/New_York"}, timeout=180)
            if r.status_code == 200:
                w = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": COL_TIME})
                w[COL_TIME] = pd.to_datetime(w[COL_TIME])
                w.to_csv(RAW_DIR / "weather_raw.csv", index=False)
                print(f"      got {len(w)} hourly rows, {len(variables)} variables")
                return w, variables
            print(f"      weather request failed (HTTP {r.status_code}): {r.text[:200]}")
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"      weather request error: {e}")
    sys.exit("Could not download weather from Open-Meteo. Check your connection and rerun "
             "(weather is the third required source).")


# ------------------------------ calendar -------------------------------------
def school_closed_dates():
    days = set(pd.to_datetime(SCHOOL_CLOSED_DAYS).date)
    for a, b in SCHOOL_CLOSED_RANGES:
        days.update(d.date() for d in pd.date_range(a, b))
    return days


def us_ny_holidays(years):
    try:
        return holidays.country_holidays("US", subdiv="NY", years=years)
    except Exception:
        return holidays.US(years=years)


# ------------------------------ feature building -----------------------------
def build_dataset(raw, chosen, supply, weather, weather_vars):
    print("[5/6] Building features ...")
    pieces = []
    for (route, wid), g in raw.groupby([COL_ROUTE, "window_id"]):
        idx = pd.date_range(g["grid_start"].iloc[0], g["grid_end"].iloc[0],
                            freq=pd.Timedelta(hours=1), name=COL_TIME)
        s = g.set_index(COL_TIME)[COL_RIDERS].groupby(level=0).sum().reindex(idx)
        s = s.fillna(0)  # assumption: no record for an hour = no boardings recorded
        f = s.rename(COL_RIDERS).reset_index()
        f[COL_ROUTE] = route
        f["window_id"] = wid
        # history features, per route and window, using ONLY past values
        f["lag_1h"] = f[COL_RIDERS].shift(1)
        f["lag_24h"] = f[COL_RIDERS].shift(24)
        f["lag_168h"] = f[COL_RIDERS].shift(168)
        f["roll_24h_mean"] = f[COL_RIDERS].shift(1).rolling(24).mean()
        f["roll_168h_mean"] = f[COL_RIDERS].shift(1).rolling(168).mean()
        f["_keep"] = f[COL_TIME] >= g["win_start"].iloc[0]   # drop warm-up days
        pieces.append(f)
    df = pd.concat(pieces, ignore_index=True)
    df = df[df["_keep"]].drop(columns="_keep")

    # calendar
    t = df[COL_TIME]
    df["hour"] = t.dt.hour
    df["day_of_week"] = t.dt.day_of_week
    df["month"] = t.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_am_peak"] = df["hour"].between(7, 9).astype(int)
    df["is_pm_peak"] = df["hour"].between(16, 19).astype(int)
    hol = us_ny_holidays(sorted(t.dt.year.unique().tolist()))
    df["is_holiday"] = t.dt.date.isin(set(hol.keys())).astype(int)
    closed = school_closed_dates()
    df["is_school_day"] = ((df["day_of_week"] < 5) & ~t.dt.date.isin(closed)).astype(int)

    # route-level features
    df["borough"] = df[COL_ROUTE].map(borough_of)
    df["_route_key"] = df[COL_ROUTE].map(chosen)
    df["_day_type"] = np.select([df["day_of_week"] < 5, df["day_of_week"] == 5], ["weekday", "saturday"], "sunday")

    route_cols, supply_cols = ["borough"], []
    if supply is not None:
        info, sup = supply
        df = df.merge(info, left_on="_route_key", right_on="route_key", how="left").drop(columns="route_key")
        df = df.merge(sup, left_on=["_route_key", "_day_type", "hour"],
                      right_on=["route_key", "day_type", "hour"], how="left").drop(columns=["route_key", "day_type"])
        route_cols += ["gtfs_n_stops", "gtfs_max_stops_per_trip", "gtfs_n_directions"]
        supply_cols = ["sched_trips_hr", "sched_headway_min", "sched_trips_day"]

    # weather
    df = df.merge(weather, on=COL_TIME, how="left")
    miss = int(df[weather_vars].isna().any(axis=1).sum())
    print(f"      rows with missing weather after join: {miss}")

    df = df.drop(columns=["_route_key", "_day_type"])
    df = df.sort_values([COL_TIME, COL_ROUTE]).reset_index(drop=True)

    groups = {
        "route": [COL_ROUTE] + route_cols,
        "supply": supply_cols,
        "weather": list(weather_vars),
        "calendar": ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
                     "is_am_peak", "is_pm_peak", "is_holiday", "is_school_day"],
        "history": ["lag_1h", "lag_24h", "lag_168h", "roll_24h_mean", "roll_168h_mean"],
    }
    groups = {k: v for k, v in groups.items() if v}
    ordered = [COL_TIME, "window_id", COL_RIDERS] + [c for v in groups.values() for c in v]
    return df[ordered], groups


def write_sources(df, groups, used_gtfs, chosen, weather_vars):
    today = dt.date.today().isoformat()
    lines = [
        "# Data sources and collection log", "",
        f"Dataset built on: {today}", "",
        "## 1. MTA Bus Hourly Ridership: Beginning 2025",
        "- Publisher: MTA via NY Open Data (data.ny.gov), dataset id gxb3-akrn",
        f"- Method: Socrata SODA API (`{MTA_URL}`) with server-side filtering and aggregation; "
        "ridership summed over payment method and fare class to one row per route per hour",
        f"- Windows: {', '.join(a + ' to ' + b for a, b in WINDOWS)} (plus {WARMUP_DAYS} warm-up days each for lag features)",
        f"- Routes: {', '.join(chosen)}",
        "- Licence: the dataset page states that no licence information was provided and that it is "
        "covered by different Terms of Use than Data.gov. Cite the NY Open Data terms of use.", "",
        "## 2. MTA GTFS static schedule (bus)",
        f"- Used: {'yes' if used_gtfs else 'NO (skipped with --no-gtfs or download failed)'}",
        "- Source: https://www.mta.info/developers (static GTFS feeds for the bus system)",
        "- Method: trips per route per hour of day, split by weekday / Saturday / Sunday, from the first "
        "departure of every trip; plus stops, route length proxy and directions per route",
        "- Caveat: the feed is a snapshot of the schedule valid when it was downloaded, which may differ "
        "from the 2025 service actually operated.", "",
        "## 3. Open-Meteo historical weather",
        "- Source: https://open-meteo.com (archive API), New York City coordinates",
        f"- Variables: {', '.join(weather_vars)}",
        "- Licence: check https://open-meteo.com/en/terms and cite accordingly.", "",
        "## Calendar features",
        "- Holidays: Python `holidays` package (US, NY subdivision)",
        "- School days: hand-typed from the NYC public school calendar (https://schools.nyc.gov); verify dates.", "",
        "## Final dataset",
        f"- Rows: {len(df)}; columns: {df.shape[1]}; features: {df.shape[1] - 3} "
        "(excluding timestamp, window_id and the target `ridership`)",
        "- Personal identifiers: none (aggregated ridership counts only).",
    ]
    (OUT_DIR / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gtfs", action="store_true")
    ap.add_argument("--ref-date", default=dt.date.today().isoformat())
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    supply = None
    if args.no_gtfs:
        print("[1/6] GTFS skipped (--no-gtfs)")
    else:
        print("[1/6] Loading MTA GTFS schedule (supply features) ...")
        supply = build_supply(args.ref_date)
        if supply is None:
            sys.exit("\nCould not get any GTFS feed. Either put the bus GTFS .zip files in ./gtfs_zips/ "
                     "(download them from https://www.mta.info/developers) and rerun, "
                     "or rerun with --no-gtfs to build the dataset without the schedule source.")

    keys = set(supply[0]["route_key"]) if supply else None
    chosen = pick_routes(keys)
    if supply:  # sanity check: do the schedule numbers look plausible for the chosen routes?
        sup = supply[1]
        print("      GTFS schedule summary (weekday; trips count both directions together):")
        for r, k in chosen.items():
            w = sup[(sup["route_key"] == k) & (sup["day_type"] == "weekday")]
            print(f"        {r:<7} trips/day={int(w['sched_trips_hr'].sum()):>4}  "
                  f"peak trips/hour={int(w['sched_trips_hr'].max()):>3}  "
                  f"(about one bus every {w['sched_headway_min'].min():.1f} min per direction at peak)")
        print("      Spot-check one route against the MTA's published timetable if you want to be sure.")
    raw = fetch_ridership(chosen)

    lo = raw["grid_start"].min()
    hi = raw["grid_end"].max()
    weather, wvars = fetch_weather(lo, hi)

    df, groups = build_dataset(raw, chosen, supply, weather, wvars)

    print("[6/6] Saving ...")
    df.to_csv(OUT_DIR / "transit_dataset.csv", index=False)
    (OUT_DIR / "feature_groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    write_sources(df, groups, supply is not None, chosen, wvars)

    n_feat = df.shape[1] - 3
    print(f"\nSaved {OUT_DIR / 'transit_dataset.csv'}: {len(df)} rows, {df.shape[1]} columns "
          f"({n_feat} features + target '{COL_RIDERS}')")
    for g, cols in groups.items():
        print(f"   {g:<8} {len(cols):>2} features")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    print("   columns with missing values:", dict(nulls) if len(nulls) else "none")
    if len(df) < 5000:
        print("   WARNING: fewer than 5,000 rows. Increase N_ROUTES or lengthen the windows.")
    print("\nNext: split by TIME (oldest 70% train / next 15% validation / newest 15% test).")


if __name__ == "__main__":
    main()