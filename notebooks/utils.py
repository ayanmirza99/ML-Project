import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HISTORY = ["lag_1h", "lag_24h", "lag_168h", "roll_24h_mean", "roll_168h_mean"]
NOT_INPUTS = ["transit_timestamp", "window_id", "month", "gtfs_n_directions", "ridership"]
CATEGORY = ["bus_route", "borough"]
TARGET = "ridership"


def load_data():
    df = pd.read_csv("../data/transit_dataset.csv", parse_dates=["transit_timestamp"])
    return df.sort_values(["transit_timestamp", "bus_route"]).reset_index(drop=True)


def split_by_days(df):
    day = df["transit_timestamp"].dt.normalize()
    all_days = np.sort(day.unique())
    a, b = int(len(all_days) * 0.70), int(len(all_days) * 0.85)
    return df[day.isin(all_days[:a])], df[day.isin(all_days[a:b])], df[day.isin(all_days[b:])]


def planning_features(df):
    return [c for c in df.columns if c not in NOT_INPUTS + HISTORY]


def make_pipeline(model, features, scale=False):
    cat = [c for c in features if c in CATEGORY]
    num = [c for c in features if c not in CATEGORY]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler() if scale else "passthrough", num),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def best_forest():
    return RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=1, random_state=42, n_jobs=-1)


def scores(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
