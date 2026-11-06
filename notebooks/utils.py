import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
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


WEATHER_DROPPED = ["apparent_temperature", "relative_humidity_2m", "dew_point_2m", "pressure_msl", "cloud_cover", "wind_gusts_10m"]


def planning_features(df, fewer_weather=True):
    drop = HISTORY + (WEATHER_DROPPED if fewer_weather else [])
    return [c for c in df.columns if c not in NOT_INPUTS + drop]


TYPICAL_KEYS = ["bus_route", "hour", "is_weekend"]


class TypicalRidership(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        stats = pd.DataFrame({"y": np.asarray(y)}, index=X.index).join(X[TYPICAL_KEYS]).groupby(TYPICAL_KEYS)["y"].agg(["sum", "count"])
        self.sum_, self.count_, self.global_mean_ = stats["sum"], stats["count"], float(np.mean(y))
        return self

    def _lookup(self, X):
        index = pd.MultiIndex.from_frame(X[TYPICAL_KEYS])
        return self.sum_.reindex(index).to_numpy(), self.count_.reindex(index).to_numpy()

    def transform(self, X):
        total, count = self._lookup(X)
        X = X.copy()
        X["typical"] = np.where(count > 0, total / np.where(count > 0, count, 1), self.global_mean_)
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        total, count = self._lookup(X)
        loo = (total - np.asarray(y)) / np.where(count > 1, count - 1, 1)
        X = X.copy()
        X["typical"] = np.where(count > 1, loo, self.global_mean_)
        return X


def make_pipeline(model, features, scale=False, typical=False):
    cat = [c for c in features if c in CATEGORY]
    num = [c for c in features if c not in CATEGORY] + (["typical"] if typical else [])
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler() if scale else "passthrough", num),
    ], sparse_threshold=0)
    steps = [("typical", TypicalRidership())] if typical else []
    return Pipeline(steps + [("prep", prep), ("model", model)])


def final_model():
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500, max_leaf_nodes=63, random_state=42)


def best_forest():
    return RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=1, random_state=42, n_jobs=-1)


def scores(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
