import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")
RANDOM_STATE = 12345

try:
    data = pd.read_csv(
        "/Users/kolotukhin.md/Downloads/jupyter_notebook/10/taxi.csv",
        index_col=[0],
        parse_dates=[0],
    )
except:
    data = pd.read_csv(
        "https://code.s3.yandex.net/datasets/taxi.csv",
        index_col=[0],
        parse_dates=[0],
    )

data = data.sort_index().resample("1H").sum()

decomposition = seasonal_decompose(data["num_orders"])
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
decomposition.trend.plot(ax=axes[0], title="Trend")
decomposition.seasonal.plot(ax=axes[1], title="Seasonality")
decomposition.resid.plot(ax=axes[2], title="Residuals")
plt.tight_layout()


def make_features(frame, max_lag=60, rolling_mean_size=30):
    frame = frame.copy()
    frame["hour"] = frame.index.hour
    frame["day"] = frame.index.day
    frame["day_of_week"] = frame.index.dayofweek
    frame["days_in_month"] = frame.index.days_in_month

    for lag in range(1, max_lag + 1):
        frame[f"lag_{lag}"] = frame["num_orders"].shift(lag)

    frame["rolling_mean"] = (
        frame["num_orders"]
        .shift(1)
        .rolling(rolling_mean_size)
        .mean()
    )

    return frame


model_data = make_features(data)
train, test = train_test_split(model_data, shuffle=False, test_size=0.1)
train = train.dropna()

features_train = train.drop(columns=["num_orders"])
target_train = train["num_orders"]
features_test = test.drop(columns=["num_orders"])
target_test = test["num_orders"]

tscv = TimeSeriesSplit(n_splits=10)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
    "CatBoost": CatBoostRegressor(random_state=RANDOM_STATE, verbose=False),
    "LightGBM": LGBMRegressor(random_state=RANDOM_STATE),
}

param_grids = {
    "Ridge": {"solver": ["auto", "svd", "cholesky", "lsqr"]},
    "DecisionTree": {
        "max_depth": list(range(2, 20, 3)),
        "min_samples_leaf": list(range(1, 20, 3)),
    },
    "RandomForest": {
        "max_depth": list(range(2, 15, 3)),
        "n_estimators": list(range(25, 201, 25)),
    },
    "CatBoost": {"learning_rate": np.arange(0.1, 1.1, 0.1)},
    "LightGBM": {
        "num_leaves": np.arange(2, 10, 1),
        "learning_rate": np.arange(0.1, 1.1, 0.1),
    },
}

for name, model in models.items():
    if name == "LinearRegression":
        scores = -GridSearchCV(
            model,
            param_grid={},
            scoring="neg_root_mean_squared_error",
            cv=tscv,
        ).fit(features_train, target_train).best_score_
        print(f"{name}: CV RMSE = {scores:.2f}")
        continue

    search = GridSearchCV(
        model,
        param_grids[name],
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1 if name in {"DecisionTree", "RandomForest"} else None,
    )
    search.fit(features_train, target_train)
    print(f"{name}: CV RMSE = {-search.best_score_:.2f}")
    print("Best parameters:", search.best_params_)

final_model = CatBoostRegressor(
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    verbose=False,
)
final_model.fit(features_train, target_train)

predictions = final_model.predict(features_test)
test_rmse = mean_squared_error(target_test, predictions) ** 0.5
print(f"CatBoost test RMSE: {test_rmse:.2f}")

previous_value_prediction = target_test.shift(1)
previous_value_prediction.iloc[0] = target_train.iloc[-1]
baseline_rmse = mean_squared_error(target_test, previous_value_prediction) ** 0.5
print(f"Previous-value baseline RMSE: {baseline_rmse:.2f}")
