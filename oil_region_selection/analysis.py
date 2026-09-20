import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 12345
BUDGET = 10_000_000_000
CANDIDATE_WELLS = 500
SELECTED_WELLS = 200
INCOME_PER_THOUSAND_BARRELS = 450_000

try:
    regions = [
        pd.read_csv("/Users/kolotukhin.md/Downloads/jupyter_notebook/7/geo_data_0.csv"),
        pd.read_csv("/Users/kolotukhin.md/Downloads/jupyter_notebook/7/geo_data_1.csv"),
        pd.read_csv("/Users/kolotukhin.md/Downloads/jupyter_notebook/7/geo_data_2.csv"),
    ]
except:
    regions = [
        pd.read_csv("https://code.s3.yandex.net/datasets/geo_data_0.csv"),
        pd.read_csv("https://code.s3.yandex.net/datasets/geo_data_1.csv"),
        pd.read_csv("https://code.s3.yandex.net/datasets/geo_data_2.csv"),
    ]


def prepare_region(df):
    df = df.drop_duplicates(subset=["id"]).drop(columns=["id"]).copy()

    features = df.drop(columns=["product"])
    target = df["product"]

    features_train, features_valid, target_train, target_valid = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    features_train = pd.DataFrame(
        scaler.fit_transform(features_train),
        columns=features_train.columns,
        index=features_train.index,
    )
    features_valid = pd.DataFrame(
        scaler.transform(features_valid),
        columns=features_valid.columns,
        index=features_valid.index,
    )

    return features_train, features_valid, target_train, target_valid


def fit_region_model(df):
    features_train, features_valid, target_train, target_valid = prepare_region(df)

    model = LinearRegression()
    model.fit(features_train, target_train)
    predictions = pd.Series(
        model.predict(features_valid),
        index=target_valid.index,
    )

    metrics = {
        "R2": r2_score(target_valid, predictions),
        "RMSE": mean_squared_error(target_valid, predictions) ** 0.5,
        "MAE": mean_absolute_error(target_valid, predictions),
        "predicted_mean": predictions.mean(),
    }

    return target_valid, predictions, metrics


results = []
for number, region in enumerate(regions, start=1):
    target_valid, predictions, metrics = fit_region_model(region)
    results.append((target_valid, predictions))

    print(f"Region {number}")
    print(f"R2: {metrics['R2']:.3f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"Mean predicted reserves: {metrics['predicted_mean']:.2f}")
    print()


minimum_reserves = BUDGET / SELECTED_WELLS / INCOME_PER_THOUSAND_BARRELS
print(f"Break-even reserves per selected well: {minimum_reserves:.2f} thousand barrels")


def profit(target, predictions, count=SELECTED_WELLS):
    best = predictions.sort_values(ascending=False).head(count)
    selected_target = target.loc[best.index]
    return INCOME_PER_THOUSAND_BARRELS * selected_target.sum() - BUDGET


for number, (target_valid, predictions) in enumerate(results, start=1):
    region_profit = profit(target_valid, predictions)
    print(f"Region {number}: profit from top {SELECTED_WELLS} wells = {region_profit / 1e9:.2f} bn RUB")


state = np.random.RandomState(RANDOM_STATE)


def bootstrap_profit(target, predictions, n_iterations=1000):
    values = []

    for _ in range(n_iterations):
        target_sample = target.sample(
            n=CANDIDATE_WELLS,
            replace=True,
            random_state=state,
        )
        prediction_sample = predictions.loc[target_sample.index]
        values.append(profit(target_sample, prediction_sample))

    values = pd.Series(values)

    return {
        "mean_profit": values.mean(),
        "lower_95": values.quantile(0.025),
        "upper_95": values.quantile(0.975),
        "loss_risk": (values < 0).mean(),
    }


for number, (target_valid, predictions) in enumerate(results, start=1):
    estimate = bootstrap_profit(target_valid, predictions)

    print(f"Region {number}")
    print(f"Mean profit: {estimate['mean_profit'] / 1e6:.2f} m RUB")
    print(
        "95% interval: "
        f"{estimate['lower_95'] / 1e9:.2f} to {estimate['upper_95'] / 1e9:.2f} bn RUB"
    )
    print(f"Risk of loss: {estimate['loss_risk']:.1%}")
    print()
