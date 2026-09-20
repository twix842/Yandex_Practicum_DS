"""Modeling stage from the gold-recovery prediction project.

The full notebook in this folder includes process-data validation and exploratory analysis.
"""

import numpy as np

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 12345


def smape(target, prediction):
    score = abs(target - prediction) / ((abs(target) + abs(prediction)) / 2) * 100
    score = score.fillna(method="bfill")
    return np.mean(score)


def total_smape(rougher_smape, final_smape):
    return 0.25 * rougher_smape + 0.75 * final_smape


smape_scorer = make_scorer(smape, greater_is_better=False)

linear_model = LinearRegression()

rougher_linear = abs(
    cross_val_score(
        linear_model,
        features_rougher_train,
        target_rougher_train,
        scoring=smape_scorer,
        cv=5,
    ).mean()
)

final_linear = abs(
    cross_val_score(
        linear_model,
        features_final_train,
        target_final_train,
        scoring=smape_scorer,
        cv=5,
    ).mean()
)

print(f"Linear Regression CV total sMAPE: {total_smape(rougher_linear, final_linear):.3f}")

forest_params = {
    "max_depth": list(range(2, 15, 3)),
    "n_estimators": list(range(1, 201, 25)),
}

rougher_search = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE),
    forest_params,
    scoring=smape_scorer,
    cv=5,
)
rougher_search.fit(features_rougher_train, target_rougher_train)

final_search = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE),
    forest_params,
    scoring=smape_scorer,
    cv=5,
)
final_search.fit(features_final_train, target_final_train)

cv_total = total_smape(
    abs(rougher_search.best_score_),
    abs(final_search.best_score_),
)
print(f"Random Forest CV total sMAPE: {cv_total:.3f}")

rougher_model = RandomForestRegressor(
    random_state=RANDOM_STATE,
    max_depth=5,
    n_estimators=151,
)
final_model = RandomForestRegressor(
    random_state=RANDOM_STATE,
    max_depth=5,
    n_estimators=121,
)

rougher_model.fit(features_rougher_train, target_rougher_train)
final_model.fit(features_final_train, target_final_train)

rougher_prediction = rougher_model.predict(features_rougher_test)
final_prediction = final_model.predict(features_final_test)

test_score = 0.25 * smape(target_rougher_test, rougher_prediction) + 0.75 * smape(
    target_final_test,
    final_prediction,
)

print(f"Test total sMAPE: {test_score:.3f}")
