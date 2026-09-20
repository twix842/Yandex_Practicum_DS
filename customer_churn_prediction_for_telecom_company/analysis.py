"""Core model-comparison workflow from the telecom churn project.

The full notebook in this folder contains data integration, exploratory analysis,
feature engineering and the complete model-selection process.
"""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 100423

models = {
    "RandomForest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [50, 100, 150],
            "max_depth": [5, 10, 15],
        },
    ),
    "CatBoost": (
        CatBoostClassifier(
            random_state=RANDOM_STATE,
            verbose=False,
        ),
        {
            "learning_rate": [0.08, 0.12, 0.16],
            "depth": [4, 6, 8],
        },
    ),
    "LightGBM": (
        LGBMClassifier(random_state=RANDOM_STATE),
        {
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.05, 0.1, 0.2],
        },
    ),
}

searches = {}

for name, (model, params) in models.items():
    search = GridSearchCV(
        model,
        params,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )
    search.fit(features_train_ohe, target_train)
    searches[name] = search

    print(name)
    print("Best parameters:", search.best_params_)
    print(f"CV ROC-AUC: {search.best_score_:.3f}")
    print()

final_model = CatBoostClassifier(
    learning_rate=0.16,
    random_state=RANDOM_STATE,
    verbose=False,
)
final_model.fit(features_train_ohe, target_train)

test_probability = final_model.predict_proba(features_test_ohe)[:, 1]
test_prediction = final_model.predict(features_test_ohe)

print(f"Test ROC-AUC: {roc_auc_score(target_test, test_probability):.3f}")
print(f"Test accuracy: {accuracy_score(target_test, test_prediction):.3f}")
