"""Core modeling workflow from the toxic-comment classification project.

The full notebook in this folder contains the complete preprocessing and exploratory analysis.
"""

import numpy as np

from lightgbm import LGBMClassifier
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

RANDOM_STATE = 12345
CV_FOLDS = 3

target = data["toxic"]
features = data["spacy_lemmatize"]

features_train, features_valid_test, target_train, target_valid_test = train_test_split(
    features,
    target,
    test_size=0.4,
    random_state=RANDOM_STATE,
    stratify=target,
)

features_valid, features_test, target_valid, target_test = train_test_split(
    features_valid_test,
    target_valid_test,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=target_valid_test,
)

stop_words = set(stopwords.words("english"))

logistic_pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                stop_words=stop_words,
                ngram_range=(1, 2),
            ),
        ),
        ("model", LogisticRegression(random_state=42)),
    ]
)

logistic_search = GridSearchCV(
    logistic_pipeline,
    {
        "model__C": np.logspace(0.5, 5, 10),
        "model__class_weight": ["balanced", None],
    },
    scoring="f1",
    cv=CV_FOLDS,
    n_jobs=-1,
)

logistic_search.fit(features_train, target_train)

test_predictions = logistic_search.predict(features_test)
test_f1 = f1_score(target_test, test_predictions)
test_auc = roc_auc_score(target_test, test_predictions)

print("Best parameters:", logistic_search.best_params_)
print(f"Test F1: {test_f1:.2f}")
print(f"Test ROC-AUC: {test_auc:.2f}")
