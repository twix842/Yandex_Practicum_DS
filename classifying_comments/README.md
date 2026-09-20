# Toxic Comment Classification

## Objective
Build a model that identifies toxic user comments for automatic moderation.

## Approach
The project treats moderation as a binary text-classification problem. Text was prepared for modeling and represented with TF-IDF features. Several linear and gradient-boosting classifiers were compared using reproducible pipelines and hyperparameter search.

**Models:** Logistic Regression, Ridge Classifier, SGD Classifier, LightGBM.

**Evaluation:** F1 score was the primary metric; ROC-AUC was also examined.

## Results
Logistic Regression provided the strongest overall performance and was selected for final testing.

- Test F1: **0.78**
- Test ROC-AUC: **0.87**

The result exceeded the project requirement of F1 ≥ 0.75.

## Skills demonstrated
NLP · TF-IDF · text preprocessing · scikit-learn pipelines · GridSearchCV · binary classification · F1 · ROC-AUC · LightGBM
