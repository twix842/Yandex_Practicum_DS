# Toxic Comment Classification

The task was to build a moderation model that flags toxic comments.

I used TF-IDF features and compared several classifiers, including Logistic Regression, Ridge, SGD and LightGBM. Model selection was based mainly on F1 because the target classes were imbalanced.

Logistic Regression gave the best final result.

- F1 on the test set: **0.78**
- ROC-AUC on the test set: **0.87**

The repository contains the original notebook and a shorter `analysis.py` with the main modeling workflow.
