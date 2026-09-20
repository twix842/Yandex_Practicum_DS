# Customer Churn Prediction

This was the final project of the Yandex Practicum Data Science program.

The goal was to predict customer churn for a telecom company using contract, service and customer data. I worked through data integration, preprocessing, feature engineering and model comparison, then evaluated the selected model on a separate test set.

I compared Random Forest, LightGBM and CatBoost. CatBoost performed best.

**Final test metrics**

- ROC-AUC: **0.914**
- Accuracy: **0.925**

I also reviewed feature importance to understand which variables contributed most to the model.

The full notebook contains the complete analysis; `analysis.py` is a compact version of the modeling stage.
