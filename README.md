# Data Science Portfolio

I am a physician specialising in anaesthesiology and intensive care, with additional training in data science. This repository contains selected machine-learning projects that demonstrate practical work with classification, regression, natural-language processing, time series, uncertainty estimation, and privacy-preserving transformations.

My current interest is the use of statistics and machine learning in clinical research and perioperative medicine.

## Featured projects

| Project | Focus | Main result |
|---|---|---|
| [Customer Churn Prediction](customer_churn_prediction_for_telecom_company/) | Predicting telecom customer churn from contract, service, and billing data. | CatBoost achieved a test ROC-AUC of 0.914 and accuracy of 0.925. |
| [Toxic Comment Classification](classifying_comments/) | Classifying user comments for moderation with TF-IDF and linear models. | The selected Logistic Regression pipeline achieved test F1 = 0.78 and ROC-AUC = 0.87. |
| [Taxi Demand Forecasting](taxi_demand_forecasting/) | Forecasting the number of airport taxi orders one hour ahead. | CatBoost achieved test RMSE = 39.54, compared with 58.8 for the naive baseline. |
| [Oil Region Selection with Bootstrap Risk Analysis](oil_region_selection/) | Selecting a drilling region under budget and downside-risk constraints. | Region 2 was recommended: estimated mean profit RUB 536.4 million, 95% interval RUB 110 million–1.00 billion, and 0.3% loss risk. |
| [Gold Recovery Prediction](machine_learning_model_for_a_metalworking_enterprise/) | Predicting recovery efficiency at two stages of an ore-processing workflow. | The selected Random Forest produced test weighted sMAPE = 6.4367 versus 7.0453 for the constant baseline. |
| [Privacy-Preserving Data Transformation](privacy_preserving_data/) | Protecting personal features with an invertible linear transformation while preserving regression quality. | R² remained 0.435 before and after transformation. |

## Technical stack

Python, pandas, NumPy, scikit-learn, CatBoost, LightGBM, matplotlib, seaborn, spaCy, NLTK, statsmodels, SciPy.

## Notes

- These projects were completed during the Yandex Practicum Data Science programme.
- Portfolio notebooks were edited for clarity: reviewer correspondence and training-platform checklists were removed.
- Reported metrics come from the original completed analyses; no results were invented or inflated.
- Source datasets are not redistributed where licensing or course-access restrictions apply.
