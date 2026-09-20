# Taxi Demand Forecasting

## Objective
Forecast taxi demand for the next hour to support driver allocation during periods of high demand.

## Approach
The time series was resampled to hourly observations and examined for trend and seasonality. Calendar variables, lagged observations and rolling statistics were used as predictive features. Several regression models were compared with time-aware validation.

## Result
CatBoost achieved a test RMSE of **39.54**, meeting the project requirement of RMSE below 48.

## Skills demonstrated
Time series · feature engineering · lag features · rolling statistics · time-aware validation · CatBoost · RMSE
