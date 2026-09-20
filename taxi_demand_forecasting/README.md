# Taxi Demand Forecasting

The aim of this project was to predict the number of taxi orders for the next hour.

I resampled the time series to hourly observations, looked at trend and seasonality, and created calendar, lag and rolling-mean features. Because this is time-series data, validation was performed with `TimeSeriesSplit` rather than random cross-validation.

Several regression models were compared. CatBoost produced the best final result.

**Test RMSE: 39.54**

The project requirement was RMSE below 48.
