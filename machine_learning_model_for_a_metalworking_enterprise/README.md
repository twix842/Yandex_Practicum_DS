# Gold Recovery Prediction

This project uses industrial process data from gold ore processing.

The main difficulty was that the target had to be predicted at two stages of the process, and the final score was calculated with a weighted sMAPE metric. Before modeling, I checked the recovery calculation, compared the train and test feature sets and examined changes in metal concentrations across processing stages.

I compared Linear Regression, Decision Tree and Random Forest models with cross-validation. A constant model was used as a baseline.

**Result**

- Random Forest cross-validation total sMAPE: **7.70**
- Test total sMAPE: **6.44**
- Constant baseline test total sMAPE: **7.05**

The full notebook includes the process-data analysis; `analysis.py` focuses on the model-selection part.
