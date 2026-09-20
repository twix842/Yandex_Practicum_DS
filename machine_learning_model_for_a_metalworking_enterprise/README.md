# Gold Recovery Prediction

## Objective
Predict gold-recovery efficiency at different stages of an industrial ore-processing workflow.

## Approach
The analysis included validation of the recovery calculation, comparison of train and test features, investigation of metal concentrations across processing stages, assessment of feed-particle distributions, and regression modeling with cross-validation.

Two recovery targets were predicted, and model quality was assessed using the project-specific weighted sMAPE metric.

**Models:** Random Forest Regressor, Decision Tree Regressor, Linear Regression, and a constant baseline.

## Results
Random Forest provided the strongest cross-validation result.

- Cross-validation final sMAPE: **7.70**
- Test final sMAPE: **6.44**
- Constant baseline test final sMAPE: **7.05**

## Skills demonstrated
Regression · industrial process data · exploratory analysis · custom metrics · cross-validation · Random Forest · baseline comparison
