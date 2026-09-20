# Oil Region Selection

This project combines regression with a simple decision-under-uncertainty problem.

A Linear Regression model was trained for each of three candidate regions. The predictions were then used to select promising wells and estimate expected profit. I used bootstrap resampling to calculate the distribution of possible profit and estimate the probability of loss.

What I find useful about this project is that the model is not treated as the final answer. Its predictions are passed into a separate economic and risk analysis.

The code in `analysis.py` contains the main modeling and bootstrap workflow.
