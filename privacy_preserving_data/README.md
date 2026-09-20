# Privacy-Preserving Data Transformation

The question in this project was whether personal data could be transformed so that the original values are difficult to recover while Linear Regression keeps the same predictive quality.

I used multiplication by an invertible random matrix and checked the transformation both mathematically and experimentally. After encoding the feature matrix, I trained the same regression model on the original and transformed data and compared R².

The predictive quality was preserved after the transformation.

This project was mainly an exercise in linear algebra and its application to data protection.
