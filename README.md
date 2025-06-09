# Medical-Insurance-Premium-Prediction
A predictive model for medical insurance premiums using Python and statsmodels. Navigate to the document in `report` Folder for details.

## Overview
This project develops a predictive model for medical insurance premiums using a dataset of 986 records. The model employs exploratory data analysis (EDA), multiple linear regression with weighted least squares (WLS), and validation techniques to achieve a 69.6% explained variance.

## Dataset
- **Source**: Demo Data from the Introduction to Business Analytics Course (Available in `data` Folder)
- **Features**: `Age`, `BMI`, `AnyTransplants`, `AnyChronicDiseases`, `HistoryOfCancerInFamily`, `PremiumPrice`
- **Size**: 986 rows, split into 690 training and 296 testing observations

## Methodology
- **EDA**: Identified non-linear `Age` effects and selected key predictors.
- **Modeling**: Applied log transformation to `PremiumPrice`, centered `Age`, capped 30 outliers, and used WLS to address heteroskedasticity.
- **Validation**: Evaluated on test set with MSE 0.02 (log scale) and 12,531,667.25 (original scale).

## Results
- **R-squared**: 0.696
- **Significant Predictors**: `Age_centered`, `Age2_centered`, `BMI`, `AnyTransplants`, `AnyChronicDiseases`, `HistoryOfCancerInFamily`
- **Limitations**: Persistent heteroskedasticity (Breusch-Pagan p = 0.0000)

## Usage
- Download and open `notebooks/Medical_Premium_Prediction.ipynb` in Jupyter Notebook.
- Install dependencies: `pip install statsmodels scikit-learn pandas numpy matplotlib`

## License
All rights reserved
