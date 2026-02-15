# Loan Approval Prediction (Classification)

## Overview
This project builds and compares multiple machine learning classifiers to predict whether a loan application is approved (`Loan_Status`) using applicant demographic and financial attributes.

Dataset file used in the notebook: `LoanApprovalPrediction.csv`.

## Problem
Given applicant information (income, loan amount, credit history, property area, etc.), predict:
- **Target:** `Loan_Status` (Y/N)

## Data
Example columns used:
- Gender, Married, Dependents, Education, Self_Employed
- ApplicantIncome, CoapplicantIncome
- LoanAmount, Loan_Amount_Term, Credit_History
- Property_Area

## Approach
1. **Load data** (`LoanApprovalPrediction.csv`)
2. **Clean + preprocess**
   - Dropped identifier column (`Loan_ID`)
   - Encoded categorical fields using `LabelEncoder`
   - Filled missing values using mean imputation
3. **EDA**
   - Correlation heatmap
   - Categorical relationship visualization (Gender x Married, colored by Loan_Status)
4. **Modeling**
   - Train/test split: `test_size=0.4`, `random_state=1`
   - Models compared:
     - RandomForestClassifier
     - LogisticRegression
     - SVC
     - KNeighborsClassifier
5. **Evaluation**
   - Metric: accuracy on the held-out test set

## Results (Test Accuracy)
| Model | Accuracy |
|------|----------|
| RandomForestClassifier | **82.5%** |
| LogisticRegression | 80.83% |
| SVC | 69.17% |
| KNeighborsClassifier | 63.75% |

## Notes / Learnings
- Random Forest achieved the best test performance.
- Training accuracy for Random Forest was much higher than test accuracy, which can indicate overfitting. Next steps could include hyperparameter tuning (max_depth, min_samples_leaf), cross-validation, and feature scaling (especially for Logistic Regression / SVC).
- Logistic Regression triggered a convergence warning in the notebook; scaling features and increasing `max_iter` would likely help.

## Tech Stack
- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn

## How to Run
1. Clone the repo
2. Install dependencies:
   - `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open and run the notebook (or export HTML)

## Next Improvements
- Use `ColumnTransformer` + `Pipeline` with:
  - Proper categorical encoding (OneHotEncoder)
  - Separate numeric imputation strategy
  - Scaling for linear/SVM models
- Add cross-validation and additional metrics (F1, ROC-AUC)
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
