# Executive Report: Credit Scoring Model (Bank Loan Risk)

## 1. Project Objective
The goal was to develop a predictive model to classify the risk level of new loan applicants. By identifying "Bad" risks early, the bank can minimize defaults and optimize its lending strategy.

## 2. Key Findings (Day 1: Underwriting Analysis)
*   **Credit Amount & Duration:** There is a strong correlation between high loan amounts, long durations, and increased default risk.
*   **Demographics:** Older applicants tend to show more stability, while younger applicants (under 25) have a slightly higher variance in risk.
*   **Housing:** Applicants who own their homes are statistically lower risk compared to those who rent or have free housing.

## 3. Methodology
*   **Data Preprocessing:** Handled missing values in 'Saving accounts' and 'Checking account' (imputing as 'none').
*   **Feature Engineering:** Scaled numerical values and one-hot encoded categorical variables.
*   **Class Imbalance:** Addressed using **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model learns to identify "Bad" risks effectively, despite being the minority class.
*   **Algorithms:** Trained **XGBoost** and **CatBoost** classifiers. XGBoost was selected for the final deployment due to its robust performance on this dataset.

## 4. Model Performance
*   **Accuracy:** ~75-80% on test data.
*   **Recall (Bad Risk):** Significantly improved after SMOTE, allowing the bank to catch more potential defaulters.
*   **Precision:** Maintained at a level that avoids rejecting too many "Good" customers.

## 5. Deployment (Day 4)
*   The model is deployed via a **FastAPI** service.
*   Integration is seamless via a POST endpoint `/predict`.
*   Real-time scoring allows loan officers to get instant risk feedback during the application process.

## 6. Business Recommendations
1.  **Stricter Rules for High Amounts:** For loans exceeding 5,000 DM with durations over 36 months, additional documentation should be required regardless of model score.
2.  **Focus on Savings:** Applicants with 'little' or 'none' in their savings/checking accounts should be offered smaller initial loan limits.
3.  **Automation:** Integrate the API into the loan application portal to automate 80% of low-risk approvals.
