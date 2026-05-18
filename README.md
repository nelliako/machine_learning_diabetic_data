# Predicting 30-day Diabetic Hospital Readmission: A Replication Study

This repository contains a replication and extension of the research conducted by Emi-Johnson and Nkrumah (2025), exploring the extent to which Electronic Health Record (EHR) data can predict early (<30-day) hospital readmission for diabetic patients using supervised machine learning.

## Overview & Objective
The goal of this study is to forecast early hospital readmissions for diabetic patients to assist healthcare facilities in resource planning and improving patient safety. The study evaluates baseline models against ensemble methods and deep neural networks to determine the most effective architecture for handling imbalanced clinical data.

## Dataset Summary
- **Source:** UCI Diabetes 130-US Hospitals dataset (longitudinal records across 130 US hospitals from 1999–2008).
- **Size:** 101,766 unique patient encounters (after data cleaning).
- **Features:** Over 50 clinical and demographic features (e.g., age, race, time in hospital, number of medications, and historical rolling care intensity).
- **Target Variable:** 30-day readmission status (Class 1 = Readmitted within 30 days, Class 0 = Not readmitted).
- **Class Imbalance:** Significant skew, with an 11.2% base readmission rate (72,273 non-readmissions vs. 9,139 readmissions in the training set).

## Methodology & Preprocessing
1. **Data Cleaning:** Replaced missing markers (`?`) with `NaN`. Features with excessive missingness (`weight` at 97%, `payer_code` at 40%) and single-drug columns were excluded.
2. **Feature Engineering:** - Grouped ICD-9 diagnosis codes into clinical categories (e.g., Circulatory, Respiratory, Diabetes).
   - Generated rolling/cumulative historical metrics (`cum_sum`) for prior inpatient visits, total medications, and days in hospital to evaluate longitudinal care intensity without introducing temporal data leakage.
3. **Pipeline Construction:** Handled heterogeneous data using a scikit-learn `ColumnTransformer`. Implemented mode imputation (`SimpleImputer`) and `OneHotEncoder` for categorical variables, and `StandardScaler` for numerical features.
4. **Hyperparameter Tuning:** Utilized 3-fold, 3-repeat stratified cross-validation (`RepeatedStratifiedKFold` with `GridSearchCV`) optimized for the AUC-ROC metric.

## Model Performance & Key Results

Four machine learning architectures were developed and tested using scikit-learn pipelines with class weighting to address the severe class imbalance:

| Model | Accuracy | AUC-ROC | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.338 | **0.679** | 0.897 | Low |
| **Random Forest** | 0.283 | 0.660 | **0.911** | Low |
| **Logistic Regression** (Baseline) | 0.674 | 0.642 | 0.511 | 0.169 |
| **Deep Neural Network (DNN)** | **0.854** | 0.650 | 0.153 | **0.240** |

### Key Takeaways:
- **Top Performer:** **XGBoost** achieved the highest overall discriminative performance with an AUC-ROC of **0.679**, outperforming the original study's baseline (0.667) and showing a statistically significant improvement over Logistic Regression ($p = 0.019$).
- **The Precision-Recall Trade-off:** Due to aggressive class weighting, the ensemble models (XGBoost and Random Forest) heavily prioritized **Recall (~90%)** over Accuracy and Precision. This creates a "sharp" clinical model that catches almost all at-risk patients but generates high false-positive rates that could overwhelm clinical resources in practice.
- **Feature Importance (SHAP Analysis):** Model interpretability using SHAP confirmed that **discharge disposition**, **prior inpatient visits**, and **chronic medication burden** are the most influential predictors of near-term readmission.

## Limitations & Future Work
- **Computational Constraints:** Hyperparameter optimization was restricted to a 3-fold cross-validation due to hardware limits.
- **Resource Constraints for Deployment:** The high false-positive rates of high-recall models remain a barrier to real-world clinical adoption.
- **Future Directions:** Incorporating advanced sampling methodologies (e.g., SMOTE) and integrating external social determinants of health (SDOH) to balance precision and recall.

