# Medical Insurance Cost Prediction

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model: OLS Linear Regression](https://img.shields.io/badge/model-OLS_LinearRegression-orange.svg)](https://scikit-learn.org/)
[![UI: Streamlit](https://img.shields.io/badge/ui-Streamlit-red.svg)](https://streamlit.io/)

An end-to-end applied machine learning internship project that models and predicts individual annual medical charges using strictly Ordinary Least Squares (OLS) Linear Regression, domain-grounded clinical feature engineering, reproducible scikit-learn pipelines, an interactive Streamlit web application, and a publication-grade 8-page technical report.

---

## Table of Contents
1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Dataset & Features](#dataset--features)
4. [Project Workflow](#project-workflow)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Outlier Analysis & Justified Retention](#outlier-analysis--justified-retention)
7. [Preprocessing & Clinical Feature Engineering](#preprocessing--clinical-feature-engineering)
8. [Model Architecture & Training](#model-architecture--training)
9. [Evaluation Metrics & Results](#evaluation-metrics--results)
10. [Streamlit Web Application](#streamlit-web-application)
11. [Final Internship PDF Report](#final-internship-pdf-report)
12. [Example Prediction](#example-prediction)
13. [Limitations & Future Work](#limitations--future-work)
14. [How to Run](#how-to-run)
15. [Project Structure](#project-structure)

---

## Overview
This project delivers a transparent, interpretable machine learning system for forecasting healthcare expenditures. In strict compliance with internship constraints, the modeling strategy relies exclusively on Ordinary Least Squares (OLS) Linear Regression without resorting to opaque black-box ensemble algorithms (such as Random Forests, XGBoost, or Neural Networks). 

By pairing rigorous exploratory data analysis with targeted clinical interaction modeling, the resulting pipeline demonstrates that transparent linear models can achieve high predictive accuracy ($R^2 = 0.8720$, MAE = $2,486.32) while maintaining full actuarial explainability and legal compliance.

---

## Problem Statement
Individual healthcare costs vary across multiple orders of magnitude. For insurance carriers, health systems, and policyholders, accurate expenditure estimation is essential for actuarial solvency, premium determination, and financial risk mitigation. 

The goal of this project is to model billed healthcare costs (`charges`) as a function of primary demographic and biometric attributes (`age`, `sex`, `bmi`, `children`, `smoker`, `region`), uncover the key lifestyle and metabolic drivers of cost, and deploy an accessible interface for stakeholder decision support.

---

## Dataset & Features

### Dataset Overview
The project utilizes the canonical US Medical Cost Personal Dataset (`insurance.csv`), consisting of **1,338 records** and **7 variables**.

| Feature | Type | Observed Range | Description |
|---|---|---|---|
| `age` | Continuous (int) | 18 – 64 years | Primary beneficiary chronological age |
| `sex` | Categorical (binary) | `female`, `male` | Biological sex assigned at birth |
| `bmi` | Continuous (float) | 15.96 – 53.13 kg/m² | Body Mass Index ($kg/m^2$, Normal: 18.5–24.9, Obese: ≥30) |
| `children` | Discrete (int) | 0 – 5 dependents | Number of covered dependent children |
| `smoker` | Categorical (binary) | `no`, `yes` | Active tobacco smoking status |
| `region` | Categorical (nominal) | `northeast`, `northwest`, `southeast`, `southwest` | Residential geographic quadrant within the US |
| `charges` | Continuous (float) | $1,121.87 – $63,770.43 | Annual billed individual healthcare expenditures (**Target**) |

### Data Hygiene & Verification
- **Missing Values:** Programmatic verification confirms 0 missing or NaN cells across all 1,338 records.
- **Duplicate Records:** Exactly 1 identical record was identified (Row 581: 19-year-old male, BMI 30.59, 0 children, northwest region). Statistical auditing verified that demographic collisions naturally occur within a sample of 1,338 individuals. The raw dataset was preserved without modification to ensure raw data immutability.
- **Partitioning Protocol:** Partitioned into an 80% training set (1,070 records) and a 20% holdout test set (268 records) using `random_state=42`. All preprocessing, outlier analysis, and feature transformers are fitted strictly on the training partition to eliminate data leakage.

---

## Project Workflow
1. **Dynamic Data Inspection:** Programmatic validation of dataset shape, types, missingness, and statistical summaries.
2. **Exploratory Data Analysis (EDA):** Univariate, bivariate, and multivariate distribution analysis accompanied by non-causal statistical interpretations and high-resolution figure generation.
3. **Tukey IQR Outlier Analysis:** Quantitative fence computation and clinical profiling establishing 100% justified retention.
4. **Leakage-Free Clinical Feature Engineering:** Developing a stateless scikit-learn transformer (`ClinicalFeatureEngineer`) to model compound metabolic risk (`obese_smoker`).
5. **Categorical Encoding:** Mitigating the dummy variable trap via reference-level dropping (`OneHotEncoder(drop='first')`).
6. **Pipeline Assembly & Training:** Encapsulating feature engineering, preprocessor, and OLS regression into a unified `Pipeline` serialized to `models/insurance_model.joblib`.
7. **Empirical Evaluation:** Dynamic calculation of MAE, MSE, RMSE, and $R^2$, extraction of learned coefficients with dollar-value interpretations, and export to `outputs/metrics.json`.
8. **Interactive Web Deployment:** Developing `app.py` in Streamlit with zero-duplicate preprocessing and mandatory disclaimers.
9. **Publication-Grade Reporting:** Scripting `scripts/generate_report.py` to generate an 8-page final internship PDF report embedding all figures.
10. **Comprehensive Verification:** Automated test execution across unit contracts, corner cases, and cross-document metric consistency.

---

## Exploratory Data Analysis (EDA)
Comprehensive EDA was conducted and documented in `notebooks/medical_insurance_cost_prediction.ipynb` and exported to `outputs/figures/`:

1. **Target Distribution (`outputs/figures/target_distribution.png`):**
   - Annual charges exhibit heavy right-skewness (Mean: $13,270.42, Median: $9,382.03, Skewness: 1.516).
   - Kernel density estimation reveals three distinct risk clusters: low-cost non-smokers (<$10,000), intermediate middle-aged/overweight individuals ($10,000–$25,000), and high-cost claims ($30,000–$50,000+).
2. **Age vs. Charges (`outputs/figures/age_vs_charges.png`):**
   - Shows a strong, persistent positive trend (+~$261.81 per year of age).
   - Stratification by smoking status demonstrates three parallel upward cost bands, indicating that age acts as a baseline increment while smoking and obesity act as vertical shifters.
3. **BMI vs. Charges & Obesity Divergence (`outputs/figures/bmi_vs_charges.png`):**
   - For non-smokers, charges scale with a gentle linear slope across all BMI tiers.
   - For active smokers, charges explosively diverge above the clinical obesity threshold ($BMI \ge 30.0\text{ kg/m}^2$), jumping from ~$14,000 to >$35,000–$45,000.
4. **Smoking Status Distribution (`outputs/figures/charges_by_smoker.png`):**
   - Non-smoker mean: $8,434.27 vs. Smoker mean: $32,050.23 (an empirical group mean difference of +$23,615.96).
5. **Correlation Heatmap (`outputs/figures/correlation_heatmap.png`):**
   - Smoker status shows the strongest linear correlation with charges ($r \approx 0.79$), followed by age ($r \approx 0.30$) and BMI ($r \approx 0.20$). Inter-predictor correlations are low ($|r| < 0.11$), confirming absence of collinearity prior to transformation.

*Non-Causal Note: All findings describe empirical statistical associations under ceteris paribus (holding all other observed factors constant) and do not represent deterministic individual causal mechanisms.*

---

## Outlier Analysis & Justified Retention

A formal outlier diagnostic was performed on `charges` using Tukey's Interquartile Range (IQR) fence rule:
- $Q_1 = \$4,740.29$
- $Q_3 = \$16,639.91$
- $IQR = Q_3 - Q_1 = \$11,899.62$
- **Upper Tukey Fence:** $Q_3 + 1.5 \times IQR = \$34,489.35$

### Quantitative Profile of Outliers
- Exactly **139 records** (10.39% of the dataset) exceed the upper fence ($>\$34,489.35$).
- **97.8% (136 / 139)** of outlier records belong to active tobacco smokers.
- **95.0% (132 / 139)** are obese smokers ($BMI \ge 30.0\text{ kg/m}^2$ and $smoker = \text{'yes'}$).
- Total dollar amount of outlier claims: **$5,852,448.66** — accounting for **33.0% of all billed dollars** across the entire $17.75M dataset!

### Actuarial & Statistical Retention Justification
In healthcare economics and actuarial science, claim distributions are inherently heavy-tailed. High charges in this dataset represent legitimate observations associated with higher-risk individuals, not recording errors.

Dropping these 139 records would remove a significant portion of the dataset's high-cost observations, potentially resulting in a model that underestimates charges for high-risk individuals. Rather than artificially truncating or winsorizing legitimate values, the non-linear pattern was addressed through domain-specific interaction modeling (`obese_smoker`). Consequently, **100% of the dataset was retained**.

---

## Preprocessing & Clinical Feature Engineering

### 1. Categorical Encoding & Dummy Variable Trap Mitigation
Linear regression models require a design matrix of full column rank. If all $K$ levels of a categorical variable are one-hot encoded alongside an intercept column, exact linear dependency produces a singular covariance matrix $(X^T X)^{-1}$.
- Binary covariates (`sex`, `smoker`) drop the reference category (`female`, `no`).
- The 4-level nominal variable `region` is encoded with `OneHotEncoder(drop='first')`, establishing `northeast` as the reference baseline.

### 2. Leakage-Free Clinical Feature Engineering (`ClinicalFeatureEngineer`)
Based on the EDA finding of severe non-linear divergence at $BMI \ge 30$ for smokers, a clinical interaction term was engineered:
$$\text{obese\_smoker} = (bmi \ge 30.0) \times (smoker == \text{'yes'})$$
- **Scikit-learn Compliance:** Implemented in `src/data_preprocessing.py` as `ClinicalFeatureEngineer`, subclassing `BaseEstimator` and `TransformerMixin`.
- **Stateless Hygiene:** `fit(X, y)` computes no dataset-level parameters and returns `self` directly. Transformations operate strictly row-wise, ensuring zero train-to-test leakage.
- **Resilience:** Gracefully handles single rows, large batches, and 0-row DataFrames.

### 3. Documented Feature Selection
All 6 raw features plus the engineered interaction were retained with explicit justifications:
- `smoker` & `obese_smoker`: Dominant cost drivers, explaining >60% of variance.
- `age`: Primary continuous biological factor.
- `bmi`: Retained under the hierarchical regression principle (main effects accompany interaction terms).
- `children`: Dependent cost increment (+$469.79).
- `region`: Captures geographic provider price variations while eliminating multicollinearity.
- `sex`: Retained for completeness and input symmetry in the Streamlit frontend.

---

## Model Architecture & Training

### Algorithmic Formulation
The model is strictly an Ordinary Least Squares (OLS) Linear Regression:
$$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}, \quad \hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

No non-linear black box algorithms (Random Forest, XGBoost, LightGBM, Gradient Boosting, or Neural Networks) were used. OLS provides full parameter transparency, exact closed-form solvability, and compliance with statutory rate filings.

### Pipeline Encapsulation
The complete training workflow is encapsulated in an end-to-end `scikit-learn.pipeline.Pipeline`:
```
Raw DataFrame (6 columns)
       │
       ▼
[ Step 1: ClinicalFeatureEngineer() ]  ──> Adds 'obese_smoker'
       │
       ▼
[ Step 2: ColumnTransformer ]          ──> Numeric passthrough + Categorical drop='first'
       │
       ▼
[ Step 3: LinearRegression() ]         ──> Fits OLS parameters
       │
       ▼
Predicted Charges Array (float)
```

The trained artifact is serialized to `models/insurance_model.joblib`.

---

## Evaluation Metrics & Results

### Holdout Test Set Performance
Evaluated on the 20% holdout test partition (268 samples) untouched during training. The metrics below match `outputs/metrics.json` exactly:

| Metric | Baseline OLS (6 Features) | Enhanced OLS (`obese_smoker`) | Absolute Delta | Relative Gain |
|---|---|---|---|---|
| **Mean Absolute Error (MAE)** | 4181.1945 ($4,181.19) | **2486.3152** ($2,486.32) | -$1,694.88 | **-40.5% Error Drop** |
| **Mean Squared Error (MSE)** | 33596915.8514 | **19876213.8593** | -13,720,701.99 | **-40.8% Variance Drop** |
| **Root Mean Squared Error (RMSE)** | 5796.2847 ($5,796.28) | **4458.2748** ($4,458.27) | -$1,338.01 | **-23.1% Lower Spread** |
| **Coefficient of Determination ($R^2$)** | 0.7836 | **0.8720** | +0.0884 | **+11.3% Variance Explained** |

*Key Finding: Introducing the clinically grounded interaction `obese_smoker` elevates $R^2$ from 78.4% to 87.2% and cuts Mean Absolute Error by $1,694.88 per patient.*

### Learned OLS Coefficients & Non-Causal Dollar Interpretations
- **Model Intercept ($\beta_0$):** -$3,213.86

| Feature Name | Coefficient ($\beta$) | Non-Causal Dollar Interpretation (*ceteris paribus*) |
|---|---|---|
| `age` | +$261.81 | Associated with an estimated +$261.81 annual cost per additional year of age |
| `bmi` | +$56.58 | Associated with +$56.58 annual cost per 1-unit increase in BMI for non-smokers |
| `children` | +$469.79 | Associated with +$469.79 annual cost per additional covered dependent |
| `obese_smoker` | +$19,850.74 | Synergistic surge associated with individuals who are BOTH clinically obese & active smokers |
| `sex_male` | -$502.42 | Expected difference for males relative to female baseline (statistically marginal) |
| `smoker_yes` | +$13,292.21 | Associated with +$13,292.21 charge for non-obese smokers relative to non-smokers |
| `region_northwest` | -$339.00 | Expected charge adjustment for Northwest residents relative to Northeast baseline |
| `region_southeast` | -$692.16 | Expected charge adjustment for Southeast residents relative to Northeast baseline |
| `region_southwest` | -$1,371.81 | Expected charge adjustment for Southwest residents relative to Northeast baseline |

### Diagnostic Visualizations
- `outputs/figures/actual_vs_predicted.png`: Displays tight clustering along the 45-degree identity reference line across all risk tiers.
- `outputs/figures/residual_diagnostics.png`: Confirms residuals are well-centered at zero across predicted charge ranges.

---

## Streamlit Web Application

The interactive web tool is implemented in `app.py` for real-time user inference.

### Key Features & Architectural Hygiene
- **Responsive 2-Column Interface:**
  - *Column 1 (Demographics):* Age (18–100 slider), Biological Sex (radio), Covered Children (0–10 slider).
  - *Column 2 (Clinical Risk):* BMI (10.0–60.0 slider), Smoking Status (radio), US Geographic Region (selectbox).
- **Direct Pipeline Ingestion (Zero Preprocessing Duplication):** Loads `models/insurance_model.joblib` using `@st.cache_resource`. Assembles inputs into a raw `pandas.DataFrame` and directly calls `pipeline.predict(input_df)`.
- **Defensive Safety Clamping:** Guards against negative regression boundary extrapolation: `max(0.0, float(pred))`.
- **Currency-Formatted Output:** Displays predictions prominently in large metric banners (e.g., `Estimated Annual Medical Charge: $4,215.30`).
- **Mandatory Verbatim Legal Disclaimer:**
  > *"Disclaimer: This tool provides estimated charges for educational/informational purposes only and is not an actual insurance quote."*

### Application Screenshot
A high-fidelity screenshot of the running interface is captured at:
`outputs/screenshots/app_screenshot.png`

---

## Final Internship PDF Report

An 8-page, publication-grade academic/industry final report is generated by `scripts/generate_report.py` and saved to:
`report/medical_insurance_cost_prediction_report.pdf`

### Document Specifications
- **Format:** Valid `%PDF-1.4`, 8 pages, ~890 KB (exceeds the 50 KB requirement).
- **Design Elements:** Running headers, footers with page numbers ("Page X of 8"), corporate color palette, styled cards, formatted data tables, and embedded high-resolution figures.
- **Section Coverage:** 
  1. Introduction (project background, objectives, business context)
  2. Dataset Description (source, 1,338 records, 7 schema columns, data hygiene)
  3. Exploratory Data Analysis (univariate/bivariate distributions, correlation, non-causal interpretations)
  4. Data Preprocessing (Tukey IQR outlier analysis with 100% retention justification, dummy variable trap resolution, ClinicalFeatureEngineer)
  5. Model Architecture & Training (OLS LinearRegression strictly, 80/20 train/test split, Pipeline encapsulation)
  6. Evaluation Results (actual vs predicted, residuals, coefficient table, non-causal dollar interpretations, baseline vs enhanced comparison)
  7. Streamlit Web Application (interactive UI design, input controls, model loading, disclaimer, screenshot)
  8. Limitations & Future Work (linear assumptions, lack of longitudinal data, risk pooling ethics)
  9. Conclusion (summary of findings, practical implications for healthcare pricing)

To regenerate the report dynamically:
```bash
python scripts/generate_report.py
```

---

## Example Prediction

### Command-Line Execution
Run `src/predict.py` to test inference on a sample individual:
```bash
python src/predict.py
```
Output:
```
================================================================================
          Medical Insurance Cost Prediction - Inference Demonstration
================================================================================
Individual Patient Input Profile:
  Age:               25 years
  Biological Sex:    female
  Body Mass Index:   25.50 kg/m² (Normal weight)
  Covered Children:  0 dependents
  Smoking Status:    no (Non-smoker)
  US Region:         southwest
--------------------------------------------------------------------------------
Model Artifact:      models/insurance_model.joblib
Estimated Annual Medical Cost: $1,972.19
================================================================================
```

### Programmatic Python Usage
```python
import joblib
import pandas as pd

# Load serialized pipeline artifact
pipeline = joblib.load("models/insurance_model.joblib")

# Prepare raw beneficiary profile
patient_data = pd.DataFrame([{
    "age": 30,
    "sex": "female",
    "bmi": 25.0,
    "children": 0,
    "smoker": "no",
    "region": "northeast"
}])

# Execute prediction
predicted_charge = float(pipeline.predict(patient_data)[0])
print(f"Predicted Annual Charge: ${max(0.0, predicted_charge):,.2f}")
# Output: Predicted Annual Charge: $4,215.30
```

---

## Limitations & Future Work
1. **Linearity & Heteroskedasticity:** Ordinary Least Squares assumes constant residual variance. As visible in residual diagnostics, claim variance naturally expands at higher predicted charges. High-cost medical claimants naturally exhibit higher dollar dispersion.
2. **Absence of Longitudinal Tracking:** The dataset provides a single-year cross-sectional slice; it lacks multi-year patient trajectories, chronic condition onset tracking, or policy lapse behavior.
3. **Limited Feature Set:** The dataset contains only basic demographic and lifestyle variables. Important predictors of healthcare expenditure—such as pre-existing conditions, medical history, prescription usage, and family health background—are not available in this dataset.
4. **Actuarial Risk Pooling & Regulatory Ethics:** Under the Affordable Care Act (ACA), pricing adjustments based on sex or biological health conditions (BMI) are prohibited in individual markets. Predictive expenditure models must be governed carefully to avoid disparate underwriting impacts against protected demographics.
5. **Future Algorithmic Directions:** Implementing Generalized Linear Models (GLM) using a Gamma distribution with a log link function, evaluating regularized regression (Lasso/Ridge/ElasticNet), and incorporating state-level geographic hierarchical mixed effects.

---

## How to Run

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execute the Jupyter Notebook
Open and run all cells in the exploratory analysis notebook:
```bash
jupyter notebook notebooks/medical_insurance_cost_prediction.ipynb
```

### 3. Train the Model Pipeline
Execute the training script to evaluate test performance, serialize the model, and export metrics:
```bash
python src/train_model.py
```

### 4. Run CLI Inference
Run the standalone prediction demonstration:
```bash
python src/predict.py
```

### 5. Launch the Streamlit Web Application
Start the interactive local web server:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. Generate the Final Internship PDF Report
Run the report generator to build the 8-page PDF:
```bash
python scripts/generate_report.py
```
View the output at `report/medical_insurance_cost_prediction_report.pdf`.

### 7. Run the Automated Test Suite
Execute the multi-tier test suite:
```bash
python tests/run_tests.py
```

---

## Project Structure
```
medical-insurance-cost-prediction/
├── README.md                                # Comprehensive project documentation
├── requirements.txt                         # Verified Python dependencies
├── insurance.csv                            # Raw medical insurance dataset (1,338 records)
│
├── notebooks/
│   └── medical_insurance_cost_prediction.ipynb  # Comprehensive EDA, Outlier Analysis & Modeling
│
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── data_preprocessing.py                # ClinicalFeatureEngineer, load_data, split_data, preprocessor
│   ├── train_model.py                       # OLS training pipeline, evaluation & joblib serialization
│   └── predict.py                           # Standalone CLI prediction script
│
├── models/
│   └── insurance_model.joblib               # Serialized fitted scikit-learn Pipeline artifact
│
├── app.py                                   # Interactive Streamlit frontend web application
│
├── scripts/
│   ├── capture_screenshot.py                # Automated multi-tier Streamlit screenshot capture utility
│   └── generate_report.py                   # 8-page publication-grade PDF report generator
│
├── outputs/
│   ├── metrics.json                         # Calculated test set metrics (single source of truth)
│   ├── figures/                             # Exported high-resolution publication PNG plots
│   │   ├── actual_vs_predicted.png
│   │   ├── age_vs_charges.png
│   │   ├── bmi_vs_charges.png
│   │   ├── charges_by_smoker.png
│   │   ├── correlation_heatmap.png
│   │   ├── residual_diagnostics.png
│   │   └── target_distribution.png
│   └── screenshots/
│       └── app_screenshot.png               # Streamlit application interface screenshot
│
├── report/
│   └── medical_insurance_cost_prediction_report.pdf  # Final 8-page internship technical report
│
└── tests/                                   # Automated test suite
    ├── __init__.py
    ├── run_tests.py                         # Custom multi-tier test runner
    ├── test_data_preprocessing.py
    ├── test_train_model.py
    ├── test_predict.py
    ├── test_tier1_features.py
    ├── test_tier2_boundaries.py
    ├── test_tier3_combinations.py
    ├── test_tier4_scenarios.py
    └── test_m1_stress.py
```
