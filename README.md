# Liver Cirrhosis Stage Prediction

Multi-class machine learning project that predicts the **stage of liver cirrhosis** (Stage 1, 2, or 3) from clinical and lab measurements, using a Mayo Clinic–style PBC (Primary Biliary Cirrhosis) dataset.

## Overview

The notebook covers a full ML pipeline:

- **Data cleaning** — missing values, duplicates, categorical encoding (`Drug`, `Sex`, `Ascites`, `Hepatomegaly`, `Spiders`, `Edema`)
- **Feature engineering** — age in years, clinically-inspired ratios (`APRI`, `Bilirubin/Albumin`), log-transform of skewed lab values
- **Exploratory Data Analysis** — target class distribution, correlation heatmap, boxplots for outlier inspection
- **Modeling** — 8 classifiers trained and compared on an identical train/test split:
  - Logistic Regression
  - Decision Tree
  - Random Forest (GridSearchCV-tuned)
  - LightGBM (GridSearchCV-tuned)
  - SVM (RBF kernel)
  - K-Nearest Neighbors
  - XGBoost (GridSearchCV-tuned)
  - Stacking Ensemble (RF + XGBoost + LightGBM → Logistic Regression meta-model)
- **Evaluation** — Accuracy, Precision, Recall, and F1 Score (weighted) for every model, plus a confusion matrix
- **Model persistence** — best model, scaler, and label encoder saved with `joblib` for reuse without retraining

## Results

| Rank | Model | Accuracy | F1 Score |
|---|---|---|---|
| 1 | **Stacking** | 0.967 | 0.967 |
| 2 | LightGBM (tuned) | 0.963 | 0.963 |
| 3 | Random Forest (tuned) | 0.947 | 0.947 |
| 4 | XGBoost (tuned) | 0.942 | 0.942 |
| 5 | KNN | 0.875 | 0.875 |
| 6 | SVM | 0.826 | 0.826 |
| 7 | Decision Tree | 0.650 | 0.651 |
| 8 | Logistic Regression | 0.613 | 0.613 |

Tree-based and ensemble models clearly outperform the linear and distance-based models, suggesting the relationship between lab values and disease stage is non-linear and driven by feature interactions rather than any single strong predictor.

## Dataset

`liver_cirrhosis.csv` is included in this repo (25,000 rows, 19 columns): `Patient_ID`, `N_Days_of suffering`, `Drug`, `Age_in days`, `Sex`, `Ascites`, `Hepatomegaly`, `Spiders`, `Edema`, `Bilirubin`, `Cholesterol`, `Albumin`, `Copper`, `Alk_Phos`, `SGOT`, `Tryglicerides`, `Platelets`, `Prothrombin`, and the target `Stage_of_disease`.

> **Note:** the near-perfectly balanced class distribution (~8,300 rows per stage out of ~25,000 total) suggests this dataset may be synthetic or augmented rather than raw clinical data. Results should be interpreted with that in mind if used beyond a learning/portfolio context.

## Getting Started

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run `liver_cirrhosis_project.ipynb` top to bottom (Run All) — `liver_cirrhosis.csv` is already in the project root.

## Project Structure

```
.
├── liver_cirrhosis_project.ipynb   # main notebook (EDA + modeling + evaluation)
├── liver_cirrhosis.csv             # dataset
├── requirements.txt                # Python dependencies
├── README.md
└── .gitignore
```

After running the notebook, it will also produce:
- `best_stacking_model.pkl` — trained Stacking ensemble
- `scaler.pkl` — fitted `StandardScaler` (needed for LR/SVM/KNN inference)
- `label_encoder.pkl` — fitted `LabelEncoder` for the target classes

## Possible Next Steps

- SHAP-based interpretability for the best model
- Stratified k-fold cross-validation instead of a single train/test split
- A small inference script / API that loads the saved `.pkl` files and predicts on new patient data
