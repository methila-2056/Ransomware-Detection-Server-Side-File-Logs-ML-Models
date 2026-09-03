# Pretrained Models

This directory contains the trained ML models used by the Ransomware Detection System.

## Models

| File | Algorithm | Description |
|------|-----------|-------------|
| ransomware_rf.joblib | Random Forest | Ensemble of decision trees |
| ransomware_xgb.joblib | XGBoost | Gradient boosting (primary detector) |
| ransomware_svm.joblib | SVM | Support vector machine with calibration |
| ransomware_dt.joblib | Decision Tree | Single decision tree |
| ransomware_ada.joblib | AdaBoost | Adaptive boosting |

## Supporting Files

- ransomware_scaler.joblib: StandardScaler for feature normalization
- ransomware_metrics.joblib: Training evaluation metrics

## Feature Schema

Each model takes 5 features: [nc, nw, nr, nm, nu]
- nc: file creations
- nw: file writes
- nr: file reads
- nm: file renames
- nu: file deletions

## Model Backup

Models based on the legacy 3-feature schema are stored in `../models_backup_3feature/`.
