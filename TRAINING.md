# Model Training Guide

This guide explains how to train and evaluate the ML models in this system.

## Prerequisites

```bash
pip install -r requirements.txt
```

Dependencies: scikit-learn, xgboost, numpy, pandas, joblib

## Training Data Sources

The model engine can be trained from two data sources:

1. **Logged server-side operations** (from the SQLite database)
   - Automatically used when `database.get_training_data()` returns >= 200 samples
   - Features: nc, nw, nr, nm, nu + label att (0=benign, 1=attack)

2. **Synthetic data** (from `simulation.generate_training_data()`)
   - Used when insufficient logged data exists
   - Generates balanced benign/attack samples
   - Follows paper simulation parameters

## Training Process

Models are trained via `MLEngine.train()`:

```python
engine = MLEngine()
X, y = prepare_training_data(data)
results = engine.train(X, y, use_grid_search=True)
engine.save_models()
```

### Hyperparameter Optimization

- **GridSearchCV** with 5-fold cross-validation
- Scoring metric: **recall** (prioritizes attack detection sensitivity)
- SVM uses CalibratedClassifierCV (fixed params for speed)

## Model Files

After training, models are saved to `models/` directory:

| File | Description |
|------|-------------|
| `ransomware_rf.joblib` | Random Forest classifier |
| `ransomware_xgb.joblib` | XGBoost classifier |
| `ransomware_svm.joblib` | SVM with probability calibration |
| `ransomware_dt.joblib` | Decision Tree classifier |
| `ransomware_ada.joblib` | AdaBoost classifier |
| `ransomware_scaler.joblib` | Feature StandardScaler |
| `ransomware_metrics.joblib` | Training performance metrics |

## Standalone Training

Run the ML engine directly to train models:

```bash
python ml_engine.py
```

This:
1. Generates 8000 synthetic training samples
2. Trains all 5 models with grid search
3. Prints model comparison metrics
4. Runs inference tests

## Evaluation Metrics

For each model, the following metrics are computed:
- **Accuracy**: Overall correct predictions
- **Sensitivity** (Recall): True positive rate for attacks
- **Specificity**: True negative rate
- **Precision**: Positive predictive value
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP/FP/TN/FN breakdown
- **Prediction Latency**: Average inference time (ms)
