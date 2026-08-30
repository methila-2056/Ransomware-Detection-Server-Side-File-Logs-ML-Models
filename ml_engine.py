"""
Ransomware Detection - Machine Learning Engine

Trains and evaluates 5 ML models for ransomware detection:
- Random Forest (RF)
- Support Vector Machine (SVM)
- Decision Tree (DT)
- AdaBoost (ADA)
- XGBoost (XGB)

Based on Aranyi et al. (2026) methodology.
"""

import os
import logging
import time
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import xgboost as xgb

import config

log = logging.getLogger("ransomware.ml")

MODELS_DIR = config.MODEL_DIR
os.makedirs(MODELS_DIR, exist_ok=True)


class MLEngine:
    """
    Machine Learning engine for ransomware detection.

    Trains, saves, loads, and runs inference with 5 ML models.
    """

    def __init__(self):
        self.models: Dict[str, object] = {}
        self.scaler = StandardScaler()
        self.model_names = ["Random Forest", "SVM", "Decision Tree", "AdaBoost", "XGBoost"]
        self.model_keys = ["rf", "svm", "dt", "ada", "xgb"]
        self.is_trained = False
        self.training_metrics: Dict = {}
        self.feature_names = ["nc", "nr", "nu"]

    def _get_model_configs(self) -> Dict:
        """Get model constructors and hyperparameter grids."""
        return {
            "rf": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                },
            },
            "svm": {
                "model": SVC(random_state=42),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"],
                },
            },
            "dt": {
                "model": DecisionTreeClassifier(random_state=42),
                "params": {
                    "max_depth": [5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"],
                },
            },
            "ada": {
                "model": AdaBoostClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 150],
                    "learning_rate": [0.5, 1.0, 1.5],
                },
            },
            "xgb": {
                "model": xgb.XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                ),
                "params": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.1, 0.3, 0.5],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                },
            },
        }

    def train(self, X: np.ndarray, y: np.ndarray, use_grid_search: bool = True) -> Dict:
        """
        Train all 5 ML models.

        Args:
            X: Feature matrix (nc, nr, nu)
            y: Labels (0=benign, 1=attack)
            use_grid_search: Whether to use GridSearchCV for hyperparameter tuning

        Returns:
            Dictionary of training metrics per model
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        configs = self._get_model_configs()
        results = {}

        for key in self.model_keys:
            model_cfg = configs[key]
            model = model_cfg["model"]
            params = model_cfg["params"]

            name = self.model_names[self.model_keys.index(key)]
            log.info("Training %s...", name)

            start_time = time.time()

            if use_grid_search and key != "svm":
                grid = GridSearchCV(
                    model,
                    params,
                    cv=5,
                    scoring="recall",
                    n_jobs=-1,
                    verbose=0,
                )
                grid.fit(X_train_scaled, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                if key == "svm":
                    from sklearn.calibration import CalibratedClassifierCV
                    model.set_params(C=10, kernel="rbf", gamma="scale")
                    calibrated = CalibratedClassifierCV(model, ensemble=False)
                    calibrated.fit(X_train_scaled, y_train)
                    best_model = calibrated
                    best_params = calibrated.get_params()
                else:
                    model.fit(X_train_scaled, y_train)
                    best_model = model
                    best_params = model.get_params()

            train_time = time.time() - start_time

            y_pred = best_model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            sensitivity = recall_score(y_test, y_pred)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            pred_start = time.time()
            for _ in range(100):
                best_model.predict(X_test_scaled[:1])
            avg_latency = (time.time() - pred_start) / 100 * 1000

            self.models[key] = best_model

            results[key] = {
                "name": name,
                "key": key,
                "accuracy": round(accuracy, 4),
                "f1_score": round(f1, 4),
                "precision": round(precision, 4),
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "confusion_matrix": {
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                },
                "train_time": round(train_time, 2),
                "avg_prediction_latency_ms": round(avg_latency, 3),
                "best_params": str(best_params)[:200],
            }

            log.info("  -> Accuracy: %.4f, Sensitivity: %.4f, F1: %.4f",
                     accuracy, sensitivity, f1)

        self.is_trained = True
        self.training_metrics = results
        return results

    @staticmethod
    def validate_features(features: List[float]) -> List[float]:
        """
        Validate and coerce a feature vector (nc, nr, nu).

        Ensures the input is a numeric collection of exactly
        ``len(self.feature_names)`` non-negative values, clamping
        negative counts to zero.

        Raises:
            ValueError: if the shape or types are not valid.
        """
        if features is None or not isinstance(features, (list, tuple)):
            raise ValueError("features must be a list or tuple")
        if len(features) != 3:
            raise ValueError(f"expected 3 features, got {len(features)}")
        cleaned = []
        for value in features:
            try:
                cleaned.append(max(0.0, float(value)))
            except (TypeError, ValueError):
                raise ValueError(f"non-numeric feature value: {value!r}")
        return cleaned

    def predict(self, features: List[float]) -> Dict:
        """
        Run inference on a single sample using all trained models.

        Args:
            features: [nc, nr, nu] file operation counts

        Returns:
            Dictionary of predictions per model
        """
        if not self.is_trained:
            return {"error": "Models not trained"}

        features = self.validate_features(features)
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        predictions = {}
        for key in self.model_keys:
            if key in self.models:
                model = self.models[key]
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else [0, 0]

                predictions[key] = {
                    "name": self.model_names[self.model_keys.index(key)],
                    "key": key,
                    "prediction": int(pred),
                    "probability": round(float(proba[1]), 4),
                    "confidence": round(float(max(proba)) * 100, 1),
                }

        return predictions

    def predict_batch(self, X: np.ndarray) -> List[Dict]:
        """Run inference on multiple samples."""
        if not self.is_trained:
            return [{"error": "Models not trained"}]

        X_scaled = self.scaler.transform(X)
        results = []

        for key in self.model_keys:
            if key in self.models:
                model = self.models[key]
                preds = model.predict(X_scaled)
                probas = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else None

                results.append({
                    "model": self.model_names[self.model_keys.index(key)],
                    "key": key,
                    "predictions": preds.tolist(),
                    "probabilities": probas.tolist() if probas is not None else None,
                })

        return results

    def predict_from_ops(self, nc: int, nr: int, nu: int) -> Dict:
        """
        Convenience wrapper around :meth:`predict` using named operation
        counts (nc, nr, nu) instead of a positional list.

        Example:
            >>> engine.predict_from_ops(nc=5, nr=2, nu=1)
        """
        return self.predict([nc, nr, nu])

    def save_models(self, prefix: str = None) -> bool:
        """Save all trained models and scaler to disk."""
        if not self.is_trained:
            return False

        prefix = prefix or config.MODEL_PREFIX

        for key in self.model_keys:
            if key in self.models:
                path = os.path.join(MODELS_DIR, f"{prefix}_{key}.joblib")
                joblib.dump(self.models[key], path)

        scaler_path = os.path.join(MODELS_DIR, f"{prefix}_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

        metrics_path = os.path.join(MODELS_DIR, f"{prefix}_metrics.joblib")
        joblib.dump(self.training_metrics, metrics_path)

        log.info("Models saved to %s", MODELS_DIR)
        return True

    def load_models(self, prefix: str = None) -> bool:
        """Load trained models from disk."""
        prefix = prefix or config.MODEL_PREFIX
        try:
            for key in self.model_keys:
                path = os.path.join(MODELS_DIR, f"{prefix}_{key}.joblib")
                if os.path.exists(path):
                    self.models[key] = joblib.load(path)

            scaler_path = os.path.join(MODELS_DIR, f"{prefix}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            metrics_path = os.path.join(MODELS_DIR, f"{prefix}_metrics.joblib")
            if os.path.exists(metrics_path):
                self.training_metrics = joblib.load(metrics_path)

            self.is_trained = len(self.models) == len(self.model_keys)
            if self.is_trained:
                log.info("Loaded %d models from %s", len(self.models), MODELS_DIR)
            return self.is_trained

        except Exception as e:
            log.error("Error loading models: %s", e)
            return False

    def get_model_comparison(self) -> List[Dict]:
        """Get formatted model comparison data for the dashboard."""
        if not self.training_metrics:
            return []

        comparison = []
        for key in self.model_keys:
            if key in self.training_metrics:
                m = self.training_metrics[key]
                comparison.append({
                    "name": m["name"],
                    "key": key,
                    "accuracy": m["accuracy"],
                    "f1_score": m["f1_score"],
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "precision": m["precision"],
                    "prediction_latency_ms": m["avg_prediction_latency_ms"],
                    "tp": m["confusion_matrix"]["tp"],
                    "fp": m["confusion_matrix"]["fp"],
                    "tn": m["confusion_matrix"]["tn"],
                    "fn": m["confusion_matrix"]["fn"],
                })

        return comparison


def prepare_training_data(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert list of dicts to numpy arrays for training."""
    df = pd.DataFrame(data)
    X = df[["nc", "nr", "nu"]].values
    y = df["att"].values
    return X, y


def train_and_save_models(data: List[Dict], use_grid_search: bool = None) -> MLEngine:
    """Complete training pipeline: prepare data, train, save."""
    if use_grid_search is None:
        use_grid_search = config.GRID_SEARCH

    engine = MLEngine()

    X, y = prepare_training_data(data)
    log.info("Training data: %d samples, %d features", X.shape[0], X.shape[1])
    log.info("Class distribution: Benign=%d, Attack=%d", sum(y == 0), sum(y == 1))

    results = engine.train(X, y, use_grid_search=use_grid_search)
    engine.save_models()

    return engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from simulation import generate_training_data

    log.info("Generating training data...")
    data = generate_training_data(num_samples=config.TRAINING_SAMPLES)
    log.info("Generated %d samples", len(data))

    log.info("Training models...")
    engine = train_and_save_models(data)

    log.info("=== Model Comparison ===")
    for m in engine.get_model_comparison():
        log.info(
            "%s: acc=%.4f sens=%.4f spec=%.4f f1=%.4f latency=%.3fms "
            "CM(TP=%d FP=%d TN=%d FN=%d)",
            m["name"], m["accuracy"], m["sensitivity"], m["specificity"],
            m["f1_score"], m["prediction_latency_ms"],
            m["tp"], m["fp"], m["tn"], m["fn"],
        )

    log.info("=== Inference Test ===")
    test_samples = [
        [5, 2, 1],
        [45, 15, 8],
        [100, 60, 40],
        [95, 70, 45],
    ]

    for sample in test_samples:
        pred = engine.predict(sample)
        log.info("Input: nc=%d, nr=%d, nu=%d", sample[0], sample[1], sample[2])
        for key, value in pred.items():
            if key != "error":
                label = "ATTACK" if value["prediction"] == 1 else "BENIGN"
                log.info("  %-15s: %s (conf: %.1f%%)", value["name"], label,
                         value["confidence"])
