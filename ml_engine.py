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
import time
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional
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
    classification_report,
)
import xgboost as xgb


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class MLEngine:
    """
    Machine Learning engine for ransomware detection.

    Trains, saves, loads, and runs inference with 5 ML models.
    """

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.model_names = ["Random Forest", "SVM", "Decision Tree", "AdaBoost", "XGBoost"]
        self.model_keys = ["rf", "svm", "dt", "ada", "xgb"]
        self.is_trained = False
        self.training_metrics = {}
        self.feature_names = ["nc", "nw", "nr", "nm", "nu"]

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
            X: Feature matrix (nc, nw, nr, nm, nu)
            y: Labels (0=benign, 1=attack)
            use_grid_search: Whether to use GridSearchCV for hyperparameter tuning

        Returns:
            Dictionary of training metrics per model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        configs = self._get_model_configs()
        results = {}

        for key in self.model_keys:
            config = configs[key]
            model = config["model"]
            params = config["params"]

            print(f"Training {self.model_keys[self.model_keys.index(key)]}...")

            start_time = time.time()

            if use_grid_search and key != "svm":
                # GridSearchCV for all models except SVM (too slow)
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
                # For SVM, use CalibratedClassifierCV for probability support
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

            # Evaluate
            y_pred = best_model.predict(X_test_scaled)
            y_proba = (
                best_model.predict_proba(X_test_scaled)[:, 1]
                if hasattr(best_model, "predict_proba")
                else None
            )

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            sensitivity = recall_score(y_test, y_pred)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # Prediction latency
            pred_start = time.time()
            for _ in range(100):
                best_model.predict(X_test_scaled[:1])
            avg_latency = (time.time() - pred_start) / 100 * 1000  # ms

            self.models[key] = best_model

            results[key] = {
                "name": self.model_keys[self.model_keys.index(key)],
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

            print(f"  -> Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, F1: {f1:.4f}")

        self.is_trained = True
        self.training_metrics = results
        return results

    def predict(self, features: List[float]) -> Dict:
        """
        Run inference on a single sample using all trained models.

        Args:
            features: [nc, nw, nr, nm, nu] file operation counts

        Returns:
            Dictionary of predictions per model
        """
        if not self.is_trained:
            return {"error": "Models not trained"}

        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        predictions = {}
        for key in self.model_keys:
            if key in self.models:
                model = self.models[key]
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else [0, 0]

                predictions[key] = {
                    "name": self.model_keys[self.model_keys.index(key)],
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
                    "model": self.model_keys[self.model_keys.index(key)],
                    "predictions": preds.tolist(),
                    "probabilities": probas.tolist() if probas is not None else None,
                })

        return results

    def save_models(self, prefix: str = "ransomware"):
        """Save all trained models and scaler to disk."""
        if not self.is_trained:
            return False

        for key in self.model_keys:
            if key in self.models:
                path = os.path.join(MODELS_DIR, f"{prefix}_{key}.joblib")
                joblib.dump(self.models[key], path)

        scaler_path = os.path.join(MODELS_DIR, f"{prefix}_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

        metrics_path = os.path.join(MODELS_DIR, f"{prefix}_metrics.joblib")
        joblib.dump(self.training_metrics, metrics_path)

        print(f"Models saved to {MODELS_DIR}")
        return True

    def load_models(self, prefix: str = "ransomware") -> bool:
        """Load trained models from disk."""
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
                print(f"Loaded {len(self.models)} models from {MODELS_DIR}")
            return self.is_trained

        except Exception as e:
            print(f"Error loading models: {e}")
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
    X = df[["nc", "nw", "nr", "nm", "nu"]].values
    y = df["att"].values
    return X, y


def train_and_save_models(data: List[Dict]) -> MLEngine:
    """Complete training pipeline: prepare data, train, save."""
    engine = MLEngine()

    X, y = prepare_training_data(data)
    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: Benign={sum(y==0)}, Attack={sum(y==1)}")

    results = engine.train(X, y, use_grid_search=True)
    engine.save_models()

    return engine


if __name__ == "__main__":
    from simulation import generate_training_data

    print("Generating training data...")
    data = generate_training_data(num_samples=8000)
    print(f"Generated {len(data)} samples")

    print("\nTraining models...")
    engine = train_and_save_models(data)

    print("\n=== Model Comparison ===")
    for m in engine.get_model_comparison():
        print(f"\n{m['name']}:")
        print(f"  Accuracy:    {m['accuracy']:.4f}")
        print(f"  Sensitivity: {m['sensitivity']:.4f}")
        print(f"  Specificity: {m['specificity']:.4f}")
        print(f"  F1-Score:    {m['f1_score']:.4f}")
        print(f"  Latency:     {m['prediction_latency_ms']:.3f} ms")
        print(f"  CM: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")

    # Test inference
    print("\n=== Inference Test ===")
    test_samples = [
        [5, 15, 40, 2, 1],       # Low activity (benign)
        [45, 60, 120, 15, 8],    # Normal user (benign)
        [100, 160, 240, 60, 40], # High activity (attack)
        [95, 140, 220, 70, 45],  # Attack pattern
    ]

    for sample in test_samples:
        pred = engine.predict(sample)
        print(f"\nInput: nc={sample[0]}, nw={sample[1]}, nr={sample[2]}, nm={sample[3]}, nu={sample[4]}")
        for key, value in pred.items():
            if key != "error":
                label = "ATTACK" if value["prediction"] == 1 else "BENIGN"
                print(f"  {value['name']:15s}: {label} (conf: {value['confidence']:.1f}%)")
