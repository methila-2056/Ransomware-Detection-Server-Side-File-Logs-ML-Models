"""
Model Evaluation Script
Trains models and prints detailed performance metrics.
Run: python scripts/evaluate.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation import generate_training_data
from ml_engine import MLEngine, prepare_training_data
import config


def main():
    print("=" * 60)
    print("  RANSOMWARE DETECTION - MODEL EVALUATION")
    print("=" * 60)
    print()

    print(f"Generating training data ({config.TRAINING_SAMPLES} samples)...")
    data = generate_training_data(num_samples=config.TRAINING_SAMPLES)
    X, y = prepare_training_data(data)

    print(f"Features: {X.shape[1]} (nc, nr, nu)")
    print(f"Samples:  {X.shape[0]}")
    print(f"Benign:   {sum(y == 0)}")
    print(f"Attack:   {sum(y == 1)}")
    print()

    print("Training models (GridSearchCV disabled for speed)...")
    engine = MLEngine()
    engine.train(X, y, use_grid_search=False)
    engine.save_models()
    print()

    print("=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)

    comparison = engine.get_model_comparison()
    print()
    print(f"{'Model':<12} {'Accuracy':>10} {'Sensitivity':>12} {'Specificity':>12} {'F1':>8} {'Latency':>10}")
    print("-" * 70)

    for m in comparison:
        print(
            f"{m['name']:<12} "
            f"{m['accuracy'] * 100:>9.2f}% "
            f"{m['sensitivity'] * 100:>11.2f}% "
            f"{m['specificity'] * 100:>11.2f}% "
            f"{m['f1_score'] * 100:>7.2f}% "
            f"{m['prediction_latency_ms']:>8.3f}ms"
        )

    print()
    print("=" * 60)
    print("  CONFUSION MATRIX (XGBoost)")
    print("=" * 60)

    xgb = next(m for m in comparison if m["name"] == "XGBoost")
    print(f"  TP: {xgb['tp']:>6}  FP: {xgb['fp']:>6}")
    print(f"  FN: {xgb['fn']:>6}  TN: {xgb['tn']:>6}")
    print()

    print("=" * 60)
    print("  INFERENCE TEST")
    print("=" * 60)

    test_cases = [
        ([0, 0, 0], "Idle system"),
        ([3, 1, 1], "Minimum activity"),
        ([15, 5, 2], "Light activity"),
        ([45, 15, 8], "Normal user"),
        ([80, 40, 30], "High activity"),
        ([120, 80, 60], "Extreme activity"),
    ]

    print()
    for features, label in test_cases:
        result = engine.predict(features)
        xgb_pred = result["xgb"]
        status = "ATTACK" if xgb_pred["prediction"] == 1 else "SAFE"
        prob = xgb_pred["probability"] * 100
        print(f"  {label:<20} nc={features[0]:>3} nr={features[1]:>3} nu={features[2]:>3}  ->  {status:<8} ({prob:.1f}%)")

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
