"""
Tests for the ML engine module.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml_engine import MLEngine, prepare_training_data
from simulation import generate_training_data


class TestMLEngine:
    """Test ML model training and prediction."""

    def test_train_and_predict(self):
        data = generate_training_data(num_samples=200)
        X, y = prepare_training_data(data)
        engine = MLEngine()
        engine.train(X, y, use_grid_search=False)
        assert engine.is_trained

        predictions = engine.predict([5, 2, 1])
        assert "xgb" in predictions
        assert "rf" in predictions
        assert "svm" in predictions
        assert "dt" in predictions
        assert "ada" in predictions

    def test_zero_prediction(self):
        data = generate_training_data(num_samples=200)
        X, y = prepare_training_data(data)
        engine = MLEngine()
        engine.train(X, y, use_grid_search=False)

        result = engine.predict([0, 0, 0])
        assert result["xgb"]["prediction"] == 0, "Zero values should be SAFE"

    def test_attack_prediction(self):
        data = generate_training_data(num_samples=200)
        X, y = prepare_training_data(data)
        engine = MLEngine()
        engine.train(X, y, use_grid_search=False)

        result = engine.predict([120, 80, 60])
        assert result["xgb"]["prediction"] == 1, "High values should be ATTACK"

    def test_model_comparison(self):
        data = generate_training_data(num_samples=200)
        X, y = prepare_training_data(data)
        engine = MLEngine()
        engine.train(X, y, use_grid_search=False)

        comparison = engine.get_model_comparison()
        assert len(comparison) == 5
        for model in comparison:
            assert "name" in model
            assert "accuracy" in model
            assert "sensitivity" in model
            assert "f1_score" in model
