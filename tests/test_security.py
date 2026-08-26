"""
Tests for security features: API key authentication,
input validation, and rate limiting configuration.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import app as A
import config


class TestAPIKeyAuth:
    """Verify the require_api_key decorator blocks unauthorized access."""

    def test_no_key_required_when_empty(self):
        original = config.API_KEY
        config.API_KEY = ""
        try:
            with A.app.test_client() as client:
                resp = client.get("/api/stats")
                assert resp.status_code == 200
        finally:
            config.API_KEY = original

    def test_missing_key_returns_401(self):
        original = config.API_KEY
        config.API_KEY = "test-secret-key-123"
        try:
            with A.app.test_client() as client:
                resp = client.get("/api/stats")
                assert resp.status_code == 401
                data = resp.get_json()
                assert "error" in data
        finally:
            config.API_KEY = original

    def test_wrong_key_returns_401(self):
        original = config.API_KEY
        config.API_KEY = "test-secret-key-123"
        try:
            with A.app.test_client() as client:
                resp = client.get("/api/stats",
                                  headers={"X-API-Key": "wrong-key"})
                assert resp.status_code == 401
        finally:
            config.API_KEY = original

    def test_correct_key_returns_200(self):
        original = config.API_KEY
        config.API_KEY = "test-secret-key-123"
        try:
            with A.app.test_client() as client:
                resp = client.get("/api/stats",
                                  headers={"X-API-Key": "test-secret-key-123"})
                assert resp.status_code == 200
                data = resp.get_json()
                assert "db_stats" in data
        finally:
            config.API_KEY = original

    def test_export_requires_key(self):
        original = config.API_KEY
        config.API_KEY = "export-key"
        try:
            with A.app.test_client() as client:
                resp = client.get("/api/export")
                assert resp.status_code == 401
                resp = client.get("/api/export",
                                  headers={"X-API-Key": "export-key"})
                assert resp.status_code == 200
        finally:
            config.API_KEY = original


class TestInputValidation:
    """Verify socket event input validation helpers."""

    def test_validate_speed_normal(self):
        assert A._validate_speed(1.0) == 1.0
        assert A._validate_speed(5.0) == 5.0

    def test_validate_speed_clamps(self):
        assert A._validate_speed(0.01) == 0.1
        assert A._validate_speed(50.0) == 10.0

    def test_validate_speed_bad_input(self):
        assert A._validate_speed(None) == 1.0
        assert A._validate_speed("fast") == 1.0
        assert A._validate_speed([]) == 1.0

    def test_validate_family_valid(self):
        assert A._validate_family("Ryuk") == "Ryuk"
        assert A._validate_family("WannaCry") == "WannaCry"

    def test_validate_family_invalid(self):
        assert A._validate_family("NonExistent") is None
        assert A._validate_family(123) is None
        assert A._validate_family("") is None

    def test_validate_family_none(self):
        assert A._validate_family(None) is None


class TestPublicEndpoints:
    """Verify that public endpoints work without auth."""

    def test_status_endpoint(self):
        with A.app.test_client() as client:
            resp = client.get("/api/status")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "active_mode" in data
            assert "metrics" in data

    def test_history_endpoint(self):
        with A.app.test_client() as client:
            resp = client.get("/api/history")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "history" in data
            assert "alerts" in data

    def test_history_limit_validation(self):
        with A.app.test_client() as client:
            resp = client.get("/api/history?limit=50")
            assert resp.status_code == 200
            resp = client.get("/api/history?limit=999")
            assert resp.status_code == 200

    def test_models_endpoint(self):
        with A.app.test_client() as client:
            resp = client.get("/api/models")
            assert resp.status_code == 200
            assert "models" in resp.get_json()

    def test_folders_endpoint(self):
        with A.app.test_client() as client:
            resp = client.get("/api/folders")
            assert resp.status_code == 200
            assert "folders" in resp.get_json()
