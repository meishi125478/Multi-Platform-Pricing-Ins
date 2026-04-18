"""Tests for model registry module."""

import json
from pathlib import Path
import pytest
from datetime import datetime

from ins_pricing.exceptions import GovernanceError


@pytest.fixture
def sample_model_metadata():
    """Sample model metadata."""
    return {
        "model_name": "test_model_v1",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "model_type": "xgboost",
        "metrics": {"mse": 100.5, "r2": 0.85},
        "features": ["age", "premium", "region"],
        "author": "test_user"
    }


class TestModelRegistry:
    """Test model registry functionality."""

    def test_register_new_model(self, tmp_path, sample_model_metadata):
        """Test registering a new model."""
        from ins_pricing.governance.registry import ModelRegistry

        registry = ModelRegistry(registry_path=tmp_path / "registry.json")
        registry.register(sample_model_metadata)

        assert registry.exists(sample_model_metadata["model_name"])

    def test_duplicate_registration_error(self, tmp_path, sample_model_metadata):
        """Test error on duplicate model registration."""
        from ins_pricing.governance.registry import ModelRegistry

        registry = ModelRegistry(registry_path=tmp_path / "registry.json")
        registry.register(sample_model_metadata)

        with pytest.raises(GovernanceError):
            registry.register(sample_model_metadata)  # Duplicate

    def test_get_model_metadata(self, tmp_path, sample_model_metadata):
        """Test retrieving model metadata."""
        from ins_pricing.governance.registry import ModelRegistry

        registry = ModelRegistry(registry_path=tmp_path / "registry.json")
        registry.register(sample_model_metadata)

        metadata = registry.get(sample_model_metadata["model_name"])

        assert metadata["version"] == "1.0.0"
        assert "metrics" in metadata

    def test_list_all_models(self, tmp_path):
        """Test listing all registered models."""
        from ins_pricing.governance.registry import ModelRegistry

        registry = ModelRegistry(registry_path=tmp_path / "registry.json")
        registry.register({"model_name": "model_a", "version": "1.0.0"})
        registry.register({"model_name": "model_b", "version": "2.0.0"})

        models = registry.list_all()

        assert len(models) == 2
        assert any(m["model_name"] == "model_a" for m in models)

    def test_update_model_metadata(self, tmp_path, sample_model_metadata):
        """Test updating model metadata."""
        from ins_pricing.governance.registry import ModelRegistry

        registry = ModelRegistry(registry_path=tmp_path / "registry.json")
        registry.register(sample_model_metadata)

        # Update metrics
        registry.update(
            sample_model_metadata["model_name"],
            {"metrics": {"mse": 95.0, "r2": 0.87}}
        )

        updated = registry.get(sample_model_metadata["model_name"])
        assert updated["metrics"]["mse"] == 95.0

    def test_delete_model(self, tmp_path, sample_model_metadata):
        """Test deleting a model from registry."""
        from ins_pricing.governance.registry import ModelRegistry

        registry = ModelRegistry(registry_path=tmp_path / "registry.json")
        registry.register(sample_model_metadata)

        registry.delete(sample_model_metadata["model_name"])

        assert not registry.exists(sample_model_metadata["model_name"])


class TestModelVersioning:
    """Test model versioning functionality."""

    def test_register_multiple_versions(self, tmp_path):
        """Test registering multiple versions of same model."""
        from ins_pricing.governance.registry import ModelRegistry

        registry = ModelRegistry(registry_path=tmp_path / "registry.json")

        registry.register({"model_name": "my_model", "version": "1.0.0"})
        registry.register({"model_name": "my_model", "version": "1.1.0"})

        versions = registry.get_versions("my_model")

        assert len(versions) == 2
        assert "1.0.0" in versions
        assert "1.1.0" in versions

    def test_get_latest_version(self, tmp_path):
        """Test getting the latest version of a model."""
        from ins_pricing.governance.registry import ModelRegistry

        registry = ModelRegistry(registry_path=tmp_path / "registry.json")
        registry.register({"model_name": "my_model", "version": "1.0.0"})
        registry.register({"model_name": "my_model", "version": "2.0.0"})

        latest = registry.get_latest("my_model")

        assert latest["version"] == "2.0.0"
