"""Tests for model release module."""

import pytest
from pathlib import Path
from datetime import datetime

from ins_pricing.exceptions import GovernanceError


class TestModelRelease:
    """Test model release workflow."""

    def test_create_release(self, tmp_path):
        """Test creating a new model release."""
        from ins_pricing.governance.release import ReleaseManager

        manager = ReleaseManager(release_dir=tmp_path)
        release_id = manager.create_release(
            model_name="test_model",
            version="1.0.0",
            artifacts=["model.pkl", "config.json"]
        )

        assert release_id is not None
        assert manager.release_exists(release_id)

    def test_get_release_info(self, tmp_path):
        """Test getting release information."""
        from ins_pricing.governance.release import ReleaseManager

        manager = ReleaseManager(release_dir=tmp_path)
        release_id = manager.create_release(
            model_name="test_model",
            version="1.0.0"
        )

        info = manager.get_release_info(release_id)

        assert info["model_name"] == "test_model"
        assert info["version"] == "1.0.0"

    def test_promote_release(self, tmp_path):
        """Test promoting a release to production."""
        from ins_pricing.governance.release import ReleaseManager

        manager = ReleaseManager(release_dir=tmp_path)
        release_id = manager.create_release(
            model_name="test_model",
            version="1.0.0"
        )

        manager.promote_to_production(release_id)

        info = manager.get_release_info(release_id)
        assert info["status"] == "production"

    def test_rollback_release(self, tmp_path):
        """Test rolling back a release."""
        from ins_pricing.governance.release import ReleaseManager

        manager = ReleaseManager(release_dir=tmp_path)

        # Create and promote two releases
        release1 = manager.create_release("test_model", "1.0.0")
        manager.promote_to_production(release1)

        release2 = manager.create_release("test_model", "2.0.0")
        manager.promote_to_production(release2)

        # Rollback to version 1.0.0
        manager.rollback_to(release1)

        current = manager.get_production_release("test_model")
        assert current["version"] == "1.0.0"

    def test_deploy_rejects_invalid_env_name(self, tmp_path):
        from ins_pricing.governance.release import ReleaseManager

        manager = ReleaseManager(release_dir=tmp_path)
        with pytest.raises(GovernanceError):
            manager.deploy(env="../prod", name="pricing_model", version="1.0.0")

    def test_deploy_reverts_state_when_registry_promote_fails(self, tmp_path):
        from ins_pricing.governance.release import ReleaseManager

        class _FailingRegistry:
            @staticmethod
            def promote(name, version, new_status="production"):
                _ = name, version, new_status
                raise RuntimeError("registry unavailable")

        manager = ReleaseManager(release_dir=tmp_path, registry=_FailingRegistry())
        with pytest.raises(GovernanceError, match="deployment state reverted"):
            manager.deploy(env="staging", name="pricing_model", version="1.0.0")

        assert manager.get_active("staging") is None

    def test_promote_release_reverts_manifest_when_registry_promote_fails(self, tmp_path):
        from ins_pricing.governance.release import ReleaseManager

        class _FailingRegistry:
            @staticmethod
            def promote(name, version, new_status="production"):
                _ = name, version, new_status
                raise RuntimeError("registry unavailable")

        manager = ReleaseManager(release_dir=tmp_path, registry=_FailingRegistry())
        release_id = manager.create_release(model_name="pricing_model", version="1.0.0")

        with pytest.raises(GovernanceError, match="manifest reverted"):
            manager.promote_to_production(release_id)

        info = manager.get_release_info(release_id)
        assert info["status"] == "candidate"

    def test_rollback_reverts_manifest_when_registry_promote_fails(self, tmp_path):
        from ins_pricing.governance.release import ReleaseManager

        class _FlakyRegistry:
            def __init__(self):
                self.calls = 0

            def promote(self, name, version, new_status="production"):
                _ = name, version, new_status
                self.calls += 1
                if self.calls >= 3:
                    raise RuntimeError("registry unavailable")

        registry = _FlakyRegistry()
        manager = ReleaseManager(release_dir=tmp_path, registry=registry)
        release1 = manager.create_release(model_name="pricing_model", version="1.0.0")
        release2 = manager.create_release(model_name="pricing_model", version="2.0.0")
        manager.promote_to_production(release1)
        manager.promote_to_production(release2)

        with pytest.raises(GovernanceError, match="manifest reverted"):
            manager.rollback_to(release1)

        current = manager.get_production_release("pricing_model")
        assert current["version"] == "2.0.0"
