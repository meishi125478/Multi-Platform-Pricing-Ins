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
