"""Tests for audit logging module."""

import pytest
from datetime import datetime
from pathlib import Path

from ins_pricing.exceptions import GovernanceError


class TestAuditLogging:
    """Test audit logging functionality."""

    def test_log_model_action(self, tmp_path):
        """Test logging a model action."""
        from ins_pricing.governance.audit import AuditLogger

        logger = AuditLogger(audit_dir=tmp_path)
        logger.log(
            action="model_registered",
            model_name="test_model",
            user="test_user",
            details={"version": "1.0.0"}
        )

        logs = logger.get_logs(model_name="test_model")
        assert len(logs) > 0
        assert logs[0]["action"] == "model_registered"

    def test_get_audit_trail(self, tmp_path):
        """Test retrieving complete audit trail."""
        from ins_pricing.governance.audit import AuditLogger

        logger = AuditLogger(audit_dir=tmp_path)

        # Log multiple actions
        logger.log("registered", "model_a", "user1")
        logger.log("trained", "model_a", "user1")
        logger.log("deployed", "model_a", "user2")

        trail = logger.get_audit_trail("model_a")

        assert len(trail) == 3
        assert trail[-1]["action"] == "deployed"

    def test_filter_logs_by_date(self, tmp_path):
        """Test filtering audit logs by date range."""
        from ins_pricing.governance.audit import AuditLogger

        logger = AuditLogger(audit_dir=tmp_path)
        logger.log("action1", "model", "user")

        # Filter by today
        today = datetime.now().date()
        logs = logger.get_logs(start_date=today, end_date=today)

        assert len(logs) > 0
