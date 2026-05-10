"""Tests for the model validation gate module.

Tests the fail-closed validation contract:
- No production model → PASS
- New PR-AUC < min_threshold → FAIL
- New PR-AUC >= production PR-AUC → PASS
- Otherwise (new < production and >= min_threshold) → FAIL unless manual override
- Manual override requires all three audit fields: override_by, override_reason, override_time
"""

import pytest

from training.src.model_validation import validate_model_gate


class TestValidationGate:
    """Test suite for fail-closed model validation gate."""

    def test_no_production_model_returns_pass(self):
        """Test: No production model exists → PASS."""
        result = validate_model_gate(
            new_model_pr_auc=0.65,
            production_model_pr_auc=None,
            min_threshold=0.5,
        )
        assert result is True

    def test_new_model_below_min_threshold_returns_fail(self):
        """Test: New model PR-AUC < min_threshold → FAIL."""
        result = validate_model_gate(
            new_model_pr_auc=0.45,
            production_model_pr_auc=0.60,
            min_threshold=0.5,
        )
        assert result is False

    def test_new_model_meets_or_exceeds_production_returns_pass(self):
        """Test: New model PR-AUC >= production PR-AUC → PASS."""
        result = validate_model_gate(
            new_model_pr_auc=0.70,
            production_model_pr_auc=0.65,
            min_threshold=0.5,
        )
        assert result is True

    def test_new_model_equal_to_production_returns_pass(self):
        """Test: New model PR-AUC == production PR-AUC → PASS."""
        result = validate_model_gate(
            new_model_pr_auc=0.65,
            production_model_pr_auc=0.65,
            min_threshold=0.5,
        )
        assert result is True

    def test_new_model_between_threshold_and_production_fails_without_override(self):
        """Test: New < production but >= min_threshold → FAIL (no override)."""
        result = validate_model_gate(
            new_model_pr_auc=0.60,
            production_model_pr_auc=0.65,
            min_threshold=0.5,
        )
        assert result is False

    def test_manual_override_with_all_audit_fields_returns_pass(self):
        """Test: Manual override with all three audit fields → PASS."""
        result = validate_model_gate(
            new_model_pr_auc=0.60,
            production_model_pr_auc=0.65,
            min_threshold=0.5,
            override_enabled=True,
            override_by="alice@example.com",
            override_reason="Approved by MLOps team after review",
            override_time="2026-05-10T14:30:00Z",
        )
        assert result is True

    def test_override_enabled_but_missing_override_by_raises_error(self):
        """Test: Override enabled but override_by missing → ValueError."""
        with pytest.raises(ValueError, match="override_by"):
            validate_model_gate(
                new_model_pr_auc=0.60,
                production_model_pr_auc=0.65,
                min_threshold=0.5,
                override_enabled=True,
                override_by=None,
                override_reason="Approved by MLOps team",
                override_time="2026-05-10T14:30:00Z",
            )

    def test_override_enabled_but_missing_override_reason_raises_error(self):
        """Test: Override enabled but override_reason missing → ValueError."""
        with pytest.raises(ValueError, match="override_reason"):
            validate_model_gate(
                new_model_pr_auc=0.60,
                production_model_pr_auc=0.65,
                min_threshold=0.5,
                override_enabled=True,
                override_by="alice@example.com",
                override_reason=None,
                override_time="2026-05-10T14:30:00Z",
            )

    def test_override_enabled_but_missing_override_time_raises_error(self):
        """Test: Override enabled but override_time missing → ValueError."""
        with pytest.raises(ValueError, match="override_time"):
            validate_model_gate(
                new_model_pr_auc=0.60,
                production_model_pr_auc=0.65,
                min_threshold=0.5,
                override_enabled=True,
                override_by="alice@example.com",
                override_reason="Approved by MLOps team",
                override_time=None,
            )
