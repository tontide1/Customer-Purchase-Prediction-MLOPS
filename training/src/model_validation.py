"""Model validation gate with fail-closed contract.

Implements the validation gate for comparing new candidate models against
production models before deployment. The gate is fail-closed: any promotion
must be explicitly approved either through passing the validation criteria
or through manual override with full audit trail.

Contracts:
- If no production model exists → PASS (first deployment)
- If new model PR-AUC < min_threshold → FAIL (quality floor)
- If new model PR-AUC >= production PR-AUC → PASS (improvement or parity)
- Otherwise (new < production but >= min_threshold) → FAIL unless manual override
- Manual override requires: override_by, override_reason, override_time (all 3 or raise)
"""


def validate_model_gate(
    new_model_pr_auc: float,
    production_model_pr_auc: float | None,
    min_threshold: float,
    override_enabled: bool = False,
    override_by: str | None = None,
    override_reason: str | None = None,
    override_time: str | None = None,
) -> bool:
    """
    Validate a candidate model against the fail-closed validation gate.

    Args:
        new_model_pr_auc: PR-AUC score of the candidate model.
        production_model_pr_auc: PR-AUC score of the production model, or None if no
            production model exists.
        min_threshold: Minimum PR-AUC threshold; any model below this fails.
        override_enabled: Whether manual override is requested.
        override_by: Email or ID of the person authorizing the override (required if
            override_enabled is True).
        override_reason: Reason for the override (required if override_enabled is True).
        override_time: Timestamp of the override decision in ISO 8601 format (required
            if override_enabled is True).

    Returns:
        True if the model passes validation or is approved by override, False otherwise.

    Raises:
        ValueError: If override_enabled is True but any of override_by, override_reason,
            or override_time is None.
    """
    # Fail-closed: Check if any required audit field is missing when override is enabled
    if override_enabled:
        if override_by is None:
            raise ValueError("override_by is required when override_enabled is True")
        if override_reason is None:
            raise ValueError(
                "override_reason is required when override_enabled is True"
            )
        if override_time is None:
            raise ValueError("override_time is required when override_enabled is True")
        # All three audit fields are present → approve override
        return True

    # No production model exists → PASS (first deployment)
    if production_model_pr_auc is None:
        return True

    # New model PR-AUC < min_threshold → FAIL (quality floor)
    if new_model_pr_auc < min_threshold:
        return False

    # New model PR-AUC >= production PR-AUC → PASS (improvement or parity)
    if new_model_pr_auc >= production_model_pr_auc:
        return True

    # Otherwise: new < production but >= min_threshold → FAIL (regression without override)
    return False
