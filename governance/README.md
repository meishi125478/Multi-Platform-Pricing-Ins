# governance

## Purpose

`governance` owns model lifecycle controls: version registry, approval workflows, audit logging,
deployment state, and rollback metadata. Storage is file-based (JSON/JSONL), so no database is
required.

## Use When / Not For

- Use when a model version must be registered, approved, promoted, deployed, or rolled back.
- Use when auditability and traceability are required for model changes.
- Not for scoring/inference execution (handled by `production`).
- Not for training and hyperparameter search (handled by `modelling`).

## Public Entrypoints

- Registries and release: `ModelRegistry`, `ReleaseManager`, `ModelVersion`, `ModelArtifact`, `ModelRef`, `DeploymentState`
- Approvals: `ApprovalStore`, `ApprovalRequest`, `ApprovalAction`
- Auditing: `AuditLogger`, `AuditEvent`

## Minimal Flow

```python
from ins_pricing.governance import ModelRegistry, ApprovalStore, ReleaseManager, AuditLogger

registry = ModelRegistry("Registry/models.json")
registry.register("pricing_ft", "v1", metrics={"rmse": 0.12}, status="candidate")

approvals = ApprovalStore("Registry/approvals.json")
approvals.request("pricing_ft", "v1", requested_by="data_science")
approvals.act("pricing_ft", "v1", actor="risk_lead", decision="approved")

release = ReleaseManager("Registry/deployments", registry=registry)
release.deploy("prod", "pricing_ft", "v1", actor="ops")
AuditLogger("Registry/audit").log("deploy", actor="ops", model_name="pricing_ft")
```

## Further Reading

- Public export index: [../docs/api_reference.md](../docs/api_reference.md)
- Package navigation: [../README.md](../README.md)
