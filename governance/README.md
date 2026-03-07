# governance

Lightweight model registry, approval workflows, audit logging, and release management.
All state is persisted to JSON/JSONL files — no database required.

## Modules

| File | Description |
|------|-------------|
| `registry.py` | Model versioning, metadata, promotion |
| `approval.py` | Multi-actor approval requests and decisions |
| `audit.py` | Append-only audit event logging with filtering |
| `release.py` | Deployment state, rollback, release packaging |

## Quick Start

```python
from ins_pricing.governance import ModelRegistry, ReleaseManager, ApprovalStore, AuditLogger

# Register a model version
registry = ModelRegistry("Registry/models.json")
registry.register("pricing_ft", "v1", metrics={"rmse": 0.12}, status="candidate")

# Approval workflow
approvals = ApprovalStore("Registry/approvals.json")
approvals.request("pricing_ft", "v1", requested_by="data_science")
approvals.act("pricing_ft", "v1", actor="risk_lead", decision="approved")

# Promote and deploy
registry.promote("pricing_ft", "v1", new_status="production")
release = ReleaseManager("Registry/deployments", registry=registry)
release.deploy("prod", "pricing_ft", "v1", actor="ops")

# Rollback
release.rollback("prod", actor="ops")

# Audit trail
audit = AuditLogger(audit_dir="Registry/audit")
audit.log("deploy", actor="ops", model_name="pricing_ft", metadata={"env": "prod"})
trail = audit.get_audit_trail("pricing_ft")
```

## API Reference

### ModelRegistry

- `register(name, version, *, metrics=None, tags=None, artifacts=None, status="candidate", notes=None)` - register a model version
- `exists(name)` / `get(name, version=None)` / `get_latest(name)` - lookup
- `list_all()` / `list_versions(name)` / `get_versions(name)` - enumeration
- `update(name, updates, *, version=None)` - update metadata
- `delete(name, *, version=None)` - remove entry
- `promote(name, version, *, new_status="production")` - change status

### ApprovalStore

- `request(model_name, model_version, requested_by)` - create approval request
- `list_requests(model_name=None)` - list pending/all requests
- `act(model_name, model_version, *, actor, decision, comment=None)` - approve/reject

### AuditLogger

- `log(action, *args, model_name=None, actor=None, metadata=None, note=None)` - append event
- `get_logs(*, model_name=None, start_date=None, end_date=None)` - filtered query
- `get_audit_trail(model_name)` - all events for a model

### ReleaseManager

- `deploy(env, name, version, *, actor=None, note=None, update_registry_status=True)` - activate version
- `rollback(env, *, actor=None, note=None)` - revert to previous version
- `get_active(env)` / `list_history(env)` - inspect state
- `create_release(model_name, version, artifacts=None)` - package release
- `promote_to_production(release_id)` / `rollback_to(release_id)` - release lifecycle

### Data Classes

- `ModelVersion` - name, version, created_at, metrics, tags, artifacts, status, notes
- `ModelArtifact` - path, description
- `ApprovalRequest` / `ApprovalAction` - request and decision records
- `AuditEvent` - action, actor, timestamp, metadata, note
- `ModelRef` / `DeploymentState` - deployment tracking
