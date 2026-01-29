# governance

Lightweight model registry, approval, audit log, and release management.

Example (deploy + rollback):

```python
from ins_pricing.governance import ModelRegistry, ReleaseManager

registry = ModelRegistry("Registry/models.json")
release = ReleaseManager("Registry/deployments", registry=registry)

registry.register("pricing_ft", "v1", metrics={"rmse": 0.12})
release.deploy("prod", "pricing_ft", "v1", actor="ops")

# rollback to previous active version
release.rollback("prod", actor="ops")
```
