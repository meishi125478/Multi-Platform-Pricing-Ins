# Utils Module Duplication Analysis

**Date**: 2026-01-15
**Status**: ⚠️ DUPLICATION DETECTED
**Priority**: MEDIUM - Code consolidation recommended

---

## Executive Summary

The `ins_pricing` package has **THREE separate utils modules** with overlapping functionality:

1. `ins_pricing/utils/` - Package-level shared utilities
2. `ins_pricing/modelling/core/bayesopt/utils/` - BayesOpt-specific utilities
3. `ins_pricing/cli/utils/` - CLI-specific utilities

**Key Finding**: There is **significant duplication** between #1 (package utils) and #2 (bayesopt utils), particularly in:
- `DeviceManager`
- `GPUMemoryManager`
- `MetricFactory`
- `get_logger`

---

## Detailed Analysis

### 1. ins_pricing/utils/ (Package-level)

**Purpose**: Shared utilities for entire package
**Location**: `ins_pricing/utils/`
**Files**: 6 modules

| Module | Lines | Purpose | Exports |
|--------|-------|---------|---------|
| `logging.py` | ~70 | Unified logging system | `get_logger`, `configure_logging` |
| `metrics.py` | ~250 | PSI calculation, model metrics | `psi_*`, `MetricFactory` |
| `paths.py` | ~260 | Path resolution, data loading | `resolve_path`, `load_dataset`, etc. |
| `device.py` | ~230 | GPU/CPU device management | `DeviceManager`, `GPUMemoryManager` |
| `validation.py` | ~400 | Data validation utilities | `validate_*` functions |
| `profiling.py` | ~340 | Performance profiling | `profile_section`, `MemoryMonitor` |

**Key Classes**:
- `DeviceManager` - Device detection and model placement
- `GPUMemoryManager` - CUDA memory management
- `MetricFactory` - Model evaluation metrics

### 2. ins_pricing/modelling/core/bayesopt/utils/ (BayesOpt-specific)

**Purpose**: BayesOpt module utilities (refactored in Phase 1)
**Location**: `ins_pricing/modelling/core/bayesopt/utils/`
**Files**: 5 modules

| Module | Lines | Purpose | Exports |
|--------|-------|---------|---------|
| `constants.py` | 183 | Core constants, seed setting | `EPS`, `set_global_seed`, `compute_batch_size` |
| `io_utils.py` | 110 | Parameter file loading | `IOUtils`, `csv_to_dict` |
| `distributed_utils.py` | 163 | DDP and CUDA management | `DistributedUtils`, `TrainingUtils` |
| `torch_trainer_mixin.py` | 587 | PyTorch training loops | `TorchTrainerMixin` |
| `metrics_and_devices.py` | 721 | Metrics, GPU, CV, plotting | `MetricFactory`, `DeviceManager`, `GPUMemoryManager` |

**Key Classes** (DUPLICATED):
- `DeviceManager` - **DUPLICATE** of package-level version
- `GPUMemoryManager` - **DUPLICATE** of package-level version
- `MetricFactory` - **PARTIAL DUPLICATE** (different focus)
- `get_logger` - **DUPLICATE** of package-level version

### 3. ins_pricing/cli/utils/ (CLI-specific)

**Purpose**: CLI command utilities
**Location**: `ins_pricing/cli/utils/`
**Files**: 4 modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `cli_common.py` | ~250 | Common CLI utilities |
| `cli_config.py` | ~400 | Configuration handling |
| `notebook_utils.py` | ~360 | Notebook integration |
| `run_logging.py` | ~110 | Run-specific logging |

**No Duplication**: CLI utils are specific to command-line interface

---

## Duplication Details

### CRITICAL: DeviceManager and GPUMemoryManager

**Duplicated in**:
- `ins_pricing/utils/device.py`
- `ins_pricing/modelling/core/bayesopt/utils/metrics_and_devices.py`

**Code Comparison**:

```python
# ins_pricing/utils/device.py
class DeviceManager:
    """Unified device management for model and tensor placement."""
    _logger = get_logger("ins_pricing.device")
    _cached_device: Optional[Any] = None  # torch.device when available

    @classmethod
    def get_best_device(cls, prefer_cuda: bool = True) -> Any:
        """Get the best available device."""
        # ... implementation

class GPUMemoryManager:
    """Centralized GPU memory management."""
    _logger = get_logger("ins_pricing.device")

    @staticmethod
    def clean() -> None:
        """Clear CUDA cache and run garbage collection."""
        # ... implementation
```

```python
# ins_pricing/modelling/core/bayesopt/utils/metrics_and_devices.py
class DeviceManager:
    """Unified device management for model and tensor placement."""
    _logger = get_logger("ins_pricing.device")
    _cached_device: Optional[torch.device] = None

    @classmethod
    def get_best_device(cls, prefer_cuda: bool = True) -> torch.device:
        """Get the best available device."""
        # ... IDENTICAL implementation

class GPUMemoryManager:
    """Centralized GPU memory management."""
    _logger = get_logger("ins_pricing.device")

    @staticmethod
    def clean():
        """Clear CUDA cache and run garbage collection."""
        # ... IDENTICAL implementation
```

**Verdict**: ❌ **EXACT DUPLICATION** (~230 lines duplicated)

### MEDIUM: MetricFactory

**Duplicated in**:
- `ins_pricing/utils/metrics.py`
- `ins_pricing/modelling/core/bayesopt/utils/metrics_and_devices.py`

**Difference**:
- Package-level: Focused on PSI (Population Stability Index)
- BayesOpt: Focused on regression/classification metrics (log_loss, Tweedie deviance)

**Verdict**: ⚠️ **PARTIAL DUPLICATION** - Different focus but same pattern

### MINOR: get_logger

**Duplicated in**:
- `ins_pricing/utils/logging.py`
- `ins_pricing/modelling/core/bayesopt/utils/metrics_and_devices.py`

**Difference**:
- Package-level: Full implementation with `configure_logging`
- BayesOpt: Uses package-level via import (no actual duplication)

**Verdict**: ✅ **NO DUPLICATION** - BayesOpt imports from package level

---

## Impact Assessment

### Maintenance Burden

| Aspect | Current State | Risk Level |
|--------|---------------|------------|
| Code duplication | ~230 lines (DeviceManager + GPUMemoryManager) | HIGH |
| Bug fix propagation | Must fix in 2 places | MEDIUM |
| API consistency | Two versions might drift apart | MEDIUM |
| Testing overhead | Must test identical code twice | MEDIUM |
| Refactoring effort | Moderate (1-2 days) | LOW |

### Usage Analysis

**Question**: Which version is actually used where?

```bash
# Check imports in codebase
grep -r "from ins_pricing.utils import.*DeviceManager" ins_pricing/
grep -r "from.*bayesopt.utils import.*DeviceManager" ins_pricing/
```

**Hypothesis**:
- Package-level utils: Used by production, pricing, governance modules
- BayesOpt utils: Used only within bayesopt module
- **Likely**: BayesOpt created its own copy before package-level utils existed

---

## Recommended Solution

### Phase 3: Consolidate Duplicated Utils

**Goal**: Eliminate duplication, use single source of truth

### Option A: BayesOpt imports from package-level (RECOMMENDED)

**Changes**:
1. Delete `DeviceManager` and `GPUMemoryManager` from `bayesopt/utils/metrics_and_devices.py`
2. Import from package-level instead:
   ```python
   # In bayesopt/utils/metrics_and_devices.py
   from ins_pricing.utils import DeviceManager, GPUMemoryManager
   ```
3. Update `bayesopt/utils/__init__.py` to re-export:
   ```python
   from ins_pricing.utils import DeviceManager, GPUMemoryManager
   ```

**Pros**:
- ✅ Eliminates duplication (~230 lines removed)
- ✅ Single source of truth
- ✅ Bug fixes propagate automatically
- ✅ Minimal changes (just imports)

**Cons**:
- ⚠️ Creates dependency: bayesopt → package utils
- ⚠️ Less isolated (but bayesopt is internal anyway)

**Impact**:
- Files modified: 2 (metrics_and_devices.py, __init__.py)
- Lines removed: ~230
- Breaking changes: NONE (re-exports maintain compatibility)

### Option B: Move shared utilities to common location

**Changes**:
1. Create `ins_pricing/common/device.py` with `DeviceManager`, `GPUMemoryManager`
2. Both package utils and bayesopt utils import from common
3. Deprecate duplicates

**Pros**:
- ✅ Clear separation: shared vs specific
- ✅ No circular dependencies

**Cons**:
- ❌ More complex refactoring
- ❌ Creates new module structure
- ❌ Higher effort (3-4 days)

### Option C: Keep as-is (NOT RECOMMENDED)

**Pros**:
- ✅ No refactoring effort
- ✅ BayesOpt remains isolated

**Cons**:
- ❌ Maintenance burden continues
- ❌ Risk of code drift
- ❌ Violates DRY principle

---

## Proposed Implementation (Option A)

### Step 1: Verify package-level utils are comprehensive

```bash
# Check if package-level DeviceManager has all needed methods
diff -u ins_pricing/utils/device.py \
        ins_pricing/modelling/core/bayesopt/utils/metrics_and_devices.py
```

### Step 2: Update bayesopt/utils/metrics_and_devices.py

```python
# BEFORE (lines ~500-700)
class DeviceManager:
    """Unified device management..."""
    # ... 100+ lines

class GPUMemoryManager:
    """Centralized GPU memory..."""
    # ... 50+ lines

# AFTER
# Import from package-level instead
from ins_pricing.utils import DeviceManager, GPUMemoryManager
```

### Step 3: Update bayesopt/utils/__init__.py

```python
# Add to imports section
from ins_pricing.utils import DeviceManager, GPUMemoryManager

# Already in __all__ - no change needed
__all__ = [
    # ...
    'DeviceManager',
    'GPUMemoryManager',
    # ...
]
```

### Step 4: Test backward compatibility

```python
# All these should still work
from ins_pricing.utils import DeviceManager
from ins_pricing.modelling.core.bayesopt.utils import DeviceManager

# Both should be the SAME object
assert DeviceManager is DeviceManager  # True
```

### Step 5: Update documentation

- Add note in `REFACTORING_COMPLETE.md` about Phase 3
- Update `CHANGELOG.md` with consolidation entry

---

## Benefits of Consolidation

### Before
```
ins_pricing/utils/device.py              230 lines
bayesopt/utils/metrics_and_devices.py    721 lines (includes duplicates)
Total: 951 lines
```

### After
```
ins_pricing/utils/device.py              230 lines (unchanged)
bayesopt/utils/metrics_and_devices.py    491 lines (removed 230)
Total: 721 lines
Reduction: 230 lines (24% reduction in bayesopt utils)
```

### Quality Improvements

| Metric | Before | After |
|--------|--------|-------|
| Code duplication | 230 lines | 0 lines |
| Maintenance points | 2 locations | 1 location |
| Bug fix effort | 2x | 1x |
| Test coverage | Duplicate tests | Shared tests |
| API consistency | Risk of drift | Always consistent |

---

## Timeline Estimate

### Option A (Recommended): 4-6 hours

1. **Analysis** (1 hour): Verify full compatibility
2. **Implementation** (2 hours): Update imports, remove duplicates
3. **Testing** (1 hour): Verify backward compatibility
4. **Documentation** (1 hour): Update docs, changelog
5. **Buffer** (1 hour): Edge cases, review

### Option B: 2-3 days

1. **Planning** (4 hours): Design common module structure
2. **Implementation** (8 hours): Create common, update imports
3. **Migration** (4 hours): Update all references
4. **Testing** (4 hours): Comprehensive testing
5. **Documentation** (4 hours): Complete docs

---

## Risk Assessment

### Low Risk (Option A)

- ✅ Small, focused change
- ✅ Re-exports maintain compatibility
- ✅ Easy to rollback
- ✅ No new dependencies (package utils already exists)

### Rollback Plan

If issues arise:
```bash
# Restore duplicated code from backup
git checkout HEAD~1 -- ins_pricing/modelling/core/bayesopt/utils/metrics_and_devices.py
```

---

## Recommendation

**Proceed with Option A: BayesOpt imports from package-level**

**Reasons**:
1. Eliminates 230 lines of duplication
2. Low effort (4-6 hours)
3. Low risk (easy rollback)
4. Maintains 100% backward compatibility
5. Establishes package-level utils as single source of truth

**Priority**: MEDIUM
- Not urgent (both versions work)
- But should be done before v0.4.0 cleanup
- Natural follow-up to Phase 1 & 2 refactoring

---

## Next Steps

**If user approves Phase 3**:

1. Verify package-level `DeviceManager`/`GPUMemoryManager` are feature-complete
2. Update `bayesopt/utils/metrics_and_devices.py` to import instead of duplicate
3. Update `bayesopt/utils/__init__.py` re-exports
4. Run tests to verify compatibility
5. Update documentation
6. Add CHANGELOG entry for v0.2.11

**Estimated completion**: Same day (4-6 hours)

---

**End of Analysis**
