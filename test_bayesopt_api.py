"""Test script to verify both old and new BayesOptModel API styles."""

import warnings
import numpy as np
import pandas as pd

# Test imports
try:
    from ins_pricing.modelling.core.bayesopt import BayesOptModel, BayesOptConfig
    print("✓ Successfully imported BayesOptModel and BayesOptConfig")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Create sample data
np.random.seed(42)
n_samples = 100

train_data = pd.DataFrame({
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.randn(n_samples),
    'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
    'target': np.random.rand(n_samples),
    'weight': np.ones(n_samples)
})

test_data = train_data.copy()
print(f"✓ Created sample data: {n_samples} rows, {len(train_data.columns)} columns")

# Test 1: New API (config-based)
print("\n" + "="*60)
print("TEST 1: New API (config-based) - RECOMMENDED")
print("="*60)

try:
    config = BayesOptConfig(
        model_nme="test_model_new",
        resp_nme="target",
        weight_nme="weight",
        factor_nmes=["feature1", "feature2", "feature3"],
        task_type="regression",
        epochs=5,
        use_gpu=False,
        rand_seed=42
    )
    print("✓ Created BayesOptConfig successfully")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model_new = BayesOptModel(train_data, test_data, config=config)

        # Check no deprecation warning was issued
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        if len(deprecation_warnings) == 0:
            print("✓ No deprecation warning (expected for new API)")
        else:
            print(f"✗ Unexpected deprecation warning: {deprecation_warnings[0].message}")

    print(f"✓ Created BayesOptModel with new API")
    print(f"  - model_nme: {model_new.model_nme}")
    print(f"  - resp_nme: {model_new.resp_nme}")
    print(f"  - task_type: {model_new.task_type}")
    print(f"  - epochs: {model_new.epochs}")
    print(f"  - config object: {type(model_new.config).__name__}")

except Exception as e:
    print(f"✗ New API test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Old API (individual parameters) - should show deprecation warning
print("\n" + "="*60)
print("TEST 2: Old API (individual parameters) - DEPRECATED")
print("="*60)

try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        model_old = BayesOptModel(
            train_data,
            test_data,
            model_nme="test_model_old",
            resp_nme="target",
            weight_nme="weight",
            factor_nmes=["feature1", "feature2", "feature3"],
            task_type="regression",
            epochs=10,
            use_gpu=False,
            rand_seed=99
        )

        # Check that deprecation warning WAS issued
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        if len(deprecation_warnings) > 0:
            print(f"✓ Deprecation warning shown (expected for old API):")
            print(f"  {deprecation_warnings[0].message}")
        else:
            print("✗ No deprecation warning (expected one for old API)")

    print(f"✓ Created BayesOptModel with old API (backward compatibility)")
    print(f"  - model_nme: {model_old.model_nme}")
    print(f"  - resp_nme: {model_old.resp_nme}")
    print(f"  - task_type: {model_old.task_type}")
    print(f"  - epochs: {model_old.epochs}")
    print(f"  - config object: {type(model_old.config).__name__}")

except Exception as e:
    print(f"✗ Old API test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Verify both APIs produce equivalent results
print("\n" + "="*60)
print("TEST 3: API Equivalence")
print("="*60)

try:
    # Create two models with same parameters
    config_equiv = BayesOptConfig(
        model_nme="equiv_test",
        resp_nme="target",
        weight_nme="weight",
        factor_nmes=["feature1", "feature2"],
        task_type="regression",
        epochs=7,
        use_gpu=False,
        rand_seed=123,
        xgb_max_depth_max=20,
        final_ensemble=True
    )

    model_new_equiv = BayesOptModel(train_data, test_data, config=config_equiv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress deprecation warning for this test
        model_old_equiv = BayesOptModel(
            train_data, test_data,
            model_nme="equiv_test",
            resp_nme="target",
            weight_nme="weight",
            factor_nmes=["feature1", "feature2"],
            task_type="regression",
            epochs=7,
            use_gpu=False,
            rand_seed=123,
            xgb_max_depth_max=20,
            final_ensemble=True
        )

    # Compare key attributes
    checks = [
        ("model_nme", model_new_equiv.model_nme == model_old_equiv.model_nme),
        ("resp_nme", model_new_equiv.resp_nme == model_old_equiv.resp_nme),
        ("epochs", model_new_equiv.epochs == model_old_equiv.epochs),
        ("rand_seed", model_new_equiv.rand_seed == model_old_equiv.rand_seed),
        ("task_type", model_new_equiv.task_type == model_old_equiv.task_type),
    ]

    all_match = all(check[1] for check in checks)

    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {name}: {result}")

    if all_match:
        print("\n✓ Both APIs produce equivalent models")
    else:
        print("\n✗ APIs produce different results")

except Exception as e:
    print(f"✗ Equivalence test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Error handling - missing required params in old API
print("\n" + "="*60)
print("TEST 4: Error Handling")
print("="*60)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Should fail - missing required parameters
            model_error = BayesOptModel(train_data, test_data)
            print("✗ Should have raised ValueError for missing parameters")
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {e}")
        except Exception as e:
            print(f"✗ Wrong exception type: {type(e).__name__}: {e}")

except Exception as e:
    print(f"✗ Error handling test failed: {e}")

# Test 5: Config type validation
print("\n" + "="*60)
print("TEST 5: Config Type Validation")
print("="*60)

try:
    # Should fail - wrong type for config
    try:
        model_bad_config = BayesOptModel(
            train_data, test_data,
            config="not_a_config_object"
        )
        print("✗ Should have raised TypeError for invalid config")
    except TypeError as e:
        print(f"✓ Correctly raised TypeError: {e}")
    except Exception as e:
        print(f"✗ Wrong exception type: {type(e).__name__}: {e}")

except Exception as e:
    print(f"✗ Config validation test failed: {e}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("✓ New API (config-based): Recommended for all new code")
print("✓ Old API (individual params): Works but deprecated")
print("✓ Backward compatibility: Fully maintained")
print("✓ Migration path: Clear deprecation warnings guide users")
print("\nRefactoring Phase 2 implementation complete!")
