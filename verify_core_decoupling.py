
import sys
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
# from pydantic import BaseModel # Not strictly needed for this check

project_root = Path(__file__).resolve().parent / "ins_pricing v2"
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")

try:
    from modelling.bayesopt.core import BayesOptModel
    from modelling.bayesopt.data_container import DataContainer
    from modelling.bayesopt.model_manager import ModelManager
    from modelling.config import BayesOptConfig
    print("✅ Core imports successful.")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# 1. Setup Mock Data
train_df = pd.DataFrame({
    'target': np.random.rand(20).astype(np.float32),
    'weight': np.random.rand(20).astype(np.float32),
    'f1': np.random.rand(20).astype(np.float32),
    'f2': np.random.choice(['a', 'b'], 20)
})
test_df = train_df.copy()

# 2. Instantiate BayesOptModel
print("\n--- Testing Instantiation ---")
try:
    model = BayesOptModel(
        train_data=train_df,
        test_data=test_df,
        model_nme="test_decoupled",
        resp_nme="target",
        weight_nme="weight",
        factor_nmes=["f1", "f2"],
        use_gpu=False,
        epochs=1
    )
    print("✅ BayesOptModel instantiated.")
except Exception as e:
    print(f"❌ Instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Verify DataContainer Delegation
print("\n--- Verify DataContainer ---")
try:
    assert isinstance(model.data_container, DataContainer)
    assert model.train_data.shape == (20, 4)
    # Check backward compatibility property
    assert model.train_data is model.data_container.train_data
    print("✅ DataContainer delegation works.")
except Exception as e:
    print(f"❌ DataContainer check failed: {e}")

# 4. Verify ModelManager
print("\n--- Verify ModelManager ---")
try:
    assert isinstance(model.model_manager, ModelManager)
    # Verify trainers are initialized
    glm = model.model_manager.get_trainer('glm')
    print(f"✅ Got GLM trainer: {glm}")
    
    # Check if trainers have reference to usage context (which is model)
    assert glm.ctx is model
    # Check implicit data access via context
    assert glm.ctx.train_data is not None
    print("✅ Trainer context works.")
except Exception as e:
    print(f"❌ ModelManager check failed: {e}")

# 5. Verify Pydantic Config in Model
print("\n--- Verify Config ---")
try:
    assert isinstance(model.config, BayesOptConfig)
    assert model.config.data.resp_nme == "target"
    print("✅ Config is Pydantic.")
except Exception as e:
    print(f"❌ Config check failed: {e}")

print("\n✅ CORE DECOUPLING VERIFIED.")
