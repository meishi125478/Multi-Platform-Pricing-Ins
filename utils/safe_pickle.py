from __future__ import annotations

import pickle
from typing import BinaryIO, Dict, Optional, Set, Tuple


_SAFE_BUILTINS: Set[str] = {
    "bool",
    "bytes",
    "complex",
    "dict",
    "float",
    "frozenset",
    "int",
    "list",
    "set",
    "slice",
    "str",
    "tuple",
}

_SAFE_EXACT_GLOBALS: Dict[str, Set[str]] = {
    "collections": {"OrderedDict", "defaultdict"},
    "copyreg": {"_reconstructor"},
    "copy_reg": {"_reconstructor"},
    "numpy": {"dtype", "ndarray"},
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
    "numpy.core.numeric": {"_frombuffer"},
    "numpy.ma.core": {"_mareconstruct", "MaskedArray"},
    "pandas.core.frame": {"DataFrame"},
    "pandas.core.series": {"Series"},
    "pandas.core.indexes.base": {"Index", "_new_Index"},
    "pandas.core.indexes.range": {"RangeIndex"},
    "pandas.core.indexes.multi": {"MultiIndex"},
    "pandas._libs.internals": {"_unpickle_block"},
    "pandas.core.internals.blocks": {"new_block", "new_block_2d"},
    "pandas.core.internals.managers": {"BlockManager", "SingleBlockManager"},
    "sklearn.preprocessing._data": {"StandardScaler"},
    "sklearn.preprocessing._encoders": {"OneHotEncoder"},
}

_SAFE_PREFIXES: Tuple[str, ...] = (
    "numpy.",
    "pandas.",
    "sklearn.",
    "xgboost.",
    "joblib.numpy_pickle.",
    "joblib._memmapping_reducer.",
    "scipy.sparse.",
)


class RestrictedUnpickler(pickle.Unpickler):
    """Best-effort restricted unpickler for trusted ML artifact payloads."""

    def find_class(self, module: str, name: str):
        if module == "builtins":
            if name in _SAFE_BUILTINS:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"Blocked builtins symbol: {name}")

        allowed_names = _SAFE_EXACT_GLOBALS.get(module)
        if allowed_names is not None:
            if name in allowed_names:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(
                f"Blocked symbol outside allowlist: {module}.{name}"
            )

        if module.startswith(_SAFE_PREFIXES):
            return super().find_class(module, name)

        raise pickle.UnpicklingError(
            f"Blocked unpickling of {module}.{name}; module not allowlisted."
        )


def restricted_pickle_load(
    fh: BinaryIO,
    *,
    unpickler_cls: Optional[type[RestrictedUnpickler]] = None,
):
    cls = unpickler_cls or RestrictedUnpickler
    return cls(fh).load()


__all__ = ["RestrictedUnpickler", "restricted_pickle_load"]
