"""Local scientific imports and a configurable cache for experiment workers."""
from pathlib import Path
import importlib
import os
import sys

SOURCE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(os.environ.get("RESIDUAL_SIGNATURE_CACHE", str(ROOT / "runs/.cache"))).expanduser().resolve()
sys.dont_write_bytecode = True
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["NUMBA_CACHE_DIR"] = str(CACHE_ROOT / "numba")
_READY = False


def prepare_imports(install_solver=False):
    global _READY
    if not _READY:
        if str(SOURCE) not in sys.path:
            sys.path.insert(0, str(SOURCE))
        for name in ("guo_baselines", "ou_residual", "fractional_features", "audit_lasso", "safe_lars"):
            module = importlib.import_module(name)
            if Path(module.__file__).resolve().parent != SOURCE:
                raise RuntimeError(f"Import escaped packaged scientific sources: {name}")
        _READY = True
    if install_solver:
        from safe_lars import install_safe_lars
        return install_safe_lars()
    return {"source": str(SOURCE)}
