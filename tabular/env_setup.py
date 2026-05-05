"""
env_setup.py

Import this at the very top of every "entry point" (before numpy / autogluon / flaml imports)
to reduce hangs, segfault risk, and resource-related failures.

This module is intentionally side-effectful.
"""

# Errors to ignore
# pylint: disable=broad-exception-caught, disable=wrong-import-position

from __future__ import annotations

import os
import sys


################################################################ SECTION 1
# Thread limiting: must be set BEFORE importing numpy/scipy/numexpr/autogluon.
# These env vars are safe no-ops if the corresponding libs aren't used.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # Apple Accelerate (no-op on Linux)

# Extra safeguard to avoid runaway CPU in loky/joblib multiprocessing.
_cpu = os.cpu_count()
_loky_max = (_cpu - 2) if (_cpu is not None and _cpu > 2) else 1
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_loky_max))


################################################################ SECTION 2
# CatBoost compatibility shim:
# Some stacks (e.g. AutoGluon) may import `_catboost` directly. If catboost is installed,
# we alias the extension module so `_catboost` resolves correctly.
#
# IMPORTANT: guard this so env_setup never hard-fails if catboost is absent in a given
# environment/image.
try:
    import catboost._catboost as _cb_ext  # type: ignore

    # Tell Python that "_catboost" is already loaded; point to the real extension module.
    sys.modules.setdefault("_catboost", _cb_ext)

    # Avoid temp file collisions / on-disk artifacts (logs/snapshots).
    os.environ.setdefault("CATBOOST_ALLOW_WRITING_FILES", "False")
except Exception:
    # CatBoost not installed (or extension load failed). Skip shim safely.
    pass


################################################################ SECTION 3
# Raise the "open files" limit on Unix to reduce FLAML errors like:
# "OSError: [Errno 24] Too many open files"
if sys.platform != "win32":
    try:
        import resource  # pylint: disable=wrong-import-order, import-error

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target_soft = min(hard, 8192)
        if soft < target_soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
    except Exception:
        # Not all environments allow changing rlimits (e.g., restricted containers).
        # Failing here should not prevent the app from starting.
        pass
