"""Package-local compatibility shim (auto-imported when this dir is on PYTHONPATH).

NumPy 2.0 removed the long-deprecated ``np.trapz`` in favour of ``np.trapezoid``
(identical algorithm and numerics). Two verbatim Milestone 11 scripts
(exp_04, exp_07) and the shared core ``quantum_gravity.py`` still call
``np.trapz``. This shim restores the alias so the unmodified source runs on
NumPy >= 2.0. It changes no numerical result.
"""
import numpy as _np
if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid
