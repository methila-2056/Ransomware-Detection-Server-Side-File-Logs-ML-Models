"""ASSRD — Adaptive Server-Side Ransomware Detection (self-contained prototype).

This package is a from-scratch, minimal implementation of the framework
described in the Phase I report abstract. It intentionally depends on nothing
from the main project's ``ml/`` package.

Abstract components and where they live
---------------------------------------
* creating/renaming/deleting file-operation logs server-side ....... ``data`` (emulator), ``monitor`` (live, Phase 5)
* 3 raw baseline counts vs 26 proposed temporal features .......... ``features``
* five cost-sensitive classifiers, stratified CV .................. ``models`` (Phase 2)
* cost-optimal threshold (C_FN=10, C_FP=1) ........................ ``models`` (Phase 2)
* decision-path + permutation attribution ......................... ``explain`` (Phase 4)
* PSI concept-drift monitoring + offline retraining .............. ``drift`` (Phase 4)
* feature ablation + robustness study ........................ ``ablation``, ``phase3`` (Phase 3)

The scikit-learn/xgboost compatibility workaround lives in ``compat`` and must
be imported before any sklearn estimator introspection.

Importing this package installs the compatibility shim (a no-op on healthy
environments).
"""

from assrd import compat  # noqa: F401  installs the xgboost/sklearn shim if needed

__all__ = ["compat"]
__version__ = "0.1.0"
