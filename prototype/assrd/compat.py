"""scikit-learn / xgboost compatibility repair.

The problem
-----------
scikit-learn 1.6 replaced the old ``_more_tags`` mechanism with
``__sklearn_tags__`` and resolves it through the estimator's MRO.
``ClassifierMixin.__sklearn_tags__`` calls ``super().__sklearn_tags__()``.

xgboost 2.1.3 declares ``XGBClassifier`` with ``ClassifierMixin`` *after*
``BaseEstimator`` in its MRO, so ``super()`` inside ``ClassifierMixin`` resolves
to ``object``, which has no ``__sklearn_tags__``. Any sklearn call that inspects
those tags then raises::

    AttributeError: 'super' object has no attribute '__sklearn_tags__'

This surfaces through ``GridSearchCV``, which calls ``is_classifier`` on the
estimator before fitting.

Verification (see tests/test_phase1.py)
---------------------------------------
This module is kept only because a minimal reproducible test showed it is
necessary here. On the development environment (scikit-learn 1.6.1 +
xgboost 2.1.3):

* without the shim, ``is_classifier(XGBClassifier())`` raises ``AttributeError``;
* with the shim, it returns ``True`` and a full ``GridSearchCV`` fit succeeds.

The second half of the workaround
---------------------------------
joblib's default ``loky`` backend spawns fresh interpreters, so a monkeypatch
made in this process does not exist in the workers and ``GridSearchCV(n_jobs=-1)``
still fails. :func:`threading_backend` returns a joblib context manager that
forces the ``threading`` backend, whose workers share this interpreter and
therefore the patch. A test asserts both behaviours.

Scope of the patch
------------------
The patch touches only ``ClassifierMixin.__sklearn_tags__`` and
``XGBClassifier.__sklearn_tags__``. It is applied once (idempotently) and is a
no-op on environments where ``is_classifier`` already works, so it does not
affect any other estimator.
"""

import sklearn

# scikit-learn releases before 1.6 do not use __sklearn_tags__ at all, so no
# repair is possible or needed there.
_MIN_SKLEARN = (1, 6)


def _mro_indices():
    """Return (classifier_index, base_index) in XGBClassifier.__mro__, or None."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    mro = list(XGBClassifier.__mro__)
    try:
        return mro.index(sklearn.base.ClassifierMixin), mro.index(
            sklearn.base.BaseEstimator)
    except ValueError:
        return None


def needs_shim():
    """Whether this interpreter needs the repair.

    Returns ``True`` only when the MRO is inverted *and* the inversion actually
    breaks ``is_classifier``. Checking the behaviour (not just the ordering)
    avoids patching a future xgboost that happens to order the classes oddly but
    still works.
    """
    try:
        version = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
    except ValueError:
        return False
    if version < _MIN_SKLEARN:
        return False

    indices = _mro_indices()
    if indices is None:
        return False
    classifier_index, base_index = indices
    if classifier_index < base_index:
        # Ordering is already correct; still confirm behaviour in case another
        # mixin breaks the super() chain.
        try:
            sklearn.base.is_classifier(__import__("xgboost").XGBClassifier())
        except AttributeError:
            return True
        return False

    # Inverted ordering: verify it actually raises before declaring a need.
    try:
        from xgboost import XGBClassifier
        sklearn.base.is_classifier(XGBClassifier())
        return False
    except AttributeError:
        return True


def _classifier_tags(self):
    """Drop-in ``__sklearn_tags__`` for an estimator whose parent lacks one."""
    parent = super(sklearn.base.ClassifierMixin, self)
    parent_fn = getattr(parent, "__sklearn_tags__", None)

    if parent_fn is None:
        from sklearn.utils import ClassifierTags, Tags, TargetTags
        tags = Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            transformer_tags=None,
            regressor_tags=None,
            classifier_tags=None,
        )
    else:
        tags = parent_fn()

    tags.estimator_type = "classifier"
    from sklearn.utils import ClassifierTags
    tags.classifier_tags = ClassifierTags()
    tags.target_tags.required = True
    return tags


def apply():
    """Install the repair if needed. Idempotent; returns whether it applied."""
    if not needs_shim():
        return False

    from sklearn.base import ClassifierMixin
    from xgboost import XGBClassifier

    if not getattr(ClassifierMixin.__sklearn_tags__, "_assrd_shim", False):
        _classifier_tags._assrd_shim = True
        ClassifierMixin.__sklearn_tags__ = _classifier_tags

    XGBClassifier.__sklearn_tags__ = _classifier_tags
    return True


# Applied at import time. assrd/__init__.py imports this module, so any use of
# the package (including importing assrd.models) installs the shim first.
APPLIED = apply()


def threading_backend():
    """Context manager forcing joblib's threading backend.

    Required while the patch above is active: loky workers are separate
    interpreters and would not carry the patched method. Threads share this
    interpreter and therefore do. Returns a joblib parallel_backend context.
    """
    import joblib
    return joblib.parallel_backend("threading")


def diagnose():
    """Human-readable state of the compatibility layer, for reports/logs."""
    indices = _mro_indices()
    info = {
        "sklearn_version": sklearn.__version__,
        "needs_shim": needs_shim(),
        "shim_applied": bool(APPLIED),
    }
    if indices is not None:
        classifier_index, base_index = indices
        info["classifier_mixin_before_base"] = classifier_index < base_index
        info["mro_inverted"] = classifier_index > base_index
    try:
        from xgboost import XGBClassifier
        info["is_classifier_ok"] = bool(
            sklearn.base.is_classifier(XGBClassifier()))
    except Exception as exc:  # noqa: BLE001
        info["is_classifier_ok"] = False
        info["is_classifier_error"] = f"{type(exc).__name__}: {exc}"
    return info
