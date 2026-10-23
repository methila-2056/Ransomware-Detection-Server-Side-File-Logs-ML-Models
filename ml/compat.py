"""Compatibility shim: xgboost 2.x against scikit-learn 1.6+.

scikit-learn 1.6 replaced the ``_more_tags`` mechanism with ``__sklearn_tags__``
and walks the estimator's MRO to collect providers. ``ClassifierMixin`` implements
it as::

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        ...

xgboost 2.1.3 places ``ClassifierMixin`` *after* ``_MetadataRequester`` in
``XGBClassifier.__mro__``, so that ``super()`` resolves to ``object``, which has no
``__sklearn_tags__``. Any sklearn call that inspects tags then raises::

    AttributeError: 'super' object has no attribute '__sklearn_tags__'

This surfaces through ``GridSearchCV``, which calls ``is_classifier`` on the
estimator before fitting.

This was previously worked around by pinning ``scikit-learn<1.6`` in
``requirements.txt``. That pin is removed because it blocks the metric and
cross-validation work in P0/P1, and because xgboost 2.1.4+ and 3.x fix the MRO
ordering upstream.

The shim below restores ``ClassifierMixin.__sklearn_tags__`` so that a missing
parent is tolerated rather than fatal, and gives ``XGBClassifier`` a correct
classifier tag set. It is a no-op when the installed pair already works, so it is
safe to import unconditionally.

Import this module before any sklearn estimator introspection. ``ml.train`` and
``app.py`` both import it.
"""

import sklearn

_XGB_MIN_SKLEARN = (1, 6)


def _needs_shim():
    try:
        version = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
    except ValueError:
        return False
    if version < _XGB_MIN_SKLEARN:
        return False

    try:
        from xgboost import XGBClassifier
    except ImportError:
        return False

    # Correct sklearn 1.6 ordering is ClassifierMixin before BaseEstimator
    # (e.g. class RF(ClassifierMixin, BaseEstimator)). xgboost 2.1.3 violates it,
    # which is exactly what makes super() resolve to object.
    mro = XGBClassifier.__mro__
    try:
        classifier_index = mro.index(sklearn.base.ClassifierMixin)
        base_index = mro.index(sklearn.base.BaseEstimator)
    except ValueError:
        return False

    if classifier_index < base_index:
        try:
            sklearn.base.is_classifier(XGBClassifier())
        except AttributeError:
            return True
        return False

    return classifier_index > base_index


def apply():
    """Install the tag shim if this environment needs it. Safe to call repeatedly."""
    if not _needs_shim():
        return False

    from sklearn.base import ClassifierMixin
    from sklearn.utils import ClassifierTags, Tags, TargetTags
    from xgboost import XGBClassifier

    def _classifier_tags(self):
        parent = super(ClassifierMixin, self)
        parent_fn = getattr(parent, "__sklearn_tags__", None)

        if parent_fn is None:
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
        tags.classifier_tags = ClassifierTags()
        tags.target_tags.required = True
        return tags

    if not getattr(ClassifierMixin.__sklearn_tags__, "_xgb_shim", False):
        _classifier_tags._xgb_shim = True
        ClassifierMixin.__sklearn_tags__ = _classifier_tags

    XGBClassifier.__sklearn_tags__ = _classifier_tags
    return True


APPLIED = apply()


def thread_backend():
    """Context manager forcing joblib's threading backend.

    Required while the monkeypatch above is active. joblib's default loky backend
    spawns fresh interpreters, so the patched ``__sklearn_tags__`` would not exist
    there and ``GridSearchCV(n_jobs=-1)`` would still raise. Threads share the
    parent interpreter, so the patch applies to every worker.

    The dataset is a few thousand rows of three integers, where threading gives the
    same speedup as processes and avoids the serialisation overhead entirely.
    """
    import joblib

    return joblib.parallel_backend("threading")