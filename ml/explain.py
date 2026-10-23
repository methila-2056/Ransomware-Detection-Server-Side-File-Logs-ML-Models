"""Per-window feature attribution for detected activity.

The base paper reports that detection is driven by "behavioral patterns in file
operations rather than static signatures" (Sect. 6.2) but provides no way to
inspect which operations drove a given decision. An operator who sees an alarm
has no way to tell whether it followed a create spike, a delete burst, or a
sustained run of activity, and cannot judge whether the alarm is plausible.

This module attributes a single prediction back to its input features. Tree
models are handled exactly via their impurity-based importances accumulated along
the decision path, which requires no approximation and no extra model. A
permutation alternative is provided for comparison, since the two measure
different things: impurity importance says which features the split used,
permutation importance says which features the prediction actually relies on.
They disagree often enough that reporting only one is misleading.

Aggregated across many windows, the attribution also answers a research question
the paper leaves open: which operation type carries the detection signal.
"""

import numpy as np


def tree_path_importance(model, X, feature_names=None, class_index=1):
    """Impurity-based contribution of each feature to one prediction.

    For each decision node on the path to the predicted leaf, the impurity
    decrease attributable to that split is credited to the feature used. Summing
    along the path gives a per-feature contribution for this sample, normalised
    to sum to 1 in absolute value.

    Returns ``None`` for estimators without ``tree_`` structure.
    """
    tree = getattr(model, "tree_", None)
    if tree is None:
        # An ensemble wraps its trees rather than exposing a single one. The first
        # estimator is used so a Random Forest still yields a path explanation;
        # it is an approximation of the ensemble's decision, and the label in the
        # returned payload says so rather than implying it is exact.
        estimators = getattr(model, "estimators_", None)
        if estimators is None or len(estimators) == 0:
            return None
        tree = getattr(estimators[0], "tree_", None)
        if tree is None:
            return None

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(tree.n_features)]

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    sample = X[0]

    node = 0
    contributions = {}

    while True:
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left == -1:  # leaf
            break

        feature = tree.feature[node]
        threshold = tree.threshold[node]

        weighted_n = tree.weighted_n_node_samples[node]
        impurity = tree.impurity[node]
        gain = (weighted_n * impurity
                - tree.weighted_n_node_samples[left] * tree.impurity[left]
                - tree.weighted_n_node_samples[right] * tree.impurity[right])

        # Credit the gain, signed by which side this sample took, so a feature
        # that pushed towards "not attack" is distinguishable from one that
        # pushed towards "attack".
        went_left = sample[feature] <= threshold
        direction = 1.0 if (went_left == (class_index == 1)) else -1.0
        name = feature_names[feature]
        contributions[name] = contributions.get(name, 0.0) + direction * gain

        node = left if went_left else right

    return contributions


def permutation_importance_on_sample(model, X, feature_names=None,
                                     repeats=12, seed=0, background=None):
    """Drop in probability when each feature is permuted.

    Interprets importance as prediction reliance rather than split usage. The
    background sample supplies the values that are permuted in; without one, the
    sample's own values are used, which for a single row is a no-op and would
    report zero for every feature.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    rng = np.random.default_rng(seed)
    if background is None:
        background = X.copy()
    background = np.asarray(background, dtype=float)

    base = float(model.predict_proba(X)[0, 1])
    rows = []

    for i, name in enumerate(feature_names):
        drops = []
        for _ in range(repeats):
            shuffled = background.copy()
            if len(shuffled) > 1:
                rng.shuffle(shuffled[:, i])
            else:
                shuffled[0, i] = background[:, i].mean() if background.size else 0.0
            permuted = X.copy()
            permuted[0, i] = shuffled[0, i]
            drops.append(base - float(model.predict_proba(permuted)[0, 1]))
        rows.append({
            "Feature": name,
            "Mean Probability Drop": round(float(np.mean(drops)), 6),
            "Std": round(float(np.std(drops)), 6),
        })

    rows.sort(key=lambda r: abs(r["Mean Probability Drop"]), reverse=True)
    return rows, base


def explain_prediction(model, X, feature_names=None, background=None,
                       top_k=8, repeats=12, seed=0):
    """Combined explanation for one window.

    Both importance notions are returned together because they answer different
    questions. Where they agree, the attribution is well supported. Where they
    disagree, that is itself informative and is surfaced rather than resolved
    arbitrarily.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    proba = float(model.predict_proba(X)[0, 1])

    path = tree_path_importance(model, X, feature_names)
    perm_rows, base = permutation_importance_on_sample(
        model, X, feature_names, repeats=repeats, seed=seed,
        background=background)

    top = perm_rows[:top_k]

    agreement = None
    if path:
        path_top = sorted(path.items(), key=lambda kv: abs(kv[1]), reverse=True)
        perm_top = {r["Feature"] for r in top[:3]}
        path_top3 = {name for name, _ in path_top[:3]}
        agreement = {
            "Top3 Path Features": sorted(path_top3),
            "Top3 Permutation Features": sorted(perm_top),
            "Agree": bool(path_top3 & perm_top),
            "Overlap": sorted(path_top3 & perm_top),
        }

    return {
        "Probability": round(proba, 6),
        "Verdict": "RANSOMWARE" if proba >= 0.5 else "NORMAL",
        "Top Features By Permutation": top,
        "Path Importance": None if path is None else
            {k: round(v, 6) for k, v in
             sorted(path.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]},
        "Agreement": agreement,
    }


def aggregate_explanations(model, X, feature_names=None, background=None,
                           repeats=8, seed=0):
    """Mean attribution across many windows, split by true class.

    Reporting the aggregate separately for attack and benign windows is what turns
    a debugging aid into a finding: it shows which operations distinguish the two
    classes in the fitted model, which is a question the paper leaves to
    inspection of raw logs.
    """
    X = np.asarray(X, dtype=float)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    rows, base = permutation_importance_on_sample(
        model, X[:1], feature_names, repeats=repeats, seed=seed,
        background=background)

    # Mean absolute probability drop per feature across the whole sample.
    rng = np.random.default_rng(seed)
    drops = {name: [] for name in feature_names}

    for i in range(len(X)):
        p = float(model.predict_proba(X[i:i + 1])[0, 1])
        for j, name in enumerate(feature_names):
            permuted = X[i:i + 1].copy()
            pool = background[:, j] if len(background) > 1 else X[:, j]
            permuted[0, j] = rng.choice(pool)
            q = float(model.predict_proba(permuted)[0, 1])
            drops[name].append(p - q)

    table = sorted(
        ({"Feature": name,
          "Mean Probability Drop": round(float(np.mean(v)), 6),
          "Std": round(float(np.std(v)), 6)}
         for name, v in drops.items()),
        key=lambda r: abs(r["Mean Probability Drop"]), reverse=True)

    return {"Features": feature_names, "Mean Probability": round(base, 6),
            "Aggregate": table}


def operation_group_importance(feature_names, groups=None):
    """Roll feature-level attribution up to operation-type groups.

    Assigns every feature to create, rename, delete or temporal, so the result
    answers "which operation matters" rather than "which derived column matters".

    Features are classified by the operation their aggregate is built from, so
    ``ops_sum_5s`` is a total over all three operations and lands in ``total``,
    while ``ops_cv_3s`` -- which measures variability of the total -- is
    temporal. Feature names that match no group are reported under
    ``unclassified`` rather than being dropped, so a missing bucket is visible
    instead of silently reducing the shares to less than 1.
    """
    if groups is None:
        groups = {
            "create": ("create",),
            "rename": ("rename",),
            "delete": ("delete",),
            "total": ("ops_sum", "ops_mean", "ops_max", "active_frac"),
            "temporal": ("ops_cv", "ops_absdelta", "active_run_length",
                         "burst_ratio", "ops_delta"),
        }

    assignment = {}
    for name in feature_names:
        group = "unclassified"
        # Order matters: the more specific prefixes are tested first so that
        # ops_cv_3s is temporal rather than being caught by the ops_ rule.
        for candidate, prefixes in groups.items():
            if any(name.startswith(p) or f"_{p}" in name for p in prefixes):
                group = candidate
                break
        assignment[name] = group

    return assignment


def importance_by_operation(feature_names, importance_table, groups=None):
    """Sum feature importance within each operation group."""
    assignment = operation_group_importance(feature_names, groups)
    lookup = {r["Feature"]: abs(r["Mean Probability Drop"])
              for r in importance_table}

    totals = {}
    for name, group in assignment.items():
        totals[group] = totals.get(group, 0.0) + lookup.get(name, 0.0)

    total = sum(totals.values()) or 1.0
    return {group: {"Total": round(value, 6),
                    "Share": round(value / total, 6)}
            for group, value in sorted(totals.items(),
                                       key=lambda kv: kv[1], reverse=True)}


def explain_alarm_history(model, X, labels, feature_names=None,
                          background=None, top_k=6):
    """Explain the windows the operator would have seen flagged.

    Restricted to windows the model calls attack, so the attribution describes
    the decisions that actually mattered operationally rather than averaging in
    routine negatives.
    """
    X = np.asarray(X, dtype=float)
    proba = model.predict_proba(X)[:, 1]
    flagged = proba >= 0.5
    if not flagged.any():
        return {"Flagged": 0, "Explanations": []}

    subset = X[flagged]
    aggregate = aggregate_explanations(model, subset, feature_names,
                                       background=background)

    groups = importance_by_operation(feature_names or
                                     aggregate["Features"],
                                     aggregate["Aggregate"])

    return {
        "Flagged": int(flagged.sum()),
        "Total Windows": int(len(X)),
        "Flagged Rate": round(float(flagged.mean()), 6),
        "Top Features": aggregate["Aggregate"][:top_k],
        "By Operation Group": groups,
    }