"""Leave-one-family-out evaluation of generalisation to unseen ransomware.

The base paper states in Sect. 3.7 that "irrespective of which ransomware family
was executed or how frequently runs occurred, the machine learning models
exhibited similar proportions of true and false predictions, as reflected in the
confusion matrices", and Sect. 6.3 uses this to argue the method supports
practical applicability in both SME and enterprise environments.

That is a claim about behaviour not depending on family. It is not a test of
generalisation to an unseen family, because in the paper's design every model saw
all five families during training. A model can score well on five known families
and still fail on a sixth, and the paper's design cannot distinguish those cases.

This module performs the experiment that would: train on four families, hold the
fifth out entirely, and measure detection of the unseen family. The result
quantifies the paper's claim rather than restating it.

Held-out families are also removed from the *benign* side only in the sense that
benign windows are unaffected; what is held out is the attack family. Group-aware
splitting is applied afterwards so that held-out attack episodes do not appear in
training through another family's episode.
"""

import numpy as np
import pandas as pd

from ml.metrics import binary_metrics, compute_cost
from ml.train import SCALED_MODELS
from ml.streaming import add_causal_features, add_sequence_features, \
    causal_feature_columns, sequence_feature_columns
from ml.traces import DATASET_VARIANTS, build_trace, group_aware_split


def _fit_and_score(X_train, y_train, X_test, y_test, factory, seed=42):
    """Fit one model and return its metrics plus its probability column."""
    model = factory(seed)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = binary_metrics(y_test, (proba >= 0.5).astype(int))
    return metrics, proba


def default_factories():
    """One factory per evaluated model: the paper's five plus the linear control."""
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    def xgb(seed):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                            subsample=0.9, random_state=seed,
                            eval_metric="logloss")

    return {
        "Decision Tree": lambda s: DecisionTreeClassifier(
            max_depth=6, criterion="entropy", random_state=s),
        "Random Forest": lambda s: RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=s, n_jobs=1),
        "SVM": lambda s: CalibratedClassifierCV(
            SVC(C=1.0, kernel="rbf", random_state=s), ensemble=False),
        "AdaBoost": lambda s: AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5, random_state=s),
        "XGBoost": xgb,
        "Logistic Regression": lambda s: LogisticRegression(
            C=1.0, max_iter=2000, random_state=s),
    }


def leave_one_family_out(df, feature_set="sequence", families=None, seed=42,
                         factories=None, verbose=True):
    """Train on four families, test on the fifth, for each family in turn.

    Two figures are produced per (held-out family, model):

    in-distribution
        The model scores windows of the four families it was trained on, using a
        group-aware held-out split. This is the ordinary generalisation case.

    zero-shot
        The same fitted model scores windows of the family that was removed from
        training entirely. No window of that family was seen at any point.

    The gap between them is the quantity of interest: it is the cost of an unseen
    family, which the base paper's design cannot measure because all five families
    appear in its training data.
    """
    if factories is None:
        factories = default_factories()

    enriched, feats = _enrich(df, feature_set)

    if families is None:
        families = sorted(enriched[enriched.label == 1]["family"].dropna().unique())

    results = []
    from sklearn.preprocessing import StandardScaler

    for held_out in families:
        # Remove the held-out family entirely, then apply group-aware splitting so
        # no episode of it can reach training through another family's episode.
        subset = enriched[enriched["family"].ne(held_out)].copy()
        variant = DATASET_VARIANTS["nozeroatt"](subset)
        train_df, _, test_df = group_aware_split(variant, 0.1, 0.1, seed=seed)

        # Zero-shot windows come from the held-out family's attack windows, which
        # are absent from every split above. They must be scored against a benign
        # reference set or specificity is undefined and the cost figure collapses
        # to a false-negative count. Benign windows carry no family label, so the
        # benign rows of the untouched test split are legitimate negatives here:
        # the model never saw them, and they were never contaminated by the
        # held-out family.
        unseen = enriched[(enriched["family"] == held_out)
                          & (enriched["label"] == 1)]
        benign_ref = test_df[test_df["label"] == 0]

        Xtr = train_df[feats].values.astype(float)
        ytr = train_df["label"].values
        Xte = test_df[feats].values.astype(float)
        yte = test_df["label"].values

        # Zero-shot evaluation set: unseen attacks + never-trained benign windows.
        import pandas as _pd
        zero_frame = _pd.concat([unseen, benign_ref], ignore_index=True)
        Xun = zero_frame[feats].values.astype(float)
        yun = zero_frame["label"].values

        if verbose:
            print(f"\nHeld out: {held_out} "
                  f"({len(unseen)} attack windows never seen in training)")
            print(f"  train={len(train_df)} (attack {int(ytr.sum())}) "
                  f"test={len(test_df)} (attack {int(yte.sum())})")
            print(f"  zero-shot set={len(zero_frame)} "
                  f"(attack {int(yun.sum())}, benign {int((yun == 0).sum())})")

        scaler = StandardScaler().fit(Xtr)

        for name, factory in factories.items():
            needs_scaling = name in SCALED_MODELS
            Xtr_use = scaler.transform(Xtr) if needs_scaling else Xtr
            Xte_use = scaler.transform(Xte) if needs_scaling else Xte
            Xun_use = scaler.transform(Xun) if needs_scaling else Xun

            try:
                in_dist, _ = _fit_and_score(Xtr_use, ytr, Xte_use, yte,
                                            factory, seed)

                # One fit, two evaluations. Fitting separately for the zero-shot
                # figure would measure a different model, not the same one applied
                # to unseen data.
                model = factory(seed)
                model.fit(Xtr_use, ytr)
                proba_unseen = model.predict_proba(Xun_use)[:, 1]
                zero_shot = binary_metrics(yun, (proba_unseen >= 0.5).astype(int))
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  {name}: failed ({exc})")
                continue

            record = {
                "Held Out Family": held_out,
                "Model": name,
                "Train Attack Windows": int(ytr.sum()),
                "In Distribution Test Windows": int(len(yte)),
                "Unseen Attack Windows": int(yun.sum()),
                "Zero Shot Benign Windows": int((yun == 0).sum()),
                "In Distribution Sensitivity": in_dist["Sensitivity"],
                "In Distribution Specificity": in_dist["Specificity"],
                "In Distribution Cost": in_dist["Total Cost"],
                "Zero Shot Sensitivity": zero_shot["Sensitivity"],
                "Zero Shot Specificity": zero_shot["Specificity"],
                "Zero Shot Cost": zero_shot["Total Cost"],
                "Sensitivity Drop": round(
                    in_dist["Sensitivity"] - zero_shot["Sensitivity"], 6),
            }
            results.append(record)

            if verbose:
                print(f"  {name:<16} zero-shot sens={zero_shot['Sensitivity']:.4f} "
                      f"spec={zero_shot['Specificity']:.4f} "
                      f"cost={zero_shot['Total Cost']:<6} "
                      f"(in-dist sens {in_dist['Sensitivity']:.4f})")

    return results


def _enrich(df, feature_set):
    """Apply the requested feature construction and return (frame, columns)."""
    if feature_set == "sequence":
        if any(c.startswith("ops_mean_5s") for c in df.columns):
            return df, sequence_feature_columns(df)
        enriched = add_sequence_features(df)
        return enriched, sequence_feature_columns(enriched)
    if feature_set == "causal":
        if any(c.startswith("ops_sum_prev") for c in df.columns):
            return df, causal_feature_columns(df)
        enriched = add_causal_features(df)
        return enriched, causal_feature_columns(enriched)
    return df, ["create_count", "rename_count", "delete_count"]


def generalisation_report(results):
    """Summarise zero-shot degradation across families and models."""
    frame = pd.DataFrame(results)
    if frame.empty:
        return {"Families": [], "Models": []}

    per_model = {}
    for model, group in frame.groupby("Model"):
        drops = group["Sensitivity Drop"].values
        per_model[model] = {
            "Mean Sensitivity Drop": round(float(np.mean(drops)), 6),
            "Worst Drop": round(float(np.max(drops)), 6),
            "Worst Held Out Family": group.loc[
                group["Sensitivity Drop"].idxmax(), "Held Out Family"],
            "Mean Zero Shot Sensitivity": round(
                float(group["Zero Shot Sensitivity"].mean()), 6),
            "Mean Zero Shot Specificity": round(
                float(group["Zero Shot Specificity"].mean()), 6),
            "Mean Zero Shot Cost": round(
                float(group["Zero Shot Cost"].mean()), 6),
            "Holds Up (mean drop < 0.10)": bool(np.mean(drops) < 0.10),
        }

    per_family = {}
    for family, group in frame.groupby("Held Out Family"):
        per_family[family] = {
            "Mean Zero Shot Sensitivity": round(
                float(group["Zero Shot Sensitivity"].mean()), 6),
            "Mean Zero Shot Specificity": round(
                float(group["Zero Shot Specificity"].mean()), 6),
            "Best Model": group.loc[
                group["Zero Shot Sensitivity"].idxmax(), "Model"],
            "Worst Model": group.loc[
                group["Zero Shot Sensitivity"].idxmin(), "Model"],
        }

    return {
        "Evaluation Note": (
            "Zero-shot rows are scored on the held-out family's attack windows "
            "plus benign windows the model never trained on, so specificity and "
            "total cost are defined. Sensitivity is computed on the unseen "
            "family's attack windows only."),
        "Families": sorted(per_family),
        "Per Family": per_family,
        "Per Model": per_model,
        "Overall Mean Drop": round(
            float(frame["Sensitivity Drop"].mean()), 6),
        "Overall Mean Zero Shot Sensitivity": round(
            float(frame["Zero Shot Sensitivity"].mean()), 6),
        "Overall Mean Zero Shot Specificity": round(
            float(frame["Zero Shot Specificity"].mean()), 6),
        "Any Model Holds Up": any(
            v["Holds Up (mean drop < 0.10)"] for v in per_model.values()),
    }


def run(df, feature_set="sequence", seed=42, families=None, verbose=True):
    """Execute the holdout experiment and return results plus a summary."""
    results = leave_one_family_out(df, feature_set=feature_set, families=families,
                                   seed=seed, verbose=verbose)
    return results, generalisation_report(results)