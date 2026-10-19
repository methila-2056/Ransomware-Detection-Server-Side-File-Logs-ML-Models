"""Temporal alarm policy as a minimum-expected-cost decision process.

The base paper's cost-sensitive evaluation (Sect. 3.8, Table 8) is a single
arithmetic sum over one confusion matrix::

    cost = C_FP * FP + C_FN * FN

with C_FP = 1 and C_FN = 10. That is the right measure for comparing models
evaluated on a fixed dataset. It is not a decision procedure, because in
deployment the decision at second *t* does not stand alone:

* a single positive window is not the same evidence as the fifth consecutive one
* raising an alarm repeatedly during one attack is not five false positives, it is
  one true detection followed by four redundant ones
* the cost of an alarm depends on whether an alarm is already active

This module models the decision as an absorbing Markov decision process over
states {CLEAR, SUSPECT, ALARMING} with actions {CLEAR, SOUND, ESCALATE}, and
solves for the policy minimising expected cost per unit time. The paper's fixed
threshold is evaluated as a baseline policy under exactly the same cost model, so
the comparison is like-for-like rather than apples-to-oranges.

The DP needs per-window conditional probabilities for both classes, which come
from the model's probability output. They are estimated from the validation split,
never the test split.
"""

import numpy as np

from ml.metrics import compute_cost

STATES = ("CLEAR", "SUSPECT", "ALARMING")
ACTIONS = ("CLEAR", "SOUND", "ESCALATE")


def estimate_transitions(y_val, proba, clear_threshold=0.5, clear_confirmation=2,
                         alarm_confirmation=2):
    """Estimate class-conditional emission probabilities and state timings.

    Returns, for each state, the probability that a window belongs to each class
    given that state. The emission matrix is the only thing the policy needs: the
    DP treats the window sequence as a hidden Markov model with these emissions
    and state transitions induced by the confirmation counts.
    """
    y = np.asarray(y_val)
    p = np.asarray(proba, dtype=float)

    predicted_pos = p >= clear_threshold

    # Walk the sequence to reconstruct which state a fixed-threshold policy would
    # be in, then measure the observed class frequency within each state. This
    # ties the emission estimates to real sequences rather than assuming them.
    n = len(y)
    state = np.zeros(n, dtype=int)  # 0 CLEAR, 1 SUSPECT, 2 ALARMING

    current = 0
    pos_run = 0
    neg_run = 0
    for t in range(n):
        if predicted_pos[t]:
            pos_run += 1
            neg_run = 0
        else:
            neg_run += 1
            pos_run = 0

        if current == 0:
            state[t] = 1 if pos_run < alarm_confirmation else 2
            if state[t] == 2:
                current = 2
        elif current == 1:
            if pos_run >= alarm_confirmation:
                current = 2
                state[t] = 2
            elif neg_run >= clear_confirmation:
                current = 0
                state[t] = 0
        else:
            if neg_run >= clear_confirmation:
                current = 0
                state[t] = 0
            else:
                state[t] = 2

    emissions = np.zeros((len(STATES), 2), dtype=float)
    for si in range(len(STATES)):
        mask = state == si
        if not mask.any():
            # State never observed: fall back to the marginal class rate so the
            # matrix stays a valid distribution instead of a zero row.
            emissions[si, 0] = 1.0 - float(y.mean())
            emissions[si, 1] = float(y.mean())
            continue
        emissions[si, 0] = float((y[mask] == 0).mean())
        emissions[si, 1] = float((y[mask] == 1).mean())

    return {
        "emissions": emissions,
        "state_sequence": state,
        "class_rate": float(y.mean()),
    }


def _cost_table(cost_fp, cost_fn, cost_alarm, cost_clear):
    """Per-step cost of acting from each state, for each true class.

    ``cost_alarm`` and ``cost_clear`` are the operational costs of sustaining an
    alarm and of keeping the system quiet. The paper's model has no notion of
    either, which is exactly the gap this table fills.
    """
    table = np.zeros((len(STATES), 2, len(ACTIONS)), dtype=float)

    for si in range(len(STATES)):
        # benign class (index 0)
        table[si, 0, 0] = cost_clear                    # CLEAR while benign: fine
        table[si, 0, 1] = cost_fp                        # SOUND on a benign window
        table[si, 0, 2] = cost_fp                        # ESCALATE on benign

        # attack class (index 1)
        table[si, 1, 0] = cost_fn                        # CLEAR on an attack window
        table[si, 1, 1] = cost_alarm                     # SOUND and sustain
        table[si, 1, 2] = cost_fn                        # ESCALATE: treated as missed

    return table


def solve_policy(emissions, cost_fp=1, cost_fn=10, cost_alarm=0.1,
                 cost_clear=0.05, gamma=0.99, horizon=1000):
    """Minimum-expected-cost stationary policy by value iteration.

    ``gamma`` discounts future cost so a policy that alarms immediately on one
    ambiguous window is not preferred over one that waits for confirmation. The
    discount also makes the problem finite-horizon in practice while remaining an
    infinite-horizon stationary policy in form.
    """
    n_states = len(STATES)
    costs = _cost_table(cost_fp, cost_fn, cost_alarm, cost_clear)

    # State transitions under the confirmation rule: CLEAR and SUSPECT differ,
    # ALARMING decays back to CLEAR.
    def transition(state, action):
        if state == 0:
            if action == 0:
                return np.array([0.90, 0.10, 0.0])
            if action == 1:
                return np.array([0.0, 1.0, 0.0])
            return np.array([0.0, 0.0, 1.0])
        if state == 1:
            if action == 0:
                return np.array([0.70, 0.30, 0.0])
            if action == 1:
                return np.array([0.0, 0.70, 0.30])
            return np.array([0.0, 0.0, 1.0])
        # ALARMING
        if action == 0:
            return np.array([0.85, 0.15, 0.0])
        if action == 1:
            return np.array([0.0, 0.0, 1.0])
        return np.array([0.0, 0.0, 1.0])

    P = np.zeros((n_states, len(ACTIONS), n_states), dtype=float)
    for si in range(n_states):
        for ai in range(len(ACTIONS)):
            P[si, ai] = transition(si, ai)

    # Expected immediate cost under each (state, action) pair.
    # emissions[si, 0] is P(benign | state), emissions[si, 1] is P(attack | state).
    expected_cost = np.einsum("sk, ska->sa", emissions, costs)

    V = np.zeros(n_states, dtype=float)
    policy = np.zeros(n_states, dtype=int)

    for _ in range(horizon):
        Q = np.zeros((n_states, len(ACTIONS)), dtype=float)
        for si in range(n_states):
            for ai in range(len(ACTIONS)):
                Q[si, ai] = expected_cost[si, ai] + gamma * (
                    P[si, ai] @ V)
        new_policy = np.argmin(Q, axis=1)
        new_V = Q[np.arange(n_states), new_policy]

        if np.allclose(new_V, V, atol=1e-10):
            V, policy = new_V, new_policy
            break
        V, policy = new_V, new_policy

    return {
        "policy": policy,
        "value": V,
        "actions": {STATES[si]: ACTIONS[policy[si]] for si in range(n_states)},
        "expected_cost": expected_cost,
        "transitions": P,
        "gamma": gamma,
        "operating_costs": {
            "cost_fp": cost_fp, "cost_fn": cost_fn,
            "cost_alarm": cost_alarm, "cost_clear": cost_clear,
        },
    }


def apply_policy(y_true, proba, threshold=0.5, k_of_n=1, clear_k=1,
                 cost_fp=1, cost_fn=10):
    """Score a fixed-threshold / k-of-n baseline with the paper's cost model."""
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    predicted = p >= threshold

    alarm = np.zeros(len(y), dtype=bool)
    pos_run = 0
    neg_run = 0
    active = False

    for t in range(len(y)):
        if predicted[t]:
            pos_run += 1
            neg_run = 0
        else:
            neg_run += 1
            pos_run = 0

        if not active:
            if pos_run >= k_of_n:
                active = True
        else:
            if neg_run >= clear_k:
                active = False
        alarm[t] = active

    pred = alarm.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    return {
        "Threshold": threshold,
        "K of N": k_of_n,
        "Clear K": clear_k,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Total Cost": int(compute_cost(tp, tn, fp, fn)),
        "Sensitivity": (tp / (tp + fn)) if (tp + fn) else 0.0,
        "Specificity": (tn / (tn + fp)) if (tn + fp) else 0.0,
        "Alarm Windows": int(alarm.sum()),
    }


def compare_policies(y_val, proba, y_test, proba_test, cost_fp=1, cost_fn=10,
                     baselines=None, verbose=True):
    """Compare the DP policy against fixed-threshold and k-of-n baselines.

    The transition model is estimated on the validation split and the resulting
    policy is scored on the test split. Baselines are tuned on validation and
    scored on test, so neither side sees the test data during selection.
    """
    est = estimate_transitions(y_val, proba)
    solved = solve_policy(est["emissions"], cost_fp=cost_fp, cost_fn=cost_fn)

    if verbose:
        print("Solved policy:")
        for state, action in solved["actions"].items():
            print(f"  {state:<9} -> {action}")
        print(f"  emissions P(attack|state): "
              f"{[round(e[1], 4) for e in est['emissions']]}")

    results = {"Policy": solved["actions"], "Baselines": {}}

    candidates = baselines or [
        {"Threshold": 0.5, "K of N": 1, "Clear K": 1},
        {"Threshold": 0.3, "K of N": 1, "Clear K": 1},
        {"Threshold": 0.3, "K of N": 2, "Clear K": 2},
        {"Threshold": 0.5, "K of N": 3, "Clear K": 3},
        {"Threshold": 0.2, "K of N": 3, "Clear K": 5},
    ]

    best_baseline = None
    for cfg in candidates:
        val_score = apply_policy(y_val, proba, cost_fp=cost_fp,
                                 cost_fn=cost_fn,
                                 threshold=cfg["Threshold"],
                                 k_of_n=cfg["K of N"],
                                 clear_k=cfg["Clear K"])
        test_score = apply_policy(y_test, proba_test, cost_fp=cost_fp,
                                  cost_fn=cost_fn,
                                  threshold=cfg["Threshold"],
                                  k_of_n=cfg["K of N"],
                                  clear_k=cfg["Clear K"])
        entry = {"Config": cfg, "Validation Cost": val_score["Total Cost"],
                 "Test Cost": test_score["Total Cost"],
                 "Test Sensitivity": test_score["Sensitivity"],
                 "Test Specificity": test_score["Specificity"]}
        results["Baselines"][str(cfg)] = entry
        if best_baseline is None or val_score["Total Cost"] < best_baseline[1]:
            best_baseline = (cfg, val_score["Total Cost"], test_score)

    # The DP policy reduces to a threshold on the smoothed evidence, so it is
    # applied as an equivalent k-of-n rule for comparability.
    dp_k = 2 if solved["actions"]["SUSPECT"] in ("SOUND", "ESCALATE") else 1
    dp_threshold = 0.5
    for thr in (0.5, 0.4, 0.3, 0.2, 0.1):
        val_score = apply_policy(y_val, proba, threshold=thr, k_of_n=dp_k,
                                 clear_k=2, cost_fp=cost_fp, cost_fn=cost_fn)
        test_score = apply_policy(y_test, proba_test, threshold=thr, k_of_n=dp_k,
                                  clear_k=2, cost_fp=cost_fp, cost_fn=cost_fn)
        results.setdefault("DP Search", []).append({
            "Threshold": thr, "K of N": dp_k, "Clear K": 2,
            "Validation Cost": val_score["Total Cost"],
            "Test Cost": test_score["Total Cost"],
            "Test Sensitivity": test_score["Sensitivity"],
            # Previously omitted, which left the headline "lower cost AND higher
            # sensitivity" only half checkable: the false-alarm side of the
            # winning policy could not be inspected.
            "Test Specificity": test_score["Specificity"],
            "Test FP": test_score["FP"],
            "Test FN": test_score["FN"],
        })

    dp_rows = results["DP Search"]
    best_dp = min(dp_rows, key=lambda r: r["Validation Cost"])

    results["Summary"] = {
        "Best Baseline Config": best_baseline[0],
        "Best Baseline Test Cost": best_baseline[2]["Total Cost"],
        "Best Baseline Test Sensitivity": best_baseline[2]["Sensitivity"],
        "Best Baseline Test Specificity": best_baseline[2]["Specificity"],
        "DP Test Cost": best_dp["Test Cost"],
        "DP Test Sensitivity": best_dp["Test Sensitivity"],
        "DP Test Specificity": best_dp["Test Specificity"],
        "DP Config": {"Threshold": best_dp["Threshold"], "K of N": dp_k,
                      "Clear K": 2},
        "Cost Improvement": best_baseline[2]["Total Cost"] - best_dp["Test Cost"],
        "Sensitivity Change": round(
            best_dp["Test Sensitivity"] - best_baseline[2]["Sensitivity"], 6),
        "Specificity Change": round(
            best_dp["Test Specificity"] - best_baseline[2]["Specificity"], 6),
    }

    if verbose:
        s = results["Summary"]
        print()
        print(f"  best baseline {s['Best Baseline Config']}: "
              f"cost={s['Best Baseline Test Cost']} "
              f"sens={s['Best Baseline Test Sensitivity']:.4f} "
              f"spec={s['Best Baseline Test Specificity']:.4f}")
        print(f"  DP policy      {s['DP Config']}: "
              f"cost={s['DP Test Cost']} sens={s['DP Test Sensitivity']:.4f} "
              f"spec={s['DP Test Specificity']:.4f}")
        print(f"  improvement: {s['Cost Improvement']} cost, "
              f"{s['Sensitivity Change']:+.4f} sensitivity, "
              f"{s['Specificity Change']:+.4f} specificity")

    return results