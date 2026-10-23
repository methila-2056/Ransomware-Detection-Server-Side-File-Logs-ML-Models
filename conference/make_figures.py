"""Generate IEEE-style figures for the conference paper from real project artifacts.
Outputs PNG files in conference/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "data"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
})

with open(DATA / "model_results.json") as f:
    results = json.load(f)

df = pd.read_csv(DATA / "training_dataset.csv")

MODELS = [r["Model"] for r in results]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


# ---- Fig 1: architecture ----
def fig_architecture():
    fig, ax = plt.subplots(figsize=(3.4, 4.6))
    ax.axis("off")
    boxes = [
        ("Monitored directory\n(demo_server/)\ncreate / rename / delete", 0.50, 0.94),
        ("watchdog observer\n(on_created, on_deleted, on_moved)", 0.50, 0.79),
        ("event buffer (deque)\nthread-safe, timestamped", 0.50, 0.64),
        ("1-second window aggregation\n(cc, rc, dc)", 0.50, 0.49),
        ("Classifier\nbest_model.pkl", 0.50, 0.34),
        ("SQLite detections\n+ live prediction", 0.50, 0.19),
        ("Flask dashboard + REST API\n(live alerts, history)", 0.50, 0.05),
    ]
    w, h = 0.86, 0.10
    for text, x, y in boxes:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.01", linewidth=0.8,
                                    edgecolor="black", facecolor="#EEEEEE"))
        ax.text(x, y, text, ha="center", va="center", fontsize=7)
    for i in range(len(boxes) - 1):
        y0 = boxes[i][2] - h / 2
        y1 = boxes[i + 1][2] + h / 2
        ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1),
                                     arrowstyle="-|>", mutation_scale=9,
                                     linewidth=0.8, color="black"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_architecture.png", bbox_inches="tight")
    plt.close(fig)


# ---- Fig 3: metrics ----
def fig_metrics():
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(MODELS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, m in enumerate(metrics):
        vals = [r[m] for r in results]
        ax.bar(x + (i - 1.5) * width, vals, width, label=m,
               color=COLORS[i], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(["DT", "RF", "SVM", "Ada", "XGB"], fontsize=8)
    ax.set_ylim(0.9, 1.01)
    ax.set_ylabel("Score (test N = 80)")
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_metrics.png", bbox_inches="tight")
    plt.close(fig)


# ---- Fig 4: total cost ----
def fig_cost():
    costs = [r["Total Cost"] for r in results]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    bars = ax.bar(["DT", "RF", "SVM", "Ada", "XGB"], costs,
                  color=["#55A868" if c == 0 else "#C44E52" for c in costs],
                  edgecolor="black", linewidth=0.3)
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.2, str(c),
                ha="center", fontsize=7)
    ax.set_ylabel("Total cost  (1·FP + 10·FN)")
    ax.set_ylim(0, max(costs) + 2)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_cost.png", bbox_inches="tight")
    plt.close(fig)


# ---- Fig 5: feature separability ----
def fig_separability():
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    normal = df[df.label == 0]
    ransom = df[df.label == 1]
    ax.scatter(normal.create_count, normal.delete_count, s=8, alpha=0.6,
               c="#4C72B0", label="Normal", edgecolors="none")
    ax.scatter(ransom.create_count, ransom.delete_count, s=8, alpha=0.6,
               c="#C44E52", label="Ransomware", edgecolors="none")
    ax.set_xlabel("Create count / window")
    ax.set_ylabel("Delete count / window")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_separability.png", bbox_inches="tight")
    plt.close(fig)


# ---- Fig 6: ROC ----
def fig_roc():
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for i, r in enumerate(results):
        rd = r["ROC Data"]
        ax.plot(rd["fpr"], rd["tpr"], color=COLORS[i], linewidth=0.9,
                label=f'{r["Model"]} (AUC={rd["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_roc.png", bbox_inches="tight")
    plt.close(fig)


def fig_equations():
    e1 = r"$\mathbf{x}_i = (c_i,\ r_i,\ d_i)$"
    e2 = r"$\mathrm{Cost} = C_{FP}\!\cdot\! FP + C_{FN}\!\cdot\! FN = FP + 10\!\cdot\! FN$"
    for name, tex in [("eq1.png", e1), ("eq2.png", e2)]:
        fig = plt.figure(figsize=(3.2, 0.45))
        fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=11)
        fig.savefig(FIG / name, bbox_inches="tight", transparent=True)
        plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_metrics()
    fig_cost()
    fig_separability()
    fig_roc()
    fig_equations()
    print("Figures written to", FIG)
    for p in sorted(FIG.glob("*.png")):
        print(" -", p.name, p.stat().st_size, "bytes")
