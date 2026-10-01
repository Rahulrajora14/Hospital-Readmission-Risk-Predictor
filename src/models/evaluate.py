"""
Evaluation helpers: metrics + plots.

Why not just accuracy?
Only ~9% of patients are readmitted. A model that says "nobody will be
readmitted" gets ~91% accuracy and is completely useless. So we use:

- ROC-AUC  : how well the model RANKS risky patients above safe ones (0.5 = random)
- PR-AUC   : same idea but focused on the rare positive class (baseline = 0.09)
- Recall   : of all patients who were readmitted, how many did we catch?
- Precision: of all patients we flagged, how many were really readmitted?
- F1       : balance of precision and recall
- Top-10% lift: if a care team can only call the 10% riskiest patients,
               how much more readmission-dense is that list than random?
               (the most "business" metric here)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config

COLORS = {"Logistic Regression": "#7f7f7f", "Random Forest": "#4f6d8f", "XGBoost": "#c0504d"}


def find_best_threshold(y_true, y_proba) -> float:
    """
    Models output a probability. We must pick a cut-off to say "high risk".
    0.5 is a bad default when positives are rare, so we pick the threshold
    that maximises F1 on the VALIDATION set (never on the test set).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = np.argmax(f1[:-1])  # last point has no threshold
    return float(thresholds[best])


def top_k_lift(y_true, y_proba, k: float = 0.10) -> float:
    """Readmission rate among the top-k% riskiest patients / overall rate."""
    y_true = np.asarray(y_true)
    n_top = max(1, int(len(y_true) * k))
    top_idx = np.argsort(y_proba)[::-1][:n_top]
    return float(y_true[top_idx].mean() / y_true.mean())


def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "top10_lift": round(top_k_lift(y_true, y_proba, 0.10), 2),
        "threshold": round(threshold, 4),
    }


def plot_roc_curves(y_true, probas: dict, filename="roc_curves.png"):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, proba):.3f})",
                color=COLORS.get(name), lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title="ROC curves on the test set")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130)
    plt.close(fig)


def plot_pr_curves(y_true, probas: dict, filename="pr_curves.png"):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, proba in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ax.plot(rec, prec, label=f"{name} (AP = {average_precision_score(y_true, proba):.3f})",
                color=COLORS.get(name), lw=2)
    ax.axhline(np.mean(y_true), color="k", ls="--", lw=1, label="Baseline (positive rate)")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves on the test set")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_proba, threshold, model_name, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, (y_proba >= threshold).astype(int))
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["Predicted: not readmitted", "Predicted: readmitted"])
    ax.set_yticks([0, 1], ["Actual:\nnot readmitted", "Actual:\nreadmitted"])
    ax.set_title(f"{model_name} - confusion matrix (threshold {threshold:.2f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130)
    plt.close(fig)
