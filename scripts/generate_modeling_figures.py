"""Generate all predictive-modeling figures and save to figures/.

Run from repo root:
    python scripts/generate_modeling_figures.py

Expected runtime: 20-35 min (RandomForest + XGBoost on 500K rows).
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.plot_style import (
    ACCENT, BIN_PALETTE, LOSS_COLOR, PROFIT_COLOR, SAVEFIG_KW, apply_style,
)
from src.modeling import save_modeling_outputs

FIGURES   = Path("figures")
PROCESSED = Path("data/processed")


# ---------------------------------------------------------------------------
# Fig 1 — ROC curves (all 4 models)
# ---------------------------------------------------------------------------

def fig_roc_comparison(probas: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = plt.cm.tab10.colors

    for i, (name, y_prob) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=palette[i], linewidth=2,
                label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Entry-time Features Only")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_roc_comparison.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_roc_comparison.png")


# ---------------------------------------------------------------------------
# Fig 2 — Precision-Recall curves
# ---------------------------------------------------------------------------

def fig_pr_comparison(probas: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = plt.cm.tab10.colors
    base_rate = float(y_test.mean())

    for i, (name, y_prob) in enumerate(probas.items()):
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, color=palette[i], linewidth=2,
                label=f"{name}  (AP={ap:.3f})")

    ax.axhline(base_rate, color="grey", linestyle="--", linewidth=1.2,
               label=f"No-skill baseline ({base_rate:.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Entry-time Features Only\n"
                 "(more informative than ROC given ~27% base loss rate)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_pr_comparison.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_pr_comparison.png")


# ---------------------------------------------------------------------------
# Fig 3 — Calibration curve (best model)
# ---------------------------------------------------------------------------

def fig_calibration(probas: dict, y_test, best_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    y_prob = probas[best_name]
    fop, mpv = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

    ax.plot(mpv, fop, "s-", color=LOSS_COLOR, linewidth=2, markersize=6,
            label=f"{best_name}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5,
            label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (observed)")
    ax.set_title(f"Calibration Curve — {best_name}")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_calibration.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_calibration.png")


# ---------------------------------------------------------------------------
# Fig 4 — Confusion matrix (best model, threshold 0.5)
# ---------------------------------------------------------------------------

def fig_confusion_best(probas: dict, y_test, best_name: str) -> None:
    y_prob = probas[best_name]
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Profit", "Loss"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}\n(threshold = 0.5)")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_confusion_best.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_confusion_best.png")


# ---------------------------------------------------------------------------
# Fig 5 — SHAP beeswarm (best entry-only model)
# ---------------------------------------------------------------------------

def fig_shap_summary(sv) -> None:
    import shap
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("SHAP Feature Importance — Best Entry-time Model",
              pad=12, fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig_shap_summary.png", **SAVEFIG_KW)
    plt.close("all")
    print("  saved fig_shap_summary.png")


# ---------------------------------------------------------------------------
# Fig 6 — Entry-only vs Entry+Duration AUC bar
# ---------------------------------------------------------------------------

def fig_auc_entry_vs_duration(auc_entry: float, auc_duration: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [
        "Entry-only\n(deployable)",
        "Entry + Duration\n(post-close — not deployable)",
    ]
    values = [auc_entry, auc_duration]
    colors = [PROFIT_COLOR, "dimgrey"]

    bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor="white",
                  linewidth=0)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.6,
               label="Random baseline (0.5)")
    ax.set_ylim(0, max(values) * 1.20)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        "Entry-time AUC vs Duration-augmented AUC\n"
        "(Duration is observed only after the trade closes)"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_auc_entry_vs_duration.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  saved fig_auc_entry_vs_duration.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    apply_style()
    print("Loading clean trades...")
    df = pd.read_parquet(PROCESSED / "trades_clean.parquet")

    print("Running modeling pipeline (20-35 min expected)...")
    results = save_modeling_outputs(df)

    print("\nGenerating figures...")
    fig_roc_comparison(results["probas"], results["y_test"])
    fig_pr_comparison(results["probas"], results["y_test"])
    fig_calibration(results["probas"], results["y_test"], results["best_model_name"])
    fig_confusion_best(results["probas"], results["y_test"], results["best_model_name"])
    fig_shap_summary(results["shap_sv"])
    fig_auc_entry_vs_duration(results["auc_entry"], results["auc_duration"])

    print("\n=== Model comparison (entry-time features only) ===")
    print(results["comparison"].to_string(index=False))
    print(f"\nBest model : {results['best_model_name']}")
    print(f"5-fold CV ROC-AUC : {results['cv_auc']:.4f}")

    print("\n=== Leakage check ===")
    print(f"  Entry-only AUC     : {results['auc_entry']:.4f}")
    print(f"  Entry+duration AUC : {results['auc_duration']:.4f}")
    print(f"  Duration adds      : {results['auc_duration'] - results['auc_entry']:.4f}")

    print("\n=== Top SHAP features ===")
    print(results["shap_df"].head(10).to_string(index=False))

    print("\nAll done.")
