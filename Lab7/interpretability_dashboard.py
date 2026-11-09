"""
Unified Explainable AI (XAI) Dashboard.
Consolidates global feature importances, permutation sensitivity,
decision tree paths, and local counterfactual what-if analysis into a single visualization.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

from data_loader import get_trained_decision_tree


def generate_dashboard(output_dir):
    """Constructs and saves the 4-panel Explainable AI dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    dashboard_path = os.path.join(output_dir, "dashboard.png")

    model, X_train, X_test, y_train, y_test, feature_names, _, test_acc = get_trained_decision_tree()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#FAFAFA")

    # -------------------------------------------------------------------------
    # Panel 1: Global Impurity-Based Importance (Top-Left)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    importances = model.feature_importances_
    idx1 = np.argsort(importances)
    ax1.barh(np.array(feature_names)[idx1], importances[idx1], color="#3B82F6", edgecolor="#1E3A8A", height=0.55)
    ax1.set_title("Panel A: Global Feature Importance (Gini Impurity / MDI)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Importance Score (Train Set)")
    ax1.set_xlim(0, 1.15)
    for i, v in enumerate(importances[idx1]):
        ax1.text(v + 0.02, i, f"{v:.3f} ({v*100:.1f}%)", va="center", fontsize=8.5, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 2: Permutation Feature Importance (Top-Right)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    perm = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring="accuracy")
    idx2 = np.argsort(perm.importances_mean)
    ax2.barh(np.array(feature_names)[idx2], perm.importances_mean[idx2],
             xerr=perm.importances_std[idx2], color="#10B981", edgecolor="#065F46",
             height=0.55, capsize=4, error_kw=dict(lw=1.2, capthick=1.2, color="#065F46"))
    ax2.set_title("Panel B: Permutation Feature Importance (Accuracy Drop / Test Set)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Mean Accuracy Drop Upon Feature Shuffling")
    ax2.set_xlim(0, max(perm.importances_mean + perm.importances_std) + 0.15)
    for i, (m, s) in enumerate(zip(perm.importances_mean[idx2], perm.importances_std[idx2])):
        ax2.text(m + s + 0.02, i, f"{m:.3f} ± {s:.3f}", va="center", fontsize=8.5, fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 3: Decision Tree Architecture (Bottom-Left)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Not Placed", "Placed"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax3
    )
    ax3.set_title("Panel C: Global Decision Tree Architecture & Split Boundaries", fontsize=11, fontweight="bold", pad=10)

    # -------------------------------------------------------------------------
    # Panel 4: Local Counterfactual What-If Sensitivity (Bottom-Right)
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    cgpa_range = np.linspace(5.0, 9.0, 100)
    # Vary CGPA while holding other features constant at borderline student profile (IQ=102, intern=0, work=0)
    synth_samples = pd.DataFrame({
        "cgpa": cgpa_range,
        "iq": [102.0] * 100,
        "internships": [0] * 100,
        "work_experience": [0] * 100
    })
    probs = model.predict_proba(synth_samples)[:, 1] * 100

    ax4.plot(cgpa_range, probs, color="#8B5CF6", linewidth=2.5, label="Placement Probability (%)")
    ax4.axvline(x=6.95, color="#EF4444", linestyle="--", linewidth=1.5, label="Primary Split Boundary (CGPA = 6.95)")
    ax4.scatter([6.5], [0], color="#DC2626", s=80, zorder=5, label="Target Unplaced Student (CGPA = 6.5)")
    ax4.scatter([7.3], [100], color="#16A34A", s=80, zorder=5, label="Counterfactual Goal (CGPA = 7.3)")

    ax4.set_title("Panel D: Local Counterfactual Analysis (CGPA vs Placement Prob)", fontsize=11, fontweight="bold", pad=10)
    ax4.set_xlabel("Student CGPA (Simulated Range 5.0 to 9.0)", fontsize=9.5)
    ax4.set_ylabel("Predicted Placement Probability (%)", fontsize=9.5)
    ax4.set_ylim(-5, 105)
    ax4.legend(loc="center left", fontsize=8.5)
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Explainable AI (XAI) Comprehensive Interpretability Dashboard", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Dashboard Saved] Unified XAI dashboard saved to: {dashboard_path}")


def main():
    print("=" * 65)
    print("  EXPLAINABLE AI: UNIFIED INTERPRETABILITY DASHBOARD")
    print("=" * 65)
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    generate_dashboard(output_dir)
    print("=" * 65)


if __name__ == "__main__":
    main()
