"""
Global model interpretability using Gini impurity-based feature importance.
Analyzes and visualizes the overall contribution of each feature to the model.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import get_trained_decision_tree


def compute_global_importance():
    """Extracts and tabulates Gini impurity-based feature importances."""
    model, X_train, X_test, y_train, y_test, feature_names, _, _ = get_trained_decision_tree()

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    df_importance = pd.DataFrame({
        "Feature": np.array(feature_names)[sorted_idx],
        "Importance Score": importances[sorted_idx],
        "Percentage": [f"{v * 100:.1f}%" for v in importances[sorted_idx]]
    })

    return df_importance, feature_names, importances


def plot_global_importance(feature_names, importances, output_dir):
    """Generates and saves a horizontal bar chart of global feature importances."""
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "global_importance.png")

    sorted_idx = np.argsort(importances)
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_vals = importances[sorted_idx]

    plt.figure(figsize=(8, 4.5))
    bars = plt.barh(sorted_features, sorted_vals, color="#3B82F6", edgecolor="#1E3A8A", height=0.55)

    plt.title("Global Feature Importance (Mean Decrease in Impurity)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Gini Importance Score", fontsize=10)
    plt.ylabel("Features", fontsize=10)
    plt.xlim(0, max(sorted_vals) + 0.1)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.015, bar.get_y() + bar.get_height() / 2,
                 f"{width:.3f} ({width * 100:.1f}%)",
                 va="center", fontsize=9, fontweight="bold", color="#1E293B")

    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Plot Saved] Global feature importance chart saved to: {plot_path}")


def main():
    print("=" * 60)
    print("  EXPLAINABLE AI: GLOBAL FEATURE IMPORTANCE (MDI)")
    print("=" * 60)

    df_imp, features, importances = compute_global_importance()
    print("\nGlobal Feature Importance Ranking:")
    print(df_imp.to_string(index=False))

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    plot_global_importance(features, importances, output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
