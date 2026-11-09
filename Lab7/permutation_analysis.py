"""
Permutation feature importance analysis evaluated on the test set.
Measures the drop in model accuracy when individual features are randomly shuffled.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from data_loader import get_trained_decision_tree


def evaluate_permutation_importance(n_repeats=30, seed=42):
    """Computes permutation importance on the held-out test split."""
    model, X_train, X_test, y_train, y_test, feature_names, _, test_acc = get_trained_decision_tree(seed=seed)

    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="accuracy"
    )

    importances_mean = perm_result.importances_mean
    importances_std = perm_result.importances_std
    sorted_idx = np.argsort(importances_mean)[::-1]

    df_perm = pd.DataFrame({
        "Feature": np.array(feature_names)[sorted_idx],
        "Mean Accuracy Drop": importances_mean[sorted_idx],
        "Standard Deviation": importances_std[sorted_idx]
    })

    return df_perm, feature_names, importances_mean, importances_std, test_acc


def plot_permutation_importance(feature_names, importances_mean, importances_std, output_dir):
    """Plots horizontal bar chart with error bars indicating stability across repeats."""
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "permutation_importance.png")

    sorted_idx = np.argsort(importances_mean)
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_means = importances_mean[sorted_idx]
    sorted_stds = importances_std[sorted_idx]

    plt.figure(figsize=(8, 4.5))
    bars = plt.barh(
        sorted_features, sorted_means,
        xerr=sorted_stds,
        color="#10B981", edgecolor="#065F46",
        height=0.55, capsize=4, error_kw=dict(lw=1.2, capthick=1.2, color="#065F46")
    )

    plt.title("Permutation Feature Importance on Test Set (Accuracy Drop)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Mean Accuracy Drop Upon Feature Shuffling", fontsize=10)
    plt.ylabel("Features", fontsize=10)
    plt.xlim(0, max(sorted_means + sorted_stds) + 0.15)

    # Annotate values
    for bar, m, s in zip(bars, sorted_means, sorted_stds):
        plt.text(m + s + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{m:.3f} ± {s:.3f}",
                 va="center", fontsize=9, fontweight="bold", color="#1E293B")

    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Plot Saved] Permutation importance chart saved to: {plot_path}")


def main():
    print("=" * 65)
    print("  EXPLAINABLE AI: PERMUTATION FEATURE IMPORTANCE (MDA)")
    print("=" * 65)

    df_perm, features, means, stds, test_acc = evaluate_permutation_importance()
    print(f"Baseline Test Accuracy: {test_acc * 100:.1f}%\n")
    print("Permutation Importance Ranking:")
    print(df_perm.to_string(index=False))

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    plot_permutation_importance(features, means, stds, output_dir)
    print("=" * 65)


if __name__ == "__main__":
    main()
