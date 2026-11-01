"""
Lab 5 - Analysis: Decision Tree Performance Evaluation & Metric Comparison
Author: Divyansh Verma (IC-2K23-31)
Course: IC-709D / IC-703 C | Artificial Intelligence Laboratory (Unit 5, CO4, CO5)

Description:
Systematically evaluates Decision Tree performance across:
1. Splitting Criteria: Gini Impurity vs. Information Gain (Entropy)
2. Tree Depth Constraints: max_depth from 1 to 6 (overfitting vs underfitting)
3. Multi-Metric Analysis: Accuracy, Precision, Recall, F1-Score, and Train/Test Latency
Outputs: outputs/metrics_comparison.png
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    print("=" * 70)
    print("  ANALYSIS: DECISION TREE PERFORMANCE & METRIC COMPARISON")
    print("=" * 70)

    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "placement_processed.csv")
    df = pd.read_csv(data_path)
    X = df[["cgpa", "iq", "internships", "work_experience"]]
    y = df["placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. Experiment A: Gini vs Entropy Comparison across depths
    depths = [1, 2, 3, 4, 5, None]
    results = []

    print("\n--- Hyperparameter & Splitting Criteria Experiment ---")
    print(f"{'Criterion':<10} | {'Max Depth':<10} | {'Train Acc':<11} | {'Test Acc':<10} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} | {'Latency (ms)'}")
    print("-" * 92)

    for criterion in ["gini", "entropy"]:
        for depth in depths:
            t0 = time.perf_counter()
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
            clf.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            y_pred = clf.predict(X_test)
            latency_ms = (time.perf_counter() - t0) * 1000

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            depth_str = str(depth) if depth is not None else "None (Full)"
            print(f"{criterion:<10} | {depth_str:<10} | {train_acc*100:>8.1f} %  | {acc*100:>7.1f} % | {prec:>9.3f} | {rec:>8.3f} | {f1:>8.3f} | {latency_ms:>7.2f} ms")

            results.append({
                "criterion": criterion,
                "depth": depth if depth is not None else 6,
                "depth_label": depth_str,
                "train_acc": train_acc,
                "test_acc": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "latency_ms": latency_ms
            })

    res_df = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 3. Multi-Metric Comparative Dashboard
    # -------------------------------------------------------------------------
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    plot_path = os.path.join(outputs_dir, "metrics_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Train vs Test Accuracy across depths (Checking Overfitting)
    gini_res = res_df[res_df["criterion"] == "gini"]
    entropy_res = res_df[res_df["criterion"] == "entropy"]

    x_labels = gini_res["depth_label"]
    x_pos = np.arange(len(x_labels))

    axes[0].plot(x_pos, gini_res["train_acc"], marker="o", linestyle="-", color="#2563EB", label="Gini - Train Acc")
    axes[0].plot(x_pos, gini_res["test_acc"], marker="s", linestyle="--", color="#3B82F6", label="Gini - Test Acc")
    axes[0].plot(x_pos, entropy_res["train_acc"], marker="^", linestyle="-", color="#DC2626", label="Entropy - Train Acc")
    axes[0].plot(x_pos, entropy_res["test_acc"], marker="d", linestyle="--", color="#EF4444", label="Entropy - Test Acc")

    axes[0].set_title("Model Capacity vs Generalization (Depth vs Accuracy)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Tree Max Depth")
    axes[0].set_ylabel("Accuracy Score")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_ylim(0.5, 1.05)
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Multi-Metric Comparison (Depth = 3)
    best_gini = res_df[(res_df["criterion"] == "gini") & (res_df["depth_label"] == "3")].iloc[0]
    best_entropy = res_df[(res_df["criterion"] == "entropy") & (res_df["depth_label"] == "3")].iloc[0]

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    gini_vals = [best_gini["test_acc"], best_gini["precision"], best_gini["recall"], best_gini["f1"]]
    entropy_vals = [best_entropy["test_acc"], best_entropy["precision"], best_entropy["recall"], best_entropy["f1"]]

    w = 0.35
    m_pos = np.arange(len(metrics))
    axes[1].bar(m_pos - w/2, gini_vals, width=w, label="Gini Impurity (Depth=3)", color="#3B82F6", edgecolor="#1E3A8A")
    axes[1].bar(m_pos + w/2, entropy_vals, width=w, label="Information Gain (Depth=3)", color="#EF4444", edgecolor="#991B1B")

    axes[1].set_title("Metric Comparison: Gini vs Information Gain", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Score (0.0 to 1.0)")
    axes[1].set_xticks(m_pos)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylim(0.0, 1.15)
    for i in m_pos:
        axes[1].text(i - w/2, gini_vals[i] + 0.02, f"{gini_vals[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        axes[1].text(i + w/2, entropy_vals[i] + 0.02, f"{entropy_vals[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=8.5)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Laboratory 5: Decision Tree Empirical Performance Analysis", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Analysis Plot] Performance metric comparison saved to: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
