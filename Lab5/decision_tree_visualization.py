"""
Lab 5 - Activity 4: Decision Tree Comprehensive Visualizations
Author: Divyansh Verma (IC-2K23-31)
Course: IC-709D / IC-703 C | Artificial Intelligence Laboratory (Unit 5, CO4, CO5)

Description:
Builds a unified visual analytics dashboard for all three Decision Tree models:
1. Multi-panel Confusion Matrices (Placement, Disease, Loan)
2. Relative Feature Importance Analysis across all domains
Outputs: outputs/confusion_matrices.png & outputs/feature_importances.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


def main():
    print("=" * 65)
    print("  ACTIVITY 4: DECISION TREE ADVANCED VISUALIZATIONS")
    print("=" * 65)

    base_dir = os.path.dirname(__file__)
    datasets_dir = os.path.join(base_dir, "..", "datasets")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Train models and generate predictions
    # -------------------------------------------------------------------------
    # Domain 1: Placement
    df_p = pd.read_csv(os.path.join(datasets_dir, "placement_processed.csv"))
    X_p = df_p[["cgpa", "iq", "internships", "work_experience"]]
    y_p = df_p["placed"]
    X_p_tr, X_p_te, y_p_tr, y_p_te = train_test_split(X_p, y_p, test_size=0.2, random_state=42, stratify=y_p)
    dt_p = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_p_tr, y_p_tr)
    cm_p = confusion_matrix(y_p_te, dt_p.predict(X_p_te))

    # Domain 2: Disease
    df_d = pd.read_csv(os.path.join(datasets_dir, "disease_data.csv"))
    X_d = df_d[["age", "glucose", "blood_pressure", "bmi"]]
    y_d = df_d["disease_present"]
    X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(X_d, y_d, test_size=0.2, random_state=42, stratify=y_d)
    dt_d = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42).fit(X_d_tr, y_d_tr)
    cm_d = confusion_matrix(y_d_te, dt_d.predict(X_d_te))

    # Domain 3: Loan
    df_l = pd.read_csv(os.path.join(datasets_dir, "loan_data.csv"))
    df_l["education"] = LabelEncoder().fit_transform(df_l["education"])
    X_l = df_l[["applicant_income", "loan_amount_k", "credit_history", "education"]]
    y_l = df_l["loan_approved"]
    X_l_tr, X_l_te, y_l_tr, y_l_te = train_test_split(X_l, y_l, test_size=0.2, random_state=42, stratify=y_l)
    dt_l = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_l_tr, y_l_te_l := y_l_tr) # fit
    dt_l = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_l_tr, y_l_tr)
    cm_l = confusion_matrix(y_l_te, dt_l.predict(X_l_te))

    # -------------------------------------------------------------------------
    # 2. Multi-Panel Confusion Matrix Dashboard
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    domains = [
        ("Placement Prediction", cm_p, ["Not Placed", "Placed"], "Blues", axes[0]),
        ("Disease Prediction", cm_d, ["Healthy", "Disease"], "Reds", axes[1]),
        ("Loan Approval Prediction", cm_l, ["Rejected", "Approved"], "Greens", axes[2]),
    ]

    for title, cm, labels, cmap, ax in domains:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=cmap, ax=ax, colorbar=False)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.grid(False)

    plt.suptitle("Laboratory 5: Confusion Matrices Across Three Intelligent Decision Domains", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    cm_path = os.path.join(outputs_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Dashboard 1] Multi-panel confusion matrices saved to: {cm_path}")

    # -------------------------------------------------------------------------
    # 3. Multi-Panel Feature Importance Dashboard
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    fi_data = [
        ("Placement: Feature Importance", X_p.columns, dt_p.feature_importances_, "#3B82F6", axes[0]),
        ("Disease: Feature Importance", X_d.columns, dt_d.feature_importances_, "#EF4444", axes[1]),
        ("Loan: Feature Importance", X_l.columns, dt_l.feature_importances_, "#10B981", axes[2]),
    ]

    for title, cols, importances, color, ax in fi_data:
        sorted_idx = np.argsort(importances)
        ax.barh(np.array(cols)[sorted_idx], importances[sorted_idx], color=color, edgecolor="#1F2937", alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Gini / Entropy Importance")
        ax.set_xlim(0, 1.0)
        for i, v in enumerate(importances[sorted_idx]):
            ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8, fontweight="bold")

    plt.suptitle("Laboratory 5: Decision Tree Feature Importance Profiles", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fi_path = os.path.join(outputs_dir, "feature_importances.png")
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Dashboard 2] Multi-panel feature importances saved to: {fi_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
