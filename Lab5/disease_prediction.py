"""
Lab 5 - Activity 2: Disease Prediction using Decision Tree Classifier
Author: Divyansh Verma (IC-2K23-31)
Course: IC-709D / IC-703 C | Artificial Intelligence Laboratory (Unit 5, CO4, CO5)

Description:
Trains a Decision Tree Classifier to predict metabolic/cardiovascular disease risk
from clinical parameters (Glucose, Blood Pressure, BMI, Age).
Evaluates accuracy, generates confusion matrix, and exports visual decision tree.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def load_data():
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "datasets", "disease_data.csv"),
        os.path.join("datasets", "disease_data.csv"),
        "disease_data.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"[Step 1] Loaded dataset from: {os.path.abspath(p)}")
            return pd.read_csv(p)
    raise FileNotFoundError("Could not find disease_data.csv in datasets/")


def main():
    print("=" * 65)
    print("  ACTIVITY 2: DISEASE PREDICTION (DECISION TREE)")
    print("=" * 65)

    # 1. Load Data
    df = load_data()
    print(f"Dataset Shape: {df.shape[0]} patients, {df.shape[1]} features\n")
    print(df.head())

    # 2. Features and Target
    feature_cols = ["age", "glucose", "blood_pressure", "bmi"]
    X = df[feature_cols]
    y = df["disease_present"]

    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining Cohort: {len(X_train)} | Testing Cohort: {len(X_test)}")

    # 4. Train Decision Tree Classifier
    model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    print("[Step 2] Decision Tree model successfully trained using Information Gain (Entropy).")

    # 5. Evaluate Performance
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "-" * 40)
    print("  CLINICAL MODEL EVALUATION (TEST SET)")
    print("-" * 40)
    print(f"  Accuracy  : {acc * 100:.2f} %")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}  (Sensitivity)")
    print(f"  F1-Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy / Low Risk", "Disease Diagnosed"]))

    # 6. Save Model
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "disease_decision_tree.pkl")
    joblib.dump(model, model_path)
    print(f"[Step 3] Serialized clinical model saved to: {model_path}")

    # 7. Visualize Decision Tree
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    plot_path = os.path.join(outputs_dir, "disease_prediction_tree.png")

    plt.figure(figsize=(12, 7))
    plot_tree(
        model,
        feature_names=feature_cols,
        class_names=["Healthy", "Disease"],
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title("Disease Prediction - Clinical Decision Tree Structure", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Step 4] Decision Tree visualization saved to: {plot_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
