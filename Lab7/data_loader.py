"""
Data loader and model setup helper for Explainable AI (XAI).
Loads the processed student placement dataset and prepares the decision tree model.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


DATASET_FILENAME = "placement_processed.csv"
FEATURE_COLS = ["cgpa", "iq", "internships", "work_experience"]
TARGET_COL = "placed"


def get_dataset_path():
    """Returns absolute path to datasets/placement_processed.csv."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "..", "datasets", DATASET_FILENAME)


def load_dataset():
    """Loads placement_processed.csv from datasets folder."""
    path = get_dataset_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed placement dataset not found at: {path}")
    return pd.read_csv(path)


def get_train_test_splits(test_size=0.2, seed=42):
    """Splits dataset into stratified train and test sets."""
    df = load_dataset()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, FEATURE_COLS


def get_trained_decision_tree(max_depth=3, seed=42):
    """Trains and returns a Decision Tree Classifier alongside train/test splits."""
    X_train, X_test, y_train, y_test, feature_names = get_train_test_splits(seed=seed)

    model = DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=seed)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    return model, X_train, X_test, y_train, y_test, feature_names, train_acc, test_acc


def main():
    print("=" * 60)
    print("  EXPLAINABLE AI: DATA LOADER & MODEL PREPARATION")
    print("=" * 60)

    df = load_dataset()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nSample records:")
    print(df.head(3))

    model, X_train, X_test, y_train, y_test, features, tr_acc, te_acc = get_trained_decision_tree()
    print("\nDataset Partitioning:")
    print(f"  * Training set: {X_train.shape[0]} samples")
    print(f"  * Testing set : {X_test.shape[0]} samples")
    print(f"  * Features    : {features}")

    print("\nBaseline Model Accuracy:")
    print(f"  * Train Accuracy: {tr_acc * 100:.1f}%")
    print(f"  * Test Accuracy : {te_acc * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
