"""

Description:
Creates and loads the student exam performance dataset (student_scores.csv),
handles train/test splitting, standardization, and conversion to PyTorch tensors.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_FILENAME = "student_scores.csv"


def get_dataset_path():
    """Returns absolute path to datasets/student_scores.csv."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "..", "datasets", DATASET_FILENAME)


def generate_student_data(num_samples=150, seed=42):
    """
    Generates realistic student exam performance data.
    Features:
      - study_hours: Hours studied per day (1 to 10)
      - attendance: Percentage class attendance (65 to 100)
      - past_score: Previous exam / quiz average (40 to 95)
    Target:
      - final_score: Final exam percentage (0 to 100)
    """
    np.random.seed(seed)

    study_hours = np.random.uniform(1.5, 9.5, num_samples).round(1)
    attendance = np.random.uniform(65.0, 98.0, num_samples).round(1)
    past_score = np.random.uniform(45.0, 95.0, num_samples).round(1)

    noise = np.random.normal(0, 2.5, num_samples)
    final_score = (
        4.8 * study_hours
        + 0.35 * attendance
        + 0.25 * past_score
        + noise
    )
    final_score = np.clip(final_score, 0.0, 100.0).round(1)

    df = pd.DataFrame({
        "student_id": [f"STU_{i+1:03d}" for i in range(num_samples)],
        "study_hours": study_hours,
        "attendance": attendance,
        "past_score": past_score,
        "final_score": final_score
    })
    return df


def load_data():
    """Loads existing dataset or generates it if not present."""
    filepath = get_dataset_path()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if not os.path.exists(filepath):
        print(f"Creating new dataset: {filepath}")
        df = generate_student_data()
        df.to_csv(filepath, index=False)
    else:
        print(f"Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)

    return df


def get_tensors(test_size=0.2, seed=42):
    """
    Splits dataset into train/test, applies standard scaling,
    and returns PyTorch float32 tensors.
    """
    df = load_data()
    feature_cols = ["study_hours", "attendance", "past_score"]
    target_col = "final_score"

    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch float32 tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler, feature_cols


def main():
    print("=" * 60)
    print("  STUDENT DATA PREPARATION & TENSOR LOADER")
    print("=" * 60)

    df = load_data()
    print(f"Total rows: {len(df)} | Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())

    X_train, X_test, y_train, y_test, scaler, cols = get_tensors()
    print("\nTensor Shapes:")
    print(f"  X_train: {X_train.shape} (dtype: {X_train.dtype})")
    print(f"  y_train: {y_train.shape} (dtype: {y_train.dtype})")
    print(f"  X_test : {X_test.shape} (dtype: {X_test.dtype})")
    print(f"  y_test : {y_test.shape} (dtype: {y_test.dtype})")
    print("=" * 60)


if __name__ == "__main__":
    main()
