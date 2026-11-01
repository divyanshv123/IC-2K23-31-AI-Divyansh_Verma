"""
Lab 3: Data Preprocessing Pipeline & Before-After Comparison
Author: Divyansh Verma (IC-2K23-31)
Subject: Artificial Intelligence Lab (Unit 3)

This script provides an easy, student-friendly step-by-step preprocessing
pipeline with clear side-by-side comparisons of the data before and after
each preprocessing operation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def load_dataset():
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "datasets", "placement.csv"),
        os.path.join("datasets", "placement.csv"),
        "placement.csv",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[1] Dataset loaded successfully from: {os.path.abspath(path)}")
            return pd.read_csv(path)
    raise FileNotFoundError("Could not locate 'placement.csv' in datasets/")


def main():
    print("=" * 65)
    print("      LAB 3: DATA PREPROCESSING & COMPARISON STUDY")
    print("=" * 65)

    # 1. Load Data
    raw_df = load_dataset()
    print(f"Total Records: {raw_df.shape[0]} | Columns: {raw_df.shape[1]}")
    print("\nInitial Snapshot (First 5 records):")
    print(raw_df.head())

    # -------------------------------------------------------------------------
    # 2. Missing Value Imputation Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 1: MISSING VALUES HANDLING (BEFORE vs AFTER)")
    print("=" * 65)

    null_before = raw_df.isnull().sum()
    print("Missing count before imputation:")
    print(null_before[null_before > 0])

    clean_df = raw_df.copy()
    num_cols = ["cgpa", "iq"]

    # Impute missing values using column median
    for col in num_cols:
        median_val = clean_df[col].median()
        clean_df[col] = clean_df[col].fillna(median_val)

    null_after = clean_df.isnull().sum()

    imputation_comparison = pd.DataFrame({
        "Missing Before": null_before[num_cols],
        "Median Imputed": [raw_df[col].median() for col in num_cols],
        "Missing After": null_after[num_cols]
    })
    print("\nComparison Table (Imputation):")
    print(imputation_comparison)

    # -------------------------------------------------------------------------
    # 3. Categorical Encoding Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 2: CATEGORICAL ENCODING (BEFORE vs AFTER)")
    print("=" * 65)

    encoded_df = clean_df.copy()
    le = LabelEncoder()
    encoded_df["work_experience_encoded"] = le.fit_transform(encoded_df["work_experience"])

    mapping_info = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label Encoding Mapping: {mapping_info}")

    encoding_sample = encoded_df[["work_experience", "work_experience_encoded"]].drop_duplicates().reset_index(drop=True)
    print("\nEncoded Mapping Table:")
    print(encoding_sample)

    # Replace original column with encoded numerical values
    encoded_df["work_experience"] = encoded_df["work_experience_encoded"]
    encoded_df.drop(columns=["work_experience_encoded"], inplace=True)

    # -------------------------------------------------------------------------
    # 4. Feature Scaling Comparison (StandardScaler vs MinMaxScaler)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 3: FEATURE SCALING COMPARISON")
    print("=" * 65)

    features = ["cgpa", "iq", "internships", "work_experience"]
    X = encoded_df[features]
    y = encoded_df["placed"]

    # Standardization: Mean = 0, Std = 1
    std_scaler = StandardScaler()
    X_std = pd.DataFrame(std_scaler.fit_transform(X), columns=features)

    # Normalization: Min = 0, Max = 1
    minmax_scaler = MinMaxScaler()
    X_minmax = pd.DataFrame(minmax_scaler.fit_transform(X), columns=features)

    # Create side-by-side comparison summary
    comparison_stats = pd.DataFrame({
        "Original Mean": X[num_cols].mean().round(2),
        "Original Std": X[num_cols].std().round(2),
        "Standardized Mean": X_std[num_cols].mean().round(2),
        "Standardized Std": X_std[num_cols].std().round(2),
        "MinMax Min": X_minmax[num_cols].min().round(2),
        "MinMax Max": X_minmax[num_cols].max().round(2),
    })

    print("Side-by-Side Scaling Statistics Comparison:")
    print(comparison_stats)

    # -------------------------------------------------------------------------
    # 5. Train-Test Split (80% Train, 20% Test)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("=" * 65)

    X_train, X_test, y_train, y_test = train_test_split(
        X_std, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Total Dataset size : {X.shape[0]} samples")
    print(f"Training Set (80%) : {X_train.shape[0]} samples ({X_train.shape[1]} features)")
    print(f"Testing Set (20%)  : {X_test.shape[0]} samples")
    print(f"Class Balance (Placed=1 / Not Placed=0 in train):")
    print(y_train.value_counts().to_dict())

    print("\n" + "=" * 65)
    print(" Preprocessing & Comparison Pipeline Executed Successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
