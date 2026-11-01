import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Unit 3: AI Programming Frameworks & Data Preparation (CO2, CO5)
# Practical: End-to-End Data Preprocessing Pipeline on Student Placement Dataset


def load_dataset():
    # Look for dataset in root datasets/ directory
    possible_paths = [
        os.path.join(
            os.path.dirname(__file__), "..", "datasets", "placement.csv"
        ),
        os.path.join("datasets", "placement.csv"),
        "placement.csv",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"[Step 1] Dataset loaded successfully from: {path}\n")
            return pd.read_csv(path)

    raise FileNotFoundError("Could not locate 'placement.csv' in datasets/")


def perform_eda(df):
    print("--- [Step 2] Exploratory Data Analysis (EDA) ---")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("First 5 records:")
    print(df.head(), "\n")

    print("Summary Statistics:")
    print(df.describe().round(2), "\n")


def handle_missing_values(df):
    print("--- [Step 3] Data Cleaning & Missing Value Handling ---")
    null_counts = df.isnull().sum()
    print("Missing values per column before imputation:")
    print(null_counts[null_counts > 0], "\n")

    df_clean = df.copy()
    # Impute missing numerical values using column median
    for col in ["cgpa", "iq"]:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"Filled missing values in '{col}' with median = {median_val}")

    print("Total missing values after imputation:", df_clean.isnull().sum().sum())
    print()
    return df_clean


def encode_categorical(df):
    print("--- [Step 4] Categorical Data Encoding ---")
    df_encoded = df.copy()

    # Encode binary categorical feature (work_experience)
    le = LabelEncoder()
    df_encoded["work_experience"] = le.fit_transform(
        df_encoded["work_experience"]
    )
    print("Encoded 'work_experience':", dict(zip(le.classes_, [0, 1])))
    print(
        df_encoded[["student_id", "work_experience", "placed"]].head(3), "\n"
    )
    return df_encoded


def scale_and_split(df):
    print("--- [Step 5 & 6] Feature Scaling & Train-Test Split ---")

    # Features (X) and Target (y)
    feature_cols = ["cgpa", "iq", "internships", "work_experience"]
    X = df[feature_cols]
    y = df["placed"]

    # 80% Train, 20% Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(
        f"Training set: {X_train.shape[0]} samples | Testing set:"
        f" {X_test.shape[0]} samples"
    )

    # Standardize numerical features (mean = 0, std = 1)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    scale_cols = ["cgpa", "iq", "internships"]
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    print("\nScaled Training Features sample (mean approx 0, std approx 1):")
    print(X_train_scaled.head(3).round(3))
    print()

    return X, y, X_train, X_train_scaled


def generate_visualizations(df, X_raw, X_scaled):
    print("--- [Step 7] Data Visualization ---")
    if not HAS_MATPLOTLIB:
        print(
            "Note: matplotlib is not installed. To generate visualization"
            " plots, install it with: pip install matplotlib\n"
        )
        return

    plt.figure(figsize=(11, 8))

    # 1. CGPA Distribution
    plt.subplot(2, 2, 1)
    plt.hist(df["cgpa"], bins=10, color="skyblue", edgecolor="black")
    plt.title("Distribution of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")

    # 2. Placement Count (Target Distribution)
    plt.subplot(2, 2, 2)
    placed_counts = df["placed"].value_counts()
    plt.bar(
        ["Not Placed (0)", "Placed (1)"],
        [placed_counts[0], placed_counts[1]],
        color=["salmon", "lightgreen"],
        edgecolor="black",
    )
    plt.title("Placement Outcome Distribution")
    plt.ylabel("Number of Students")

    # 3. CGPA vs IQ Scatter Plot
    plt.subplot(2, 2, 3)
    plt.scatter(
        df["cgpa"],
        df["iq"],
        c=df["placed"],
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="k",
    )
    plt.title("CGPA vs. IQ (Color = Placed)")
    plt.xlabel("CGPA")
    plt.ylabel("IQ Score")

    # 4. Feature Scaling Effect
    plt.subplot(2, 2, 4)
    plt.plot(
        X_raw["cgpa"].values[:20],
        label="Raw CGPA",
        marker="o",
        linestyle="--",
        color="orange",
    )
    plt.plot(
        X_scaled["cgpa"].values[:20],
        label="Scaled CGPA",
        marker="s",
        color="purple",
    )
    plt.title("Feature Scaling Effect (First 20 Students)")
    plt.xlabel("Student Index")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "data_preprocessing_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Visualization plots saved successfully to: {plot_path}\n")


def main():
    try:
        df_raw = load_dataset()
        perform_eda(df_raw)

        df_clean = handle_missing_values(df_raw)
        df_encoded = encode_categorical(df_clean)

        X, y, X_train, X_train_scaled = scale_and_split(df_encoded)
        generate_visualizations(df_clean, X_train, X_train_scaled)

        print("=== Data Preprocessing Pipeline Completed Successfully ===")

    except Exception as e:
        print(f"Error during preprocessing pipeline: {e}")


if __name__ == "__main__":
    main()
