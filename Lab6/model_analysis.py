"""
Comparative analysis of PyTorch vs Scikit-Learn linear regression
and learning rate convergence study.
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from student_data import get_tensors


def compare_pytorch_vs_sklearn():
    """Trains both PyTorch and Scikit-Learn models and compares parameters and metrics."""
    print("=" * 65)
    print(" 1. MODEL COMPARISON: PYTORCH (SGD) VS SCIKIT-LEARN (OLS)")
    print("=" * 65)

    X_train_t, X_test_t, y_train_t, y_test_t, scaler, feature_names = get_tensors()

    # NumPy arrays for Scikit-Learn
    X_train_np = X_train_t.numpy()
    y_train_np = y_train_t.numpy().flatten()
    X_test_np = X_test_t.numpy()
    y_test_np = y_test_t.numpy().flatten()

    # 1. Scikit-Learn (Closed-Form Ordinary Least Squares)
    t0_sk = time.perf_counter()
    sk_model = LinearRegression()
    sk_model.fit(X_train_np, y_train_np)
    sk_time = (time.perf_counter() - t0_sk) * 1000

    sk_pred = sk_model.predict(X_test_np)
    sk_mse = mean_squared_error(y_test_np, sk_pred)
    sk_mae = mean_absolute_error(y_test_np, sk_pred)
    sk_r2 = r2_score(y_test_np, sk_pred)

    # 2. PyTorch (Iterative Gradient-Based Optimization)
    torch.manual_seed(42)
    pt_model = nn.Linear(3, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(pt_model.parameters(), lr=0.05)

    t0_pt = time.perf_counter()
    for _ in range(150):
        optimizer.zero_grad()
        loss = criterion(pt_model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
    pt_time = (time.perf_counter() - t0_pt) * 1000

    pt_model.eval()
    with torch.no_grad():
        pt_pred = pt_model(X_test_t).numpy().flatten()
    pt_mse = mean_squared_error(y_test_np, pt_pred)
    pt_mae = mean_absolute_error(y_test_np, pt_pred)
    pt_r2 = r2_score(y_test_np, pt_pred)

    # Comparison summary table
    metrics_df = pd.DataFrame({
        "Metric / Parameter": [
            "Test MSE",
            "Test MAE",
            "Test R^2 Score",
            "Training Latency (ms)",
            "Weight (study_hours)",
            "Weight (attendance)",
            "Weight (past_score)",
            "Bias (Intercept)"
        ],
        "Scikit-Learn (OLS)": [
            f"{sk_mse:.4f}",
            f"{sk_mae:.4f}",
            f"{sk_r2:.4f}",
            f"{sk_time:.2f} ms",
            f"{sk_model.coef_[0]:+.4f}",
            f"{sk_model.coef_[1]:+.4f}",
            f"{sk_model.coef_[2]:+.4f}",
            f"{sk_model.intercept_:+.4f}"
        ],
        "PyTorch (SGD, 150 eps)": [
            f"{pt_mse:.4f}",
            f"{pt_mae:.4f}",
            f"{pt_r2:.4f}",
            f"{pt_time:.2f} ms",
            f"{pt_model.weight.detach().numpy()[0, 0]:+.4f}",
            f"{pt_model.weight.detach().numpy()[0, 1]:+.4f}",
            f"{pt_model.weight.detach().numpy()[0, 2]:+.4f}",
            f"{pt_model.bias.item():+.4f}"
        ]
    })

    print(metrics_df.to_string(index=False))


def analyze_learning_rates(output_dir):
    """Evaluates the effect of different learning rates on training convergence."""
    print("\n" + "=" * 65)
    print(" 2. LEARNING RATE SENSITIVITY & CONVERGENCE STUDY")
    print("=" * 65)

    X_train_t, X_test_t, y_train_t, y_test_t, _, _ = get_tensors()
    learning_rates = [0.001, 0.01, 0.05, 0.2, 0.8]
    epochs = 80

    history = {}
    colors = ["#9CA3AF", "#F59E0B", "#10B981", "#3B82F6", "#EF4444"]

    plt.figure(figsize=(10, 5.5))

    for lr, color in zip(learning_rates, colors):
        torch.manual_seed(42)
        model = nn.Linear(3, 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        losses = []
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        history[lr] = losses
        final_loss = losses[-1]
        print(f"  Learning Rate lr = {lr:<5} -> Final Loss after {epochs} epochs: {final_loss:.4f}")

        # Plot curve
        plt.plot(losses, label=f"lr = {lr} (Final Loss: {final_loss:.2f})", color=color, linewidth=1.8)

    plt.title("PyTorch Gradient Descent: Convergence Across Learning Rates", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Training Epochs", fontsize=10)
    plt.ylabel("Training Loss (MSE)", fontsize=10)
    plt.ylim(0, 1500)
    plt.legend(fontsize=9, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "learning_rate_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[Saved Plot] Learning rate convergence plot saved to: {plot_path}")


def main():
    compare_pytorch_vs_sklearn()
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    analyze_learning_rates(output_dir)
    print("\n" + "=" * 65)
    print(" Model analysis and convergence study completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
