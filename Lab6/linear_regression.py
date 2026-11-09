"""
Linear regression training pipeline using PyTorch.
Builds, trains, evaluates, and saves the student score prediction model.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from student_data import get_tensors


class LinearRegressionModel(nn.Module):
    """Linear Regression model implemented as a PyTorch neural network layer."""

    def __init__(self, in_features=3, out_features=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def train_model(model, X_train, y_train, X_test, y_test, lr=0.05, epochs=150):
    """Trains the PyTorch model using MSELoss and SGD optimizer."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    print("\n--- Training Loop Progress ---")
    for epoch in range(1, epochs + 1):
        # 1. Forward pass
        model.train()
        y_train_pred = model(X_train)
        loss = criterion(y_train_pred, y_train)

        # 2. Backward pass & optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. Test loss tracking
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = criterion(y_test_pred, y_test)

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        if epoch in [1, 10, 30, 60, 100, 150]:
            print(f"Epoch [{epoch:3d}/{epochs}] -> Train MSE: {loss.item():.4f} | Test MSE: {test_loss.item():.4f}")

    return train_losses, test_losses


def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluates performance metrics (MSE, RMSE, MAE, R2) on test data."""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
        y_true = y_test.numpy()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n" + "=" * 50)
    print(" MODEL EVALUATION METRICS (TEST SET)")
    print("=" * 50)
    print(f"  Mean Squared Error (MSE)      : {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE)     : {mae:.4f}")
    print(f"  Coefficient of Det. (R² Score): {r2:.4f} ({r2 * 100:.2f}%)")

    # Display learned parameters
    weights = model.linear.weight.detach().numpy().flatten()
    bias = model.linear.bias.item()

    print("\nLearned Model Parameters:")
    for name, w in zip(feature_names, weights):
        print(f"  * Weight ({name:12s}): {w:+.4f}")
    print(f"  * Bias (Intercept)    : {bias:+.4f}")

    return y_true, y_pred, r2


def save_visualizations(train_losses, test_losses, y_true, y_pred, r2, output_dir):
    """Plots and saves the loss curve and predicted-vs-actual scatter plots."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Loss Curve
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="Train Loss (MSE)", color="#2563EB", linewidth=2.0)
    plt.plot(test_losses, label="Test Loss (MSE)", color="#DC2626", linestyle="--", linewidth=2.0)
    plt.title("PyTorch Linear Regression: Loss Convergence Over Epochs", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"\n[Saved Plot] Loss curve saved to: {loss_path}")

    # 2. Predicted vs. Actual Scores
    pred_path = os.path.join(output_dir, "predicted_vs_actual.png")
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, color="#1D4ED8", edgecolors="#0F172A", alpha=0.8, s=55, label="Test Samples")

    # Ideal diagonal line (y = x)
    min_val = min(y_true.min(), y_pred.min()) - 2
    max_val = max(y_true.max(), y_pred.max()) + 2
    plt.plot([min_val, max_val], [min_val, max_val], color="#EF4444", linestyle="--", linewidth=1.8, label="Ideal (y = x)")

    plt.title("Actual vs. Predicted Final Exam Scores", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Actual Score", fontsize=10)
    plt.ylabel("Predicted Score", fontsize=10)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.text(min_val + 4, max_val - 6, f"R² Score = {r2:.4f}", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF3C7", edgecolor="#D97706"))
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(pred_path, dpi=150)
    plt.close()
    print(f"[Saved Plot] Predicted vs Actual plot saved to: {pred_path}")


def main():
    print("=" * 60)
    print("  LAB 6: PYTORCH LINEAR REGRESSION TRAINING PIPELINE")
    print("=" * 60)

    torch.manual_seed(42)

    # 1. Load preprocessed tensors
    X_train, X_test, y_train, y_test, scaler, feature_names = get_tensors()
    print(f"Training on {X_train.shape[0]} samples with features: {feature_names}")

    # 2. Instantiate Model
    model = LinearRegressionModel(in_features=3, out_features=1)

    # 3. Train Model
    train_losses, test_losses = train_model(model, X_train, y_train, X_test, y_test, lr=0.05, epochs=150)

    # 4. Evaluate Model
    y_true, y_pred, r2 = evaluate_model(model, X_test, y_test, feature_names)

    # 5. Save Model Checkpoint
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "linear_regression.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n[Saved Model] PyTorch weights saved to: {model_path}")

    # 6. Save Visualizations
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    save_visualizations(train_losses, test_losses, y_true, y_pred, r2, outputs_dir)

    print("\n" + "=" * 60)
    print(" Linear Regression pipeline executed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
