"""
Description:
Demonstrates automatic differentiation and gradient computation in PyTorch:
1. requires_grad tracking and computational graph inspectability (grad_fn)
2. Analytical derivative verification (Scalar & Multivariable functions)
3. Gradient accumulation mechanics and explicit resetting (.grad.zero_())
4. Contextual gradient disabling (torch.no_grad() and detach())
5. First-principles Gradient Descent optimization loop using autograd
"""

import torch


def demo_scalar_autograd():
    print("\n" + "=" * 55)
    print(" 1. SCALAR AUTOGRAD & ANALYTICAL VERIFICATION")
    print("=" * 55)

    # Function: y = 3*x^2 + 2*x + 1  at x = 4.0
    # Analytical dy/dx = 6*x + 2 = 6*(4) + 2 = 26.0
    x = torch.tensor(4.0, requires_grad=True)
    y = 3 * (x ** 2) + 2 * x + 1

    print(f"Input x               : {x.item()}")
    print(f"Evaluated y = f(x)    : {y.item()}")
    print(f"Computational Graph   : y.grad_fn = {y.grad_fn}")

    # Backward pass
    y.backward()

    analytical_grad = 6 * 4.0 + 2
    autograd_grad = x.grad.item()

    print(f"\nAutograd Gradient (dy/dx)   : {autograd_grad}")
    print(f"Analytical Gradient (6x + 2): {analytical_grad}")
    print(f"Exact match verified        : {autograd_grad == analytical_grad}")


def demo_multivariable_autograd():
    print("\n" + "=" * 55)
    print(" 2. MULTIVARIABLE GRADIENT COMPUTATION")
    print("=" * 55)

    # Function: z = 2*u^3 + v^2 - 4*u*v  at u = 2.0, v = 3.0
    # Analytical:
    # dz/du = 6*u^2 - 4*v = 6*(4) - 4*(3) = 24 - 12 = 12.0
    # dz/dv = 2*v - 4*u   = 2*(3) - 4*(2) = 6 - 8   = -2.0
    u = torch.tensor(2.0, requires_grad=True)
    v = torch.tensor(3.0, requires_grad=True)

    z = 2 * (u ** 3) + (v ** 2) - 4 * u * v
    z.backward()

    print(f"Variables: u = {u.item()}, v = {v.item()}")
    print(f"Evaluated function z = {z.item()}")
    print(f"Computed dz/du: {u.grad.item():.2f} (Expected: 12.00)")
    print(f"Computed dz/dv: {v.grad.item():.2f} (Expected: -2.00)")


def demo_gradient_accumulation():
    print("\n" + "=" * 55)
    print(" 3. GRADIENT ACCUMULATION & ZEROING")
    print("=" * 55)

    w = torch.tensor(3.0, requires_grad=True)

    # First backward pass: loss = w^2 -> dL/dw = 2*w = 6.0
    loss1 = w ** 2
    loss1.backward()
    print(f"Pass 1 -> loss = w^2 | w.grad = {w.grad.item():.2f}")

    # Second backward pass without zeroing gradients -> accumulates to 6.0 + 6.0 = 12.0
    loss2 = w ** 2
    loss2.backward()
    print(f"Pass 2 (without zeroing) -> w.grad accumulated to: {w.grad.item():.2f}")

    # Explicitly clearing gradients (standard PyTorch practice before optimizer step)
    w.grad.zero_()
    print(f"After w.grad.zero_()    -> w.grad reset to: {w.grad.item():.2f}")


def demo_no_grad_and_detach():
    print("\n" + "=" * 55)
    print(" 4. DISABLING GRADIENT TRACKING (INFERENCE)")
    print("=" * 55)

    x = torch.tensor(5.0, requires_grad=True)

    # Standard tracking
    y1 = x * 2
    print(f"Normal computation : y1.requires_grad = {y1.requires_grad}")

    # Context manager for evaluation / inference (saves memory & compute)
    with torch.no_grad():
        y2 = x * 2
        print(f"Inside torch.no_grad(): y2.requires_grad = {y2.requires_grad}")

    # Detached tensor
    y3 = (x * 2).detach()
    print(f"Detached tensor       : y3.requires_grad = {y3.requires_grad}")


def demo_gradient_descent_optimization():
    print("\n" + "=" * 55)
    print(" 5. FIRST-PRINCIPLES GRADIENT DESCENT OPTIMIZATION")
    print("=" * 55)

    # Objective: Minimize Loss L(w) = (w - 7.0)^2
    # Global minimum is at w = 7.0 where Loss = 0.0
    w = torch.tensor(0.0, requires_grad=True)
    learning_rate = 0.2
    epochs = 15

    print(f"Target: Find w that minimizes L(w) = (w - 7)^2")
    print(f"Initial parameter: w = {w.item():.4f} | Learning rate: {learning_rate}\n")

    for epoch in range(1, epochs + 1):
        loss = (w - 7.0) ** 2
        loss.backward()

        # Update parameter using gradient descent: w = w - lr * grad
        with torch.no_grad():
            w -= learning_rate * w.grad

        # Zero gradients for the next iteration
        w.grad.zero_()

        if epoch in [1, 2, 5, 10, 15]:
            print(f"  Epoch {epoch:2d}: w = {w.item():.4f} | Loss = {loss.item():.6f}")

    print(f"\nConverged parameter w: {w.item():.4f} (Optimal is 7.0000)")


def main():
    print("=" * 60)
    print("  LAB 6: AUTOMATIC DIFFERENTIATION & AUTOGRAD ENGINE")
    print("=" * 60)

    demo_scalar_autograd()
    demo_multivariable_autograd()
    demo_gradient_accumulation()
    demo_no_grad_and_detach()
    demo_gradient_descent_optimization()

    print("\n" + "=" * 60)
    print(" Gradient computation module executed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
