"""
Lab 6 - Part 1: PyTorch Tensor Operations and Manipulation
Author: Divyansh Verma (IC-2K23-31)
Course: IC-709D / IC-703 C | Artificial Intelligence Laboratory (Unit 6, CO4, CO5)

Description:
Covers fundamental tensor manipulation in PyTorch:
1. Tensor creation (zeros, ones, rand, arange, from lists)
2. Shape inspection and restructuring (view, reshape, transpose, squeeze, unsqueeze)
3. Mathematical operations, matrix algebra, and broadcasting
4. NumPy bridge (zero-copy memory sharing) and device management (CPU/CUDA)
"""

import numpy as np
import torch


def demo_tensor_creation():
    print("\n" + "=" * 50)
    print(" 1. TENSOR CREATION & METADATA")
    print("=" * 50)

    t_list = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t_zeros = torch.zeros((2, 3), dtype=torch.float32)
    t_ones = torch.ones((2, 3), dtype=torch.int32)
    t_rand = torch.rand((2, 3))
    t_range = torch.arange(0, 10, step=2)

    print(f"Tensor from list:\n{t_list}")
    print(f"  Shape: {t_list.shape} | Dtype: {t_list.dtype} | Device: {t_list.device}")
    print(f"\nZeros (2x3):\n{t_zeros}")
    print(f"Ones (int32):\n{t_ones}")
    print(f"Uniform Random [0, 1):\n{t_rand}")
    print(f"Arange (0 to 10 step 2): {t_range}")


def demo_reshaping_and_slicing():
    print("\n" + "=" * 50)
    print(" 2. RESHAPING, SLICING & TRANSPOSITION")
    print("=" * 50)

    base = torch.arange(12, dtype=torch.float32)
    print(f"Original 1D vector (length 12): {base}")

    # Reshaping: 12 -> (3, 4)
    mat = base.view(3, 4)
    print(f"\nReshaped into Matrix (3x4):\n{mat}")

    # Slicing: row 0 to 2, col 1 to 3
    sub_slice = mat[0:2, 1:3]
    print(f"\nSub-matrix slice (rows 0-1, cols 1-2):\n{sub_slice}")

    # Transposition: (3, 4) -> (4, 3)
    transposed = mat.t()
    print(f"\nTransposed Matrix (4x3):\n{transposed}")

    # Squeeze & Unsqueeze (adding and removing singleton dimensions)
    expanded = mat.unsqueeze(0)  # Shape: (1, 3, 4)
    squeezed = expanded.squeeze(0)  # Shape: (3, 4)
    print(f"\nUnsqueeze dim 0: {expanded.shape}")
    print(f"Squeeze dim 0  : {squeezed.shape}")


def demo_math_and_linalg():
    print("\n" + "=" * 50)
    print(" 3. MATHEMATICAL & MATRIX OPERATIONS")
    print("=" * 50)

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    print(f"Tensor A:\n{a}")
    print(f"Tensor B:\n{b}")

    # Element-wise operations
    print(f"\nElement-wise Addition (A + B):\n{a + b}")
    print(f"Element-wise Multiplication (A * B):\n{a * b}")

    # Matrix multiplication: (2, 2) @ (2, 2)
    matmul_res = torch.matmul(a, b)
    print(f"\nMatrix Multiplication (A @ B):\n{matmul_res}")

    # Aggregations
    print(f"\nStatistics on Matrix A:")
    print(f"  Sum  : {a.sum().item():.2f}")
    print(f"  Mean : {a.mean().item():.2f}")
    print(f"  Max  : {a.max().item():.2f}")
    print(f"  Std  : {a.std().item():.2f}")

    # Broadcasting demonstration
    v = torch.tensor([10.0, 20.0])  # Shape (2,)
    broadcast_res = a + v            # Broadcast across rows of A
    print(f"\nBroadcasting vector [10, 20] across matrix A:\n{broadcast_res}")


def demo_numpy_and_device():
    print("\n" + "=" * 50)
    print(" 4. NUMPY BRIDGE & DEVICE ACCELERATION")
    print("=" * 50)

    # NumPy -> PyTorch
    np_arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    torch_from_np = torch.from_numpy(np_arr)
    print(f"NumPy Array     : {np_arr}")
    print(f"PyTorch Tensor  : {torch_from_np}")

    # PyTorch -> NumPy (shares underlying memory on CPU)
    np_back = torch_from_np.numpy()
    torch_from_np[0] = 99.0
    print(f"\nAfter modifying PyTorch tensor in-place:")
    print(f"  NumPy Array updated via shared memory: {np_back}")

    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_device = torch.ones(3, device=device)
    print(f"\nTarget Compute Device: {device}")
    print(f"Tensor on device    : {t_device} (device={t_device.device})")


def main():
    print("=" * 60)
    print("  LAB 6: TENSOR MANIPULATION & PYTORCH FUNDAMENTALS")
    print("=" * 60)

    demo_tensor_creation()
    demo_reshaping_and_slicing()
    demo_math_and_linalg()
    demo_numpy_and_device()

    print("\n" + "=" * 60)
    print(" Tensor manipulation module executed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
