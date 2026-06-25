import numpy as np
import torch

def realignment(matrix):
    """
    Perform the realignment operation on a given matrix.
    Args:
        matrix (torch.Tensor): A 2D tensor representing the density matrix.
    Returns:
        torch.Tensor: The realigned matrix.
    """
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix.")
    
    dim = int(np.sqrt(matrix.shape[0]))
    if dim * dim != matrix.shape[0]:
        raise ValueError("Matrix dimensions must be a perfect square.")

    realigned = torch.zeros_like(matrix)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    realigned[i * dim + k, j * dim + l] = matrix[i * dim + j, k * dim + l]
    return realigned

def is_entangled(matrix):
    """
    Check if a given density matrix is entangled using the realignment criterion.
    Args:
        matrix (torch.Tensor): A 2D tensor representing the density matrix.
    Returns:
        bool: True if the matrix is entangled, False otherwise.
    """
    realigned_matrix = realignment(matrix)
    trace_norm = torch.linalg.norm(realigned_matrix, ord='nuc')  # Nuclear norm
    return trace_norm > 1

if __name__ == '__main__':
    # Example: Test with a dummy density matrix
    dim = 4  # 2x2 system
    dummy_matrix = torch.rand((dim, dim), dtype=torch.float64)
    dummy_matrix = dummy_matrix @ dummy_matrix.T  # Make it positive semi-definite
    dummy_matrix /= torch.trace(dummy_matrix)  # Normalize to make it a density matrix

    print("Input Density Matrix:")
    print(dummy_matrix)

    entangled = is_entangled(dummy_matrix)
    if entangled:
        print("The given density matrix is entangled.")
    else:
        print("The given density matrix is separable.")