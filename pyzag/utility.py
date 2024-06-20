"""Random helper functions"""

import torch


def mbmm(A1, A2):
    """
    Batched matrix-matrix multiplication with several batch dimensions
    """
    return torch.einsum("...ik,...kj->...ij", A1, A2)
