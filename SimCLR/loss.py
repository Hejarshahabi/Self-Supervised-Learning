# simclr/loss.py
import torch
import torch.nn.functional as F
from typing import Tuple


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent).
    z1, z2: (N, D) projected features
    """
    device = z1.device
    batch_size = z1.shape[0]

    # L2-normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate representations
    representations = torch.cat([z1, z2], dim=0)          # (2N, D)
    similarity_matrix = torch.matmul(representations, representations.T) / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    # Labels: positives are at positions (i, i+N) and (i+N, i)
    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels + batch_size, labels])

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss