# swav/loss.py
import torch
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(scores, epsilon=0.05, n_iter=3):
    """
    Online Sinkhorn-Knopp for prototype assignment.
    scores: (B, K) logits
    """
    Q = torch.exp(scores / epsilon).t()  # (K, B)
    B = Q.shape[1]
    K = Q.shape[0]

    # Normalize
    Q /= Q.sum()

    for _ in range(n_iter):
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= K
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()  # (B, K)


def swav_loss(z_list, model, temperature=0.1, queue=None):
    """
    z_list: list of 6 normalized embeddings (B, D)
    model: SwAV model with prototypes
    queue: (Q, D) or None
    """
    loss = 0.0
    n_crops = len(z_list)

    for i, z in enumerate(z_list):
        scores = torch.matmul(z, model.prototypes.t()) / temperature  # (B, K)

        with torch.no_grad():
            if queue is not None:
                queue_scores = torch.matmul(z, queue.t()) / temperature
                scores = torch.cat([scores, queue_scores], dim=1)
            target = sinkhorn_knopp(scores)

        loss += F.cross_entropy(scores, target)

    return loss / n_crops