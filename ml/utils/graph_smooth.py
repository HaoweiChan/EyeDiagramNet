import torch
from typing import Tuple

__all__ = [
    "knn_graph",
    "graph_laplacian_penalty",
]


def knn_graph(theta: torch.Tensor, k: int = 8, sigma: float = 0.25) -> Tuple[torch.LongTensor, torch.Tensor]:
    """
    Build a k-NN graph over theta points using a Gaussian kernel.

    Args:
        theta: Tensor of shape (N, D) or (B, N, D) containing parameter points.
        k: Number of nearest neighbours per node (excluding self).
        sigma: Gaussian kernel width for edge weights. Must be > 0.

    Returns:
        idx: LongTensor of neighbour indices with shape (N, k) or (B, N, k).
        w:   Tensor of edge weights with shape (N, k) or (B, N, k).

    Notes:
        - If N <= 1 or k <= 0, returns empty neighbour lists (k = 0).
        - Computation is batched when theta is 3D.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Normalize input to (B, N, D)
    squeeze_batch = False
    if theta.dim() == 2:
        theta = theta.unsqueeze(0)
        squeeze_batch = True
    elif theta.dim() != 3:
        raise ValueError(f"theta must have shape (N, D) or (B, N, D); got {tuple(theta.shape)}")

    batch_size, num_points, _ = theta.shape
    if num_points <= 1 or k <= 0:
        empty_idx = torch.empty(batch_size, num_points, 0, dtype=torch.long, device=theta.device)
        empty_w = torch.empty(batch_size, num_points, 0, dtype=theta.dtype, device=theta.device)
        if squeeze_batch:
            return empty_idx.squeeze(0), empty_w.squeeze(0)
        return empty_idx, empty_w

    k_eff = int(max(0, min(k, num_points - 1)))

    # Pairwise squared distances per batch: (B, N, N)
    d2 = torch.cdist(theta, theta, p=2).pow(2)

    # Exclude self by adding a large value to diagonal
    eye = torch.eye(num_points, dtype=d2.dtype, device=theta.device).unsqueeze(0)
    d2 = d2 + eye * 1e9

    # Top-k nearest (smallest distances)
    vals, idx = torch.topk(d2, k=k_eff, dim=-1, largest=False)

    # Gaussian weights
    denom = 2.0 * (sigma ** 2)
    w = torch.exp(-vals / denom)

    if squeeze_batch:
        return idx.squeeze(0), w.squeeze(0)
    return idx, w


def graph_laplacian_penalty(y: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Compute Laplacian smoothness penalty over a k-NN graph.

    Penalizes variation across edges: mean_{i,j in N(i)} w_ij * ||y_i - y_j||^2.

    Args:
        y:  Predictions at nodes with shape (N,), (N, *), (B, N) or (B, N, *).
        idx: Neighbour indices with shape (N, k) or (B, N, k).
        w:  Edge weights matching idx shape.

    Returns:
        Scalar tensor with the smoothness penalty.
    """
    if idx.numel() == 0:
        return y.new_tensor(0.0)

    # Standardize shapes to (B, N, k) for idx/w and (B, N, *) for y
    if idx.dim() == 2:
        # (N, k)
        idx = idx.unsqueeze(0)
        w = w.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0).unsqueeze(-1)
        elif y.dim() >= 2 and y.size(0) == idx.size(1):
            y = y.unsqueeze(0)
        else:
            raise ValueError("y must have leading N matching idx when idx is (N, k)")
    elif idx.dim() == 3:
        # (B, N, k)
        bsz, n_pts, _ = idx.shape
        if y.dim() == 1:
            # Broadcast single (N,) across batch
            if y.size(0) != n_pts:
                raise ValueError("y with shape (N,) must match idx N dimension")
            y = y.unsqueeze(0).unsqueeze(-1).expand(bsz, n_pts, 1)
        elif y.dim() == 2:
            # Either (B, N) or (N, F)
            if y.size(0) == bsz and y.size(1) == n_pts:
                y = y.unsqueeze(-1)
            elif y.size(0) == n_pts:
                y = y.unsqueeze(0).expand(bsz, n_pts, y.size(1))
            else:
                raise ValueError("y must be (B, N) or (N, F) to align with idx (B, N, k)")
        elif y.dim() >= 3:
            # Either (B, N, *) or (N, *)
            if y.size(0) == bsz and y.size(1) == n_pts:
                pass
            elif y.size(0) == n_pts:
                y = y.unsqueeze(0).expand(bsz, *y.shape)
            else:
                raise ValueError("y must have leading dims (B, N, ...) or (N, ...)")
    else:
        raise ValueError("idx must be (N, k) or (B, N, k)")

    # Shapes now: idx, w -> (B, N, k); y -> (B, N, *)
    bsz, n_pts, k = idx.shape

    if k == 0 or n_pts <= 1:
        return y.new_tensor(0.0)

    # Expand y to include neighbour dimension so gather has matching dims
    y_expanded = y.unsqueeze(2).expand(bsz, n_pts, k, *y.shape[2:])

    # Gather neighbour predictions along node dimension (dim=1)
    expand_index = idx.unsqueeze(-1).expand(bsz, n_pts, k, *y.shape[2:])
    y_neighbors = torch.gather(y_expanded, dim=1, index=expand_index)

    y_center = y.unsqueeze(2)
    diff2 = (y_center - y_neighbors).pow(2)

    # Reduce trailing feature dims if present
    while diff2.dim() > 3:
        diff2 = diff2.mean(dim=-1)

    penalty = (w * diff2).mean()
    return penalty
