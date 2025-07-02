import torch
import scipy
import numpy as np
from .network_utils import rsolve, nudge_eig, nudge_svd, s2z, z2s, s2y, y2s

def rsolve_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    r"""Solves X @ A = B using PyTorch.
    Same as B @ torch.linalg.inv(A) but avoids calculating the inverse and
    should be numerically slightly more accurate.
    """
    return torch.linalg.solve(A.transpose(-2, -1).conj(), B.transpose(-2, -1).conj()).transpose(-2, -1).conj()

def nudge_eig_torch(mat: torch.Tensor, cond: float = 1e-9, min_eig: float = 1e-12) -> torch.Tensor:
    r"""Nudge eigenvalues with absolute value smaller than
    max(cond * max(|eigenvalue|), min_eig) to that value.
    Can be used to avoid singularities in solving matrix equations.
    """
    # Convert to numpy, use network_utils function, convert back to torch
    mat_np = mat.cpu().numpy()
    result_np = nudge_eig(mat_np, cond=cond, min_eig=min_eig)
    return torch.from_numpy(result_np).to(mat.device)

def nudge_svd(mat: torch.Tensor, cond: float = 1e-9, min_svd: float = 1e-12) -> torch.Tensor:
    # Convert to numpy, use network_utils function, convert back to torch
    mat_np = mat.cpu().numpy()
    result_np = nudge_svd(mat_np, cond=cond, min_svd=min_svd)
    return torch.from_numpy(result_np).to(mat.device)

def s2z_torch(s: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    r"""Convert scattering parameters to impedance parameters using PyTorch."""
    # Convert to numpy, use network_utils function, convert back to torch
    s_np = s.cpu().numpy()
    z0_np = z0.cpu().numpy()
    z_np = s2z(s_np, z0_np)
    return torch.from_numpy(z_np).to(s.device)

def z2s_torch(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    r"""Convert impedance parameters to scattering parameters using PyTorch."""
    # Convert to numpy, use network_utils function, convert back to torch
    z_np = z.cpu().numpy()
    z0_np = z0.cpu().numpy()
    s_np = z2s(z_np, z0_np)
    return torch.from_numpy(s_np).to(z.device)

def s2y_torch(s: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    r"""Convert impedance parameters to scattering parameters using PyTorch."""
    # Convert to numpy, use network_utils function, convert back to torch
    s_np = s.cpu().numpy()
    z0_np = z0.cpu().numpy()
    y_np = s2y(s_np, z0_np)
    return torch.from_numpy(y_np).to(s.device)

def y2s_torch(y: torch.Tensor, z0: torch.Tensor, epsilon=1e-4) -> torch.Tensor:
    r"""Convert impedance parameters to scattering parameters using PyTorch."""
    # Convert to numpy, use network_utils function, convert back to torch
    y_np = y.cpu().numpy()
    z0_np = z0.cpu().numpy()
    s_np = y2s(y_np, z0_np, epsilon=epsilon)
    return torch.from_numpy(s_np).to(y.device)