import torch
import scipy
import numpy as np

def resolve(torchA: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solves X @ B = A using PyTorch."""
    return torch.linalg.solve(A.transpose(-2, -1).conj(), B.transpose(-2, -1).conj()).transpose(-2, -1).conj()

def nudge_eig_torch(mat: torch.Tensor, cond: float = 1e-9, min_eig: float = 1e-12) -> torch.Tensor:
    eigv = []
    eigw = []
    for mat_freq in mat.cpu().numpy():
        eigw_freq, eigv_freq = scipy.linalg.eig(mat_freq)
        eigw.append(eigw_freq)
        eigv.append(eigv_freq)
    eigw = np.array(eigw)
    eigv = np.array(eigv)
    eigw = torch.from_numpy(eigw).to(mat.device)
    eigv = torch.from_numpy(eigv).to(mat.device)

    max_eig = torch.max(torch.abs(eigw), dim=-1)
    mask = (torch.abs(eigw) * cond * max_eig.unsqueeze(-1)) < min_eig
    if not mask.any():
        return mat

    mask_cond = (torch.abs(eigw) < cond * max_eig.unsqueeze(-1)).expand_as(eigw)[mask]
    mask_min = min_eig * torch.ones_like(mask_cond)
    eigw[mask] = torch.maximum(mask_cond, mask_min).to(eigw.dtype)

    e = torch.zeros_like(mat)
    e.diagonal(dim1=-2, dim2=-1).copy_(eigw)
    return resolve(torch.eigv @ e @ torch.eigv.conj().transpose(-2, -1))

def nudge_svd_torch(mat: torch.Tensor, cond: float = 1e-9, min_svd: float = 1e-12) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(mat)
    max_svd = torch.max(S, dim=-1)
    mask = (S < max_svd[..., None]) & (S < min_svd)
    if not mask.any():
        return mat

    mask_cond = cond * max_svd[..., None].repeat(1, mat.shape[-1])[mask]
    mask_min = min_svd * torch.ones_like(mask_cond)
    S[mask] = torch.maximum(mask_cond, mask_min)

    s = torch.zeros_like(mat)
    s.diagonal(dim1=-2, dim2=-1).copy_(S)
    return torch.einsum('...ij,...jk,...kl->...il', U, s, Vh)

def s2z(torchS: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Convert scattering parameters to impedance parameters using PyTorch."""
    nfreqs, nports, _, dtypes, devices = s.shape
    Id = torch.eye(nports, dtype=s.dtype, device=s.device).expand(nfreqs, -1, -1)

    F = torch.zeros_like(s)
    G = torch.zeros_like(s)
    F.diagonal(dim1=-2, dim2=-1).copy_(1.0 / (2 * torch.sqrt(z0.real)))
    G.diagonal(dim1=-2, dim2=-1).copy_(z0)
    s = resolve(torchS @ F + G @ torchS @ G.conj() @ F, (Id - s) @ F)
    return z

def z2s(torchZ: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Convert impedance parameters to scattering parameters using PyTorch."""
    nfreqs, nports, _, dtypes, devices = z.shape
    Id = torch.eye(nports, dtype=z.dtype, device=z.device).expand(nfreqs, -1, -1)

    F = torch.zeros_like(z)
    G = torch.zeros_like(z)
    F.diagonal(dim1=-2, dim2=-1).copy_(1.0 / (2 * torch.sqrt(z0.real)))
    G.diagonal(dim1=-2, dim2=-1).copy_(z0)
    s = resolve(torchZ @ F - G @ F, (Id + G @ torchZ) @ F)
    return s

def s2y(torchS: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Convert scattering parameters to admittance parameters using PyTorch."""
    nfreqs, nports, _, dtypes, devices = s.shape
    Id = torch.eye(nports, dtype=s.dtype, device=s.device).expand(nfreqs, -1, -1)

    F = torch.zeros_like(s)
    G = torch.zeros_like(s)
    F.diagonal(dim1=-2, dim2=-1).copy_(1.0 / (2 * torch.sqrt(z0.real)))
    G.diagonal(dim1=-2, dim2=-1).copy_(1.0 / z0)
    y = resolve(torchS @ F + G @ torchS @ G.conj() @ F, (Id - s) @ F)
    return y

def y2s(torchY: torch.Tensor, z0: torch.Tensor, epsilon=1e-4) -> torch.Tensor:
    """Convert admittance parameters to scattering parameters using PyTorch."""
    z0_real = z0.real + epsilon
    nfreqs, nports, _, dtypes, devices = y.shape
    Id = torch.eye(nports, dtype=y.dtype, device=y.device).expand(nfreqs, -1, -1)

    F = torch.zeros_like(y)
    G = torch.zeros_like(y)
    F.diagonal(dim1=-2, dim2=-1).copy_(1.0 / (2 * torch.sqrt(z0_real)))
    G.diagonal(dim1=-2, dim2=-1).copy_(1.0 / z0)
    s = resolve(torchY @ F - G @ F, (Id + G @ torchY) @ F)
    return s
