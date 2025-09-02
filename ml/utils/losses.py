import torch
import torch.nn as nn
import torch.nn.functional as F


def _reduce_loss(loss, mask):
    if mask is None:
        return torch.mean(loss)
    mask_max = mask.view(mask.size(0), -1).amax(-1).view(-1, *(1,) * (mask.dim()-1))
    loss = torch.sum(loss * mask) / torch.sum(mask / (mask_max + 1e-8))
    return loss

def weighted_mse_loss(inputs, targets, mask=None):
    loss = (inputs - targets) ** 2
    return _reduce_loss(loss, mask)

def weighted_l1_loss(inputs, targets, mask=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    return _reduce_loss(loss, mask)

def weighted_focal_mse_loss(inputs, targets, mask=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta + torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta + torch.abs(inputs - targets)) - 1) ** gamma
    return _reduce_loss(loss, mask)

def weighted_focal_l1_loss(inputs, targets, mask=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta + torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta + torch.abs(inputs - targets)) - 1) ** gamma
    return _reduce_loss(loss, mask)

def weighted_huber_loss(inputs, targets, mask=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    return _reduce_loss(loss, mask)

def weighted_berhu_loss(inputs, targets, mask=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, l1_loss, 0.5 * (l1_loss ** 2 + beta ** 2) / beta)
    return _reduce_loss(loss, mask)

def weighted_log_cosh_loss(inputs, targets, mask=None, beta=1.):
    diff = inputs - targets
    loss = torch.log(torch.exp(diff) + torch.exp(-diff)) / 2
    return _reduce_loss(loss, mask)

def weighted_scale_invariant_loss(inputs, targets, mask=None):
    d = (torch.log(inputs) - torch.log(targets))
    return _reduce_loss(d ** 2, mask) - 0.5 * _reduce_loss(d, mask) ** 2

def ale_loss(inputs, targets, mask=None, gamma=2.):
    loss = inputs - targets
    loss = torch.maximum(loss / gamma, loss * gamma)
    return _reduce_loss(loss, mask)

def rale_loss(inputs, targets, mask=None, gamma=1.2):
    loss = inputs - targets
    loss = torch.maximum(loss / gamma, -loss * gamma)
    return _reduce_loss(loss, mask)

def correlation_loss(inputs, targets, mask=None):
    inputs_mean = torch.mean(inputs, dim=1, keepdim=True)
    targets_mean = torch.mean(targets, dim=1, keepdim=True)

    inputs_centered = inputs - inputs_mean
    targets_centered = targets - targets_mean

    covariance = torch.mean(inputs_centered * targets_centered, dim=1)
    inputs_std = torch.std(inputs, dim=1)
    targets_std = torch.std(targets, dim=1)

    correlation = covariance / (inputs_std * targets_std + 1e-8)
    loss = 1 - correlation
    return _reduce_loss(loss, mask)

def gaussian_nll_loss(inputs, targets, logvar, mask=None):
    """
    Gaussian negative log-likelihood loss with learnable variance.
    
    Args:
        inputs: Predicted values
        targets: Ground truth values  
        logvar: Log variance (log of variance, not log of std)
        mask: Optional mask for valid predictions
        
    Returns:
        Reduced loss value
    """
    precision = torch.exp(-logvar)
    loss = 0.5 * (logvar + precision * (inputs - targets).pow(2))

    return _reduce_loss(loss, mask)

def logit(p, eps=1e-6):
    """Numerically stable logit function."""
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)

def softmin(x, tau=0.1, dim=None):
    """
    Differentiable softmin function using log-sum-exp.
    
    Args:
        x: Input tensor.
        tau: Temperature parameter. Smaller tau makes the function closer to the true min.
        dim: The dimension to reduce. If None, reduces all dimensions.
        
    Returns:
        The soft minimum of the tensor.
    """
    return -tau * torch.logsumexp(-x / tau, dim=dim)

def focus_weighted_eye_width_loss(pred_ew, true_ew, focus_weight=5.0, tau=0.1):
    """
    Unified loss that combines trace-level and minimum-focused objectives.
    
    Args:
        pred_ew: Predicted eye widths (B, L)
        true_ew: True eye widths (B, L) 
        focus_weight: How much more to weight positions near the minimum
        tau: Temperature for soft attention
        
    Returns:
        Combined loss that focuses more on minimum regions while maintaining trace accuracy
    """
    # Compute soft attention weights based on proximity to true minimum
    true_min = torch.min(true_ew, dim=1, keepdim=True).values  # (B, 1)
    proximity_to_min = torch.exp(-(true_ew - true_min).abs() / tau)  # (B, L)
    
    # Create focus weights: higher weight for positions closer to minimum
    weights = 1.0 + (focus_weight - 1.0) * proximity_to_min  # (B, L)
    weights = weights / weights.mean(dim=1, keepdim=True)  # Normalize
    
    # Weighted MSE loss
    weighted_mse = weights * (pred_ew - true_ew).pow(2)
    return weighted_mse.mean()

def smooth_gaussian_focus_loss(pred_ew, true_ew, focus_weight=3.0, sigma=0.3, bottom_percentile=0.2):
    """
    Smooth Gaussian min-focus loss that avoids gradient spikes.
    
    Args:
        pred_ew: Predicted eye widths (B, L)
        true_ew: True eye widths (B, L) 
        focus_weight: Maximum additional weight for minimum regions (reduced from 5.0 to 3.0)
        sigma: Standard deviation for Gaussian weighting (broader than exponential tau)
        bottom_percentile: Focus on bottom X% of values rather than just absolute minimum
        
    Returns:
        Smoothly weighted loss that focuses on minimum regions without sharp gradients
    """
    B, L = true_ew.shape
    
    # Use percentile-based approach for smoother focus regions
    true_percentile = torch.quantile(true_ew, bottom_percentile, dim=1, keepdim=True)  # (B, 1)
    
    # Smooth Gaussian weighting instead of sharp exponential
    distance_to_focus = (true_ew - true_percentile).abs()  # (B, L)
    gaussian_weights = torch.exp(-(distance_to_focus ** 2) / (2 * sigma ** 2))  # (B, L)
    
    # Gentler focus weighting
    weights = 1.0 + (focus_weight - 1.0) * gaussian_weights  # (B, L)
    weights = weights / weights.mean(dim=1, keepdim=True)  # Normalize
    
    # Weighted smooth L1 loss (less sensitive to outliers than MSE)
    weighted_loss = weights * F.smooth_l1_loss(pred_ew, true_ew, reduction='none')
    return weighted_loss.mean()

def min_focused_loss(pred_ew, true_ew, alpha=0.7, tau_min=0.1):
    """
    Alternative unified loss: blend of softmin accuracy and overall correlation.
    
    Args:
        pred_ew: Predicted eye widths (B, L)
        true_ew: True eye widths (B, L)
        alpha: Weight for minimum accuracy vs overall correlation (0-1)
        tau_min: Temperature for softmin
        
    Returns:
        Blended loss focusing on minimum accuracy with overall shape preservation
    """
    # Minimum accuracy component
    pred_min = softmin(pred_ew, tau=tau_min, dim=1)  # (B,)
    true_min = torch.min(true_ew, dim=1).values      # (B,)
    min_loss = F.smooth_l1_loss(pred_min, true_min)
    
    # Overall correlation component  
    correlation_loss = 1.0 - F.cosine_similarity(pred_ew, true_ew, dim=1).mean()
    
    # Blend the two
    return alpha * min_loss + (1 - alpha) * correlation_loss