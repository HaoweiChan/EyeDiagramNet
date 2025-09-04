import torch
import torch.nn as nn
import torch.nn.functional as F


def _reduce_loss(loss, mask):
    """
    Reduce loss tensor with optional mask weighting.
    
    Args:
        loss: Loss tensor to reduce
        mask: Optional mask tensor for weighting (None for uniform weighting)
        
    Returns:
        Scalar reduced loss value
    """
    if mask is None:
        return torch.mean(loss)
    mask_max = mask.view(mask.size(0), -1).amax(-1).view(-1, *(1,) * (mask.dim()-1))
    loss = torch.sum(loss * mask) / torch.sum(mask / (mask_max + 1e-8))
    return loss

def _softmin(x, tau=0.1, dim=None):
    """
    Differentiable softmin function using log-sum-exp.
    
    Args:
        x: Input tensor
        tau: Temperature parameter. Smaller tau makes the function closer to the true min
        dim: The dimension to reduce. If None, reduces all dimensions
        
    Returns:
        The soft minimum of the tensor
    """
    return -tau * torch.logsumexp(-x / tau, dim=dim)

def weighted_mse_loss(inputs, targets, mask=None):
    """
    Weighted mean squared error loss.
    
    Args:
        inputs: Predicted values
        targets: Ground truth values
        mask: Optional mask for weighting specific elements
        
    Returns:
        Scalar loss value
    """
    loss = (inputs - targets) ** 2
    return _reduce_loss(loss, mask)

def weighted_l1_loss(inputs, targets, mask=None):
    """
    Weighted L1 (Mean Absolute Error) loss.
    
    Args:
        inputs: Predicted values
        targets: Ground truth values
        mask: Optional mask for weighting specific elements
        
    Returns:
        Scalar loss value
    """
    loss = F.l1_loss(inputs, targets, reduction='none')
    return _reduce_loss(loss, mask)

def weighted_focal_mse_loss(inputs, targets, mask=None, activate='sigmoid', beta=.2, gamma=1):
    """
    Focal MSE loss that emphasizes hard examples by modulating the standard MSE loss.
    
    Args:
        inputs: Predicted values
        targets: Ground truth values
        mask: Optional mask for weighting specific elements
        activate: Activation function for focal weighting ('sigmoid' or 'tanh')
        beta: Shift parameter for the activation function
        gamma: Exponent for focal weighting
        
    Returns:
        Scalar loss value
    """
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta + torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta + torch.abs(inputs - targets)) - 1) ** gamma
    return _reduce_loss(loss, mask)

def weighted_focal_l1_loss(inputs, targets, mask=None, activate='sigmoid', beta=.2, gamma=1):
    """
    Focal L1 loss that emphasizes hard examples by modulating the standard L1 loss.
    
    Args:
        inputs: Predicted values
        targets: Ground truth values
        mask: Optional mask for weighting specific elements
        activate: Activation function for focal weighting ('sigmoid' or 'tanh')
        beta: Shift parameter for the activation function
        gamma: Exponent for focal weighting
        
    Returns:
        Scalar loss value
    """
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta + torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta + torch.abs(inputs - targets)) - 1) ** gamma
    return _reduce_loss(loss, mask)

def weighted_huber_loss(inputs, targets, mask=None, beta=1.):
    """
    Huber loss that combines the properties of L1 and L2 losses.
    
    Args:
        inputs: Predicted values
        targets: Ground truth values
        mask: Optional mask for weighting specific elements
        beta: Threshold parameter determining the transition point
        
    Returns:
        Scalar loss value
    """
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    return _reduce_loss(loss, mask)

def weighted_berhu_loss(inputs, targets, mask=None, beta=1.):
    """
    Reverse Huber (berHu) loss combining L1 and L2 losses in opposite manner to Huber.
    
    Args:
        inputs: Predicted values
        targets: Ground truth values
        mask: Optional mask for weighting specific elements
        beta: Threshold parameter determining the transition point
        
    Returns:
        Scalar loss value
    """
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, l1_loss, 0.5 * (l1_loss ** 2 + beta ** 2) / beta)
    return _reduce_loss(loss, mask)

def weighted_log_cosh_loss(inputs, targets, mask=None, beta=1.):
    """
    Log-cosh loss that works like L2 for small errors and L1 for large errors.
    
    Args:
        inputs: Predicted values
        targets: Ground truth values
        mask: Optional mask for weighting specific elements
        beta: Scale parameter (currently unused for compatibility)
        
    Returns:
        Scalar loss value
    """
    diff = inputs - targets
    loss = torch.log(torch.exp(diff) + torch.exp(-diff)) / 2
    return _reduce_loss(loss, mask)

def weighted_scale_invariant_loss(inputs, targets, mask=None):
    """
    Scale-invariant loss for depth/scale estimation tasks.
    
    Args:
        inputs: Predicted values (must be positive)
        targets: Ground truth values (must be positive)
        mask: Optional mask for weighting specific elements
        
    Returns:
        Scalar loss value
    """
    d = (torch.log(inputs) - torch.log(targets))
    return _reduce_loss(d ** 2, mask) - 0.5 * _reduce_loss(d, mask) ** 2

def ale_loss(inputs, targets, mask=None, gamma=2.):
    """
    Asymmetric Loss Error (ALE) that penalizes underestimation more than overestimation.
    
    Args:
        inputs: Predicted values
        targets: Ground truth values
        mask: Optional mask for weighting specific elements
        gamma: Asymmetry parameter (gamma > 1 penalizes underestimation more)
        
    Returns:
        Scalar loss value
    """
    loss = inputs - targets
    loss = torch.maximum(loss / gamma, loss * gamma)
    return _reduce_loss(loss, mask)

def rale_loss(inputs, targets, mask=None, gamma=2.0):
    """
    Reverse Asymmetric Loss Error (RALE) that penalizes overestimation more than underestimation.
    
    Args:
        inputs: Predicted values
        targets: Ground truth values
        mask: Optional mask for weighting specific elements
        gamma: Asymmetry parameter (2.0 provides balanced asymmetry, gamma > 1 penalizes overestimation more)
        
    Returns:
        Scalar loss value
    """
    loss = inputs - targets
    loss = torch.maximum(loss / gamma, -loss * gamma)
    return _reduce_loss(loss, mask)

def correlation_loss(inputs, targets, mask=None):
    """
    Correlation loss that measures the correlation between predicted and target values.
    
    Args:
        inputs: Predicted values
        targets: Ground truth values
        mask: Optional mask for weighting specific elements
        
    Returns:
        Scalar loss value (1 - correlation, lower is better)
    """
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

def smooth_gaussian_focus_loss(inputs, targets, mask=None, focus_weight=2.0, sigma=0.5, bottom_percentile=0.1):
    """
    Smooth Gaussian-weighted loss that focuses on minimum regions without sharp gradients.
    
    Args:
        inputs: Predicted values (B, L)
        targets: Ground truth values (B, L)
        mask: Optional mask for weighting specific elements
        focus_weight: Maximum additional weight for minimum regions (2.0 gives moderate focus)
        sigma: Standard deviation for Gaussian weighting (0.5 provides moderate smoothing)
        bottom_percentile: Focus on bottom X% of values rather than absolute minimum (0.1 = bottom 10%)
        
    Returns:
        Smoothly weighted loss that focuses on minimum regions without sharp gradients
    """
    B, L = targets.shape
    
    # Use percentile-based approach for smoother focus regions
    target_percentile = torch.quantile(targets, bottom_percentile, dim=1, keepdim=True)  # (B, 1)
    
    # Smooth Gaussian weighting instead of sharp exponential
    distance_to_focus = (targets - target_percentile).abs()  # (B, L)
    gaussian_weights = torch.exp(-(distance_to_focus ** 2) / (2 * sigma ** 2))  # (B, L)
    
    # Gentler focus weighting
    weights = 1.0 + (focus_weight - 1.0) * gaussian_weights  # (B, L)
    weights = weights / weights.mean(dim=1, keepdim=True)  # Normalize
    
    # Weighted smooth L1 loss (less sensitive to outliers than MSE)
    weighted_loss = weights * F.smooth_l1_loss(inputs, targets, reduction='none')
    return _reduce_loss(weighted_loss, mask)

def min_focused_loss(inputs, targets, mask=None, alpha=0.5, tau_min=0.1):
    """
    Blended loss combining minimum value accuracy with overall correlation.
    
    Args:
        inputs: Predicted values (B, L)
        targets: Ground truth values (B, L)
        mask: Optional mask for weighting specific elements
        alpha: Weight for minimum accuracy vs overall correlation (0.5 = balanced, 0-1 range)
        tau_min: Temperature parameter for softmin function
        
    Returns:
        Blended loss focusing on minimum accuracy with overall shape preservation
    """
    # Minimum accuracy component
    # pred_min = _softmin(inputs, tau=tau_min, dim=1)  # (B,)
    # true_min = torch.min(targets, dim=1).values      # (B,)
    # min_loss_values = F.smooth_l1_loss(pred_min, true_min, reduction='none')
    mse_loss_values = F.mse_loss(inputs, targets, reduction='none').mean(dim=1)
    
    # Overall correlation component  
    correlation_values = 1.0 - F.cosine_similarity(inputs, targets, dim=1)
    
    # Blend the two
    # combined_loss = alpha * min_loss_values + (1 - alpha) * correlation_values
    combined_loss = alpha * mse_loss_values + (1 - alpha) * correlation_values
    return _reduce_loss(combined_loss, mask)