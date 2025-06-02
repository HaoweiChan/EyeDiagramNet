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
    loss = (torch.log1p(inputs) - torch.log1p(targets)) ** 2
    return _reduce_loss(loss, mask)

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