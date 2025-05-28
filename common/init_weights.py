import torch
import torch.nn as nn
import torch.nn.init as init

def _normal_init(m):
    """Apply Normal (truncated) init to module m."""
    with torch.no_grad():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.Parameter):
            init.trunc_normal_(m, mean=0.0, std=0.02)
        elif isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.modules.batchnorm._NormBase)):
            init.trunc_normal_(m.weight, mean=1.0, std=0.02)
            init.constant_(m.bias, 0.0)

def _xavier_init(m):
    """Apply Xavier init to module m."""
    with torch.no_grad():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.Parameter):
            init.xavier_normal_(m, mean=0.0, std=0.02)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.modules.batchnorm._NormBase)):
            init.trunc_normal_(m.weight, mean=1.0, std=0.02)
            init.constant_(m.bias, 0.0)

def _kaiming_init(m):
    """Apply Kaiming (He) init to module m."""
    with torch.no_grad():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_normal_(
                m.weight, a=0, mode='fan_out', nonlinearity='relu'
            )
            if m.bias is not None:
                init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.Parameter):
            init.kaiming_normal_(
                m, a=0, mode='fan_out', nonlinearity='relu'
            )
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(
                m.weight, a=0, mode='fan_out', nonlinearity='relu'
            )
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.modules.batchnorm._NormBase)):
            init.trunc_normal_(m.weight, mean=1.0, std=0.02)
            init.constant_(m.bias, 0.0)

def _orthogonal_init(m):
    """Apply Orthogonal init to module m."""
    with torch.no_grad():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.Parameter):
            init.orthogonal_(m, gain=1.0)
        elif isinstance(m, nn.Linear):
            init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.modules.batchnorm._NormBase)):
            init.trunc_normal_(m.weight, mean=1.0, std=0.02)
            init.constant_(m.bias, 0.0)

# Map strings to their respective init functions
INIT_METHODS = {
    'normal':   _normal_init,
    'xavier':   _xavier_init,
    'kaiming':  _kaiming_init,
    'orthogonal': _orthogonal_init,
}

def init_weights(init_type='xavier'):
    """
    Returns a function that can be passed to net.apply(...).
    The returned function will apply the chosen initialization
    to each submodule m in the network.
    """
    if init_type not in INIT_METHODS:
        raise NotImplementedError(f'Initialization type "{init_type}" is not supported.')

    chosen_init = INIT_METHODS[init_type]

    def _init_weights(m):
        chosen_init(m)

    return _init_weights