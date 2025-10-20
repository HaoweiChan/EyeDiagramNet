"""
Gaussian Process Models for Eye Width Prediction

Implements exact GP models with custom kernels for parameter hyperspace regression.
"""

import torch
import torch.nn as nn

try:
    import gpytorch
    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False


class ExactGPModel(gpytorch.models.ExactGP if GPYTORCH_AVAILABLE else object):
    """
    Exact Gaussian Process model for eye width prediction in parameter hyperspace.
    
    Uses a combination of RBF and linear kernels to capture both smooth variations
    and linear trends in the parameter space.
    """
    
    def __init__(self, train_x, train_y, likelihood, input_dim, use_ard=True):
        """
        Initialize GP model.
        
        Args:
            train_x: Training inputs [n_samples, n_features]
            train_y: Training outputs [n_samples]
            likelihood: Gaussian likelihood
            input_dim: Number of input features
            use_ard: Use Automatic Relevance Determination (different lengthscale per dimension)
        """
        if not GPYTORCH_AVAILABLE:
            raise ImportError("gpytorch is required. Install with 'pip install gpytorch'")
            
        super().__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Base kernel: RBF with ARD for automatic feature relevance detection
        base_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=input_dim if use_ard else None
        )
        
        # Add linear kernel to capture linear trends
        linear_kernel = gpytorch.kernels.LinearKernel(
            ard_num_dims=input_dim if use_ard else None
        )
        
        # Combine kernels
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel + linear_kernel
        )
    
    def forward(self, x):
        """Forward pass through GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

