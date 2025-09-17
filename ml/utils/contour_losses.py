"""
Multi-Component Loss Functions for Contour Training

Implements various loss components for training smooth, continuous contour predictors:
- Base regression loss (MSE, Gaussian NLL)
- Consistency loss (small input changes -> small output changes)  
- Gradient penalty (bounded derivatives for smoothness)
- Specification band classification (sharp decisions near thresholds)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List
import numpy as np

from ..data.variable_registry import VariableRegistry


class ContourLossFunction(nn.Module):
    """Combined loss function for smooth contour prediction."""
    
    def __init__(
        self,
        variable_registry: Optional[VariableRegistry] = None,
        lambda_consistency: float = 0.1,
        lambda_gradient: float = 0.01,
        lambda_spec_band: float = 0.05,
        consistency_noise_scale: float = 0.02,  # 2% of variable range
        gradient_penalty_type: str = "l2",  # "l1", "l2", or "spectral"
        spec_threshold: Optional[float] = None,  # Eye width specification threshold
        spec_band_width: float = 0.1,  # Band width around threshold for classification
        use_focal_loss: bool = True,  # Use focal loss for spec band classification
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.registry = variable_registry or VariableRegistry()
        self.lambda_consistency = lambda_consistency
        self.lambda_gradient = lambda_gradient
        self.lambda_spec_band = lambda_spec_band
        self.consistency_noise_scale = consistency_noise_scale
        self.gradient_penalty_type = gradient_penalty_type
        self.spec_threshold = spec_threshold
        self.spec_band_width = spec_band_width
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # For tracking loss components
        self.loss_components = {}
    
    def forward(
        self,
        model: nn.Module,
        variables: Dict[str, torch.Tensor],
        sequence_tokens: torch.Tensor,
        targets: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
        active_variables: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute combined loss for contour training.
        
        Args:
            model: ContourPredictor model
            variables: Dict of variable name -> tensor values
            sequence_tokens: Sequence structure tokens
            targets: Ground truth eye width values
            sequence_mask: Optional mask for sequence tokens
            active_variables: List of variables that were perturbed (for consistency loss)
        
        Returns:
            total_loss: Combined loss value
        """
        device = targets.device
        
        # 1. Base regression loss
        if model.predict_uncertainty:
            predictions, uncertainties = model(variables, sequence_tokens, sequence_mask)
            loss_regression = self.gaussian_nll_loss(predictions, targets, uncertainties)
        else:
            predictions = model(variables, sequence_tokens, sequence_mask)
            uncertainties = None
            loss_regression = F.mse_loss(predictions, targets)
        
        self.loss_components['regression'] = loss_regression.item()
        
        # 2. Consistency loss (local smoothness)
        loss_consistency = torch.tensor(0.0, device=device)
        if self.lambda_consistency > 0:
            loss_consistency = self.consistency_loss(
                model, variables, sequence_tokens, sequence_mask, active_variables
            )
            self.loss_components['consistency'] = loss_consistency.item()
        
        # 3. Gradient penalty (bounded derivatives)
        loss_gradient = torch.tensor(0.0, device=device)
        if self.lambda_gradient > 0:
            loss_gradient = self.gradient_penalty(
                model, variables, sequence_tokens, sequence_mask, active_variables
            )
            self.loss_components['gradient'] = loss_gradient.item()
        
        # 4. Specification band classification
        loss_spec_band = torch.tensor(0.0, device=device)
        if self.lambda_spec_band > 0 and self.spec_threshold is not None:
            loss_spec_band = self.spec_band_loss(
                predictions, targets, uncertainties
            )
            self.loss_components['spec_band'] = loss_spec_band.item()
        
        # Combined loss
        total_loss = (
            loss_regression +
            self.lambda_consistency * loss_consistency +
            self.lambda_gradient * loss_gradient +
            self.lambda_spec_band * loss_spec_band
        )
        
        self.loss_components['total'] = total_loss.item()
        
        return total_loss
    
    def gaussian_nll_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """Gaussian negative log-likelihood loss with predicted uncertainty."""
        # Clamp uncertainty to prevent numerical issues
        uncertainties = torch.clamp(uncertainties, min=epsilon)
        
        # NLL: 0.5 * (log(2π) + log(σ²) + (y - μ)²/σ²)
        log_var = torch.log(uncertainties + epsilon)
        mse_term = (predictions - targets) ** 2 / (uncertainties + epsilon)
        
        nll = 0.5 * (np.log(2 * np.pi) + log_var + mse_term)
        
        return nll.mean()
    
    def consistency_loss(
        self,
        model: nn.Module,
        variables: Dict[str, torch.Tensor],
        sequence_tokens: torch.Tensor,
        sequence_mask: Optional[torch.Tensor],
        active_variables: Optional[List[str]]
    ) -> torch.Tensor:
        """
        Consistency loss: penalize large output changes for small input perturbations.
        
        Only perturbs the variables that were actively varied in this batch.
        """
        if active_variables is None:
            active_variables = list(variables.keys())
        
        # Create perturbed version of variables
        perturbed_variables = {}
        for name, value in variables.items():
            if name in active_variables and name in self.registry:
                # Add Gaussian noise scaled to current variable scale (since no bounds)
                var_info = self.registry.get_variable(name)
                # Use a percentage of the current value as noise scale
                value_scale = torch.abs(value).clamp(min=1e-6)
                noise_std = self.consistency_noise_scale * value_scale
                
                noise = torch.randn_like(value) * noise_std
                perturbed_variables[name] = value + noise
            else:
                # Keep inactive variables unchanged
                perturbed_variables[name] = value
        
        # Get predictions for original and perturbed inputs
        with torch.no_grad():
            if model.predict_uncertainty:
                pred_original, _ = model(variables, sequence_tokens, sequence_mask)
                pred_perturbed, _ = model(perturbed_variables, sequence_tokens, sequence_mask)
            else:
                pred_original = model(variables, sequence_tokens, sequence_mask)
                pred_perturbed = model(perturbed_variables, sequence_tokens, sequence_mask)
        
        # MSE between original and perturbed predictions
        consistency_loss = F.mse_loss(pred_original, pred_perturbed)
        
        return consistency_loss
    
    def gradient_penalty(
        self,
        model: nn.Module,
        variables: Dict[str, torch.Tensor],
        sequence_tokens: torch.Tensor,
        sequence_mask: Optional[torch.Tensor],
        active_variables: Optional[List[str]]
    ) -> torch.Tensor:
        """
        Gradient penalty for bounded derivatives (Lipschitz continuity).
        
        Computes random directional derivatives and penalizes large gradients.
        """
        if active_variables is None:
            active_variables = list(variables.keys())
        
        # Prepare variables for gradient computation
        var_tensors = {}
        for name, value in variables.items():
            if name in active_variables:
                var_tensors[name] = value.detach().requires_grad_(True)
            else:
                var_tensors[name] = value.detach()
        
        # Forward pass with gradient tracking
        if model.predict_uncertainty:
            predictions, _ = model(var_tensors, sequence_tokens, sequence_mask)
        else:
            predictions = model(var_tensors, sequence_tokens, sequence_mask)
        
        # Compute gradients w.r.t. active variables
        gradients = []
        for name in active_variables:
            if name in var_tensors and var_tensors[name].requires_grad:
                grad = torch.autograd.grad(
                    outputs=predictions.sum(),
                    inputs=var_tensors[name],
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                gradients.append(grad.view(-1))
        
        if not gradients:
            return torch.tensor(0.0, device=predictions.device)
        
        # Combine gradients
        combined_gradients = torch.cat(gradients, dim=-1)
        
        if self.gradient_penalty_type == "l1":
            penalty = combined_gradients.abs().mean()
        elif self.gradient_penalty_type == "l2":
            penalty = (combined_gradients ** 2).mean()
        elif self.gradient_penalty_type == "spectral":
            # Spectral norm penalty (approximation)
            penalty = torch.max(combined_gradients.abs())
        else:
            penalty = (combined_gradients ** 2).mean()
        
        return penalty
    
    def spec_band_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Specification band classification loss.
        
        Trains a binary classifier on pass/fail relative to specification threshold,
        but only for points near the threshold to avoid interfering with regression.
        """
        # Compute distance from threshold
        target_distance = torch.abs(targets - self.spec_threshold)
        pred_distance = torch.abs(predictions - self.spec_threshold)
        
        # Only apply loss to points within the band around threshold
        in_band_mask = target_distance <= self.spec_band_width
        
        if not in_band_mask.any():
            return torch.tensor(0.0, device=predictions.device)
        
        # Binary labels: 1 if above threshold, 0 if below
        binary_targets = (targets[in_band_mask] > self.spec_threshold).float()
        binary_predictions = (predictions[in_band_mask] > self.spec_threshold).float()
        
        if self.use_focal_loss:
            # Focal loss for handling class imbalance
            ce_loss = F.binary_cross_entropy_with_logits(
                predictions[in_band_mask].squeeze(), binary_targets, reduction='none'
            )
            
            # Focal loss weighting
            pt = torch.exp(-ce_loss)
            alpha_t = self.focal_alpha * binary_targets + (1 - self.focal_alpha) * (1 - binary_targets)
            focal_weight = alpha_t * (1 - pt) ** self.focal_gamma
            
            focal_loss = focal_weight * ce_loss
            return focal_loss.mean()
        else:
            # Standard binary cross-entropy
            return F.binary_cross_entropy(
                binary_predictions, binary_targets
            )
    
    def get_loss_components(self) -> Dict[str, float]:
        """Get the most recent loss component values for logging."""
        return self.loss_components.copy()


class AdversarialSmoothingLoss(nn.Module):
    """
    Adversarial training for smoother contours.
    
    Generates adversarial perturbations to variables and ensures predictions
    remain smooth under these perturbations.
    """
    
    def __init__(
        self,
        variable_registry: Optional[VariableRegistry] = None,
        epsilon: float = 0.01,  # Perturbation magnitude
        num_steps: int = 3,     # Number of adversarial steps
        step_size: float = 0.005
    ):
        super().__init__()
        
        self.registry = variable_registry or VariableRegistry()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
    
    def forward(
        self,
        model: nn.Module,
        variables: Dict[str, torch.Tensor],
        sequence_tokens: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
        active_variables: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Generate adversarial examples and compute smoothing loss."""
        if active_variables is None:
            active_variables = list(variables.keys())
        
        # Get original predictions
        with torch.no_grad():
            if model.predict_uncertainty:
                orig_pred, _ = model(variables, sequence_tokens, sequence_mask)
            else:
                orig_pred = model(variables, sequence_tokens, sequence_mask)
        
        # Initialize adversarial perturbations
        adv_variables = {}
        perturbations = {}
        
        for name, value in variables.items():
            adv_variables[name] = value.clone().detach()
            if name in active_variables:
                perturbations[name] = torch.zeros_like(value, requires_grad=True)
            else:
                perturbations[name] = torch.zeros_like(value)
        
        # Generate adversarial perturbations
        for step in range(self.num_steps):
            # Apply current perturbations
            perturbed_variables = {}
            for name, value in variables.items():
                if name in active_variables:
                    perturbed_variables[name] = value + perturbations[name]
                else:
                    perturbed_variables[name] = value
            
            # Forward pass
            if model.predict_uncertainty:
                adv_pred, _ = model(perturbed_variables, sequence_tokens, sequence_mask)
            else:
                adv_pred = model(perturbed_variables, sequence_tokens, sequence_mask)
            
            # Maximize difference from original prediction
            loss = -F.mse_loss(adv_pred, orig_pred)
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=loss,
                inputs=[perturbations[name] for name in active_variables if name in perturbations],
                create_graph=False,
                only_inputs=True
            )
            
            # Update perturbations
            with torch.no_grad():
                for i, name in enumerate(active_variables):
                    if name in perturbations:
                        grad = gradients[i]
                        perturbations[name] += self.step_size * grad.sign()
                        
                        # Project to epsilon ball
                        perturbations[name] = torch.clamp(
                            perturbations[name], -self.epsilon, self.epsilon
                        )
        
        # Final adversarial predictions
        final_adv_variables = {}
        for name, value in variables.items():
            if name in active_variables and name in perturbations:
                final_adv_variables[name] = value + perturbations[name]
            else:
                final_adv_variables[name] = value
        
        if model.predict_uncertainty:
            final_adv_pred, _ = model(final_adv_variables, sequence_tokens, sequence_mask)
        else:
            final_adv_pred = model(final_adv_variables, sequence_tokens, sequence_mask)
        
        # Smoothing loss: minimize difference between clean and adversarial predictions
        smoothing_loss = F.mse_loss(orig_pred, final_adv_pred)
        
        return smoothing_loss


class MonotonicityLoss(nn.Module):
    """
    Enforce monotonicity constraints on specific variables.
    
    For variables where we expect monotonic relationships with eye width
    (e.g., larger conductor width -> smaller eye width), enforce this constraint.
    """
    
    def __init__(
        self,
        variable_registry: Optional[VariableRegistry] = None,
        monotonic_variables: Optional[Dict[str, str]] = None  # var_name -> "increasing"/"decreasing"
    ):
        super().__init__()
        
        self.registry = variable_registry or VariableRegistry()
        self.monotonic_variables = monotonic_variables or {}
    
    def forward(
        self,
        model: nn.Module,
        variables: Dict[str, torch.Tensor],
        sequence_tokens: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute monotonicity violation loss."""
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for var_name, direction in self.monotonic_variables.items():
            if var_name not in variables:
                continue
            
            # Create two versions: original and slightly perturbed
            delta = 0.01  # Small perturbation
            
            vars_low = variables.copy()
            vars_high = variables.copy()
            
            vars_low[var_name] = variables[var_name] - delta
            vars_high[var_name] = variables[var_name] + delta
            
            # Get predictions
            if model.predict_uncertainty:
                pred_low, _ = model(vars_low, sequence_tokens, sequence_mask)
                pred_high, _ = model(vars_high, sequence_tokens, sequence_mask)
            else:
                pred_low = model(vars_low, sequence_tokens, sequence_mask)
                pred_high = model(vars_high, sequence_tokens, sequence_mask)
            
            # Compute monotonicity violation
            if direction == "increasing":
                # Eye width should increase with variable
                violation = F.relu(pred_low - pred_high)  # Penalize if pred_low > pred_high
            elif direction == "decreasing":
                # Eye width should decrease with variable  
                violation = F.relu(pred_high - pred_low)  # Penalize if pred_high > pred_low
            else:
                continue
            
            total_loss += violation.mean()
        
        return total_loss
