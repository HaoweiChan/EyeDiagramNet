"""
Lightning Module for Variable-Agnostic Contour Training

Implements training loop with random subspace perturbations, multi-component losses,
and evaluation metrics for contour quality assessment.

Also includes Gaussian Process module for hyperspace eye width prediction.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import gpytorch
import torch.nn as nn
import torchmetrics as tm
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from ..models.contour_model import ContourPredictor
from ..models.gp_model import ExactGPModel
from ..utils.contour_losses import ContourLossFunction, AdversarialSmoothingLoss, MonotonicityLoss
from ..data.variable_registry import VariableRegistry
from ..utils.visualization import image_to_buffer, plot_contour_2d


class ContourModule(LightningModule):
    """Lightning module for contour prediction training."""
    
    def __init__(
        self,
        # Model parameters
        variable_registry: Optional[VariableRegistry] = None,
        model_config: Optional[Dict] = None,
        # Loss parameters
        lambda_consistency: float = 0.1,
        lambda_gradient: float = 0.01,
        lambda_spec_band: float = 0.05,
        lambda_adversarial: float = 0.0,
        lambda_monotonicity: float = 0.0,
        # Random subspace training
        min_active_variables: int = 2,
        max_active_variables: int = 6,
        coordinate_dropout_rate: float = 0.1,
        # Evaluation
        eval_contour_pairs: Optional[List[Tuple[str, str]]] = None,
        eval_resolution: int = 25,
        save_contour_plots: bool = True,
        spec_threshold: Optional[float] = None,
        # Specific contour variables (null means use random from eval_contour_pairs)
        var1_name: Optional[str] = None,
        var2_name: Optional[str] = None,
        # Variable range overrides for contour plotting
        var_range: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        super().__init__()
        
        # Store hyperparameters
        self.save_hyperparameters()
        
        self.registry = variable_registry or VariableRegistry()
        self.spec_threshold = spec_threshold
        
        # Apply variable range overrides if provided
        if var_range:
            for var_name, bounds in var_range.items():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    self.registry.update_variable_bounds(var_name, tuple(bounds))
                    rank_zero_info(f"Applied var_range override for {var_name}: {bounds}")
                else:
                    rank_zero_info(f"Warning: Invalid var_range format for {var_name}: {bounds}, expected [min, max]")
        
        # Initialize model
        model_config = model_config or {}
        self.model = ContourPredictor(
            variable_registry=self.registry,
            **model_config
        )
        
        # Initialize loss functions
        self.main_loss = ContourLossFunction(
            variable_registry=self.registry,
            lambda_consistency=lambda_consistency,
            lambda_gradient=lambda_gradient,
            lambda_spec_band=lambda_spec_band,
            spec_threshold=spec_threshold
        )
        
        if lambda_adversarial > 0:
            self.adversarial_loss = AdversarialSmoothingLoss(variable_registry=self.registry)
        else:
            self.adversarial_loss = None
            
        if lambda_monotonicity > 0:
            # Define some common monotonicity constraints
            monotonic_vars = {
                "W_ws": "decreasing",  # Wider signal -> less eye width
                "M_1_cond": "increasing"  # Higher conductivity -> more eye width
            }
            self.monotonicity_loss = MonotonicityLoss(
                variable_registry=self.registry,
                monotonic_variables=monotonic_vars
            )
        else:
            self.monotonicity_loss = None
        
        # Training parameters
        self.min_active_variables = min_active_variables
        self.max_active_variables = max_active_variables
        self.coordinate_dropout_rate = coordinate_dropout_rate
        
        # Evaluation parameters
        self.eval_contour_pairs = eval_contour_pairs or []
        self.eval_resolution = eval_resolution
        self.save_contour_plots = save_contour_plots
        self.var1_name = var1_name
        self.var2_name = var2_name
        
        # Initialize metrics using factory pattern like trace_ew_module
        self.metrics = nn.ModuleDict({
            "train_": self.metrics_factory(),
            "val": self.metrics_factory(),
        })
        
        # Step outputs for plotting
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def _get_contour_output_dir(self) -> Optional[Path]:
        """Get the directory for saving contour plots."""
        if self.logger is None:
            return None
        
        # Get log directory from logger
        log_dir = Path(self.logger.log_dir) if hasattr(self.logger, 'log_dir') else None
        if log_dir is None:
            return None
        
        # Create contours subdirectory
        contour_dir = log_dir / "contours"
        contour_dir.mkdir(parents=True, exist_ok=True)
        return contour_dir
    
    def forward(
        self,
        variables: Dict[str, torch.Tensor],
        sequence_tokens: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None
    ):
        """Forward pass through the model."""
        return self.model(variables, sequence_tokens, sequence_mask)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step with random subspace perturbations."""
        return self._step(batch, batch_idx, "train_")
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._step(batch, batch_idx, "val")

    def _step(self, batch: Dict, batch_idx: int, stage: str) -> torch.Tensor:
        """Shared step function for training and validation."""
        # Unpack batch
        variables = batch['variables']
        sequence_tokens = batch['sequence_tokens']
        targets = batch['targets']
        sequence_mask = batch.get('sequence_mask', None)
        active_vars = batch.get('active_variables', list(variables.keys()))
        
        # Apply perturbations only during training
        if stage == "train_":
            # Apply random subspace perturbations
            variables, active_vars = self._apply_random_subspace_perturbation(variables)
            
            # Apply coordinate dropout
            if self.coordinate_dropout_rate > 0:
                variables = self._apply_coordinate_dropout(variables, active_vars)
        
        # Forward pass
        if self.model.predict_uncertainty:
            predictions, uncertainties = self.model(variables, sequence_tokens, sequence_mask)
        else:
            predictions = self.model(variables, sequence_tokens, sequence_mask)
            uncertainties = None
        
        # Compute loss
        if stage == "train_":
            # Use comprehensive loss with smoothness constraints
            loss = self.main_loss(
                self.model, variables, sequence_tokens, targets,
                sequence_mask, active_vars
            )
            
            # Additional losses during training
            if self.adversarial_loss is not None:
                adv_loss = self.adversarial_loss(
                    self.model, variables, sequence_tokens, sequence_mask, active_vars
                )
                loss += self.hparams.lambda_adversarial * adv_loss
                self.log('train/adversarial_loss', adv_loss, on_step=True, on_epoch=True)
            
            if self.monotonicity_loss is not None:
                mono_loss = self.monotonicity_loss(
                    self.model, variables, sequence_tokens, sequence_mask
                )
                loss += self.hparams.lambda_monotonicity * mono_loss
                self.log('train/monotonicity_loss', mono_loss, on_step=True, on_epoch=True)
            
            # Log loss components
            loss_components = self.main_loss.get_loss_components()
            for component, value in loss_components.items():
                self.log(f'{stage}/{component}_loss', value, on_step=True, on_epoch=True)
        
        else:
            # Simple MSE loss for validation
            loss = nn.functional.mse_loss(predictions, targets)
            self.log('val/mse_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update metrics
        with torch.no_grad():
            self.update_metrics(stage, loss.detach(), predictions, targets, uncertainties)
        
        # Store outputs for epoch-end processing
        self._collect_step_outputs(stage, predictions, targets, uncertainties, variables, 
                                 sequence_tokens, active_vars)
        
        # Plot contours periodically during validation
        self._plot_contours_if_needed(stage, batch_idx, variables, sequence_tokens)
        
        return loss
    
    def on_training_epoch_end(self):
        """Compute epoch-level training metrics."""
        log_metrics = self.compute_metrics("train_")
        
        # Log variable usage statistics if we have outputs
        if self.training_step_outputs:
            all_active_vars = [x['active_variables'] for x in self.training_step_outputs]
            var_usage = {}
            for active_vars in all_active_vars:
                for var in active_vars:
                    var_usage[var] = var_usage.get(var, 0) + 1
            
            rank_zero_info(f"Epoch {self.current_epoch} variable usage: {var_usage}")
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Compute epoch-level validation metrics and generate contour plots."""
        log_metrics = self.compute_metrics("val")
        
        # Clear outputs
        self.validation_step_outputs.clear()

    def on_fit_end(self):
        """Save final contour plots to disk when training is complete."""
        if not self.save_contour_plots:
            return
        
        rank_zero_info("Generating final contour plots...")
        
        # Get contour pair to plot
        contour_pair = self._get_contour_pair_for_plotting()
        if contour_pair is None:
            rank_zero_info("No valid contour pair found for final plotting")
            return
        
        var1_name, var2_name = contour_pair
        
        try:
            # Use default values for fixed variables
            fixed_vars = self.registry.get_default_values()
            
            # Create a dummy sequence token (use zeros as placeholder)
            # In practice, you might want to use a representative sequence from validation
            sequence_tokens = torch.zeros(1, 96, device=self.device)  # Adjust size as needed
            
            # Get variable ranges
            var1_bounds = self.registry.get_bounds(var1_name)
            var2_bounds = self.registry.get_bounds(var2_name)
            
            if var1_bounds is None or var2_bounds is None:
                rank_zero_info(f"Cannot generate final contour: missing bounds for {var1_name} or {var2_name}")
                return
            
            # Generate high-resolution contour for final plot
            self.model.eval()
            with torch.no_grad():
                var1_grid, var2_grid, predictions = self.model.predict_contour_2d(
                    var1_name=var1_name,
                    var2_name=var2_name,
                    fixed_variables=fixed_vars,
                    sequence_tokens=sequence_tokens,
                    var1_range=var1_bounds,
                    var2_range=var2_bounds,
                    resolution=self.eval_resolution,
                    device=self.device
                )
            
            # Create plot
            fig = plot_contour_2d(
                var1_name=var1_name,
                var2_name=var2_name,
                var1_grid=var1_grid,
                var2_grid=var2_grid,
                predictions=predictions,
                spec_threshold=self.spec_threshold,
                error_data=None
            )
            
            if fig is not None:
                # Save to disk
                contour_dir = self._get_contour_output_dir()
                if contour_dir is not None:
                    filename = f"final_contour_{var1_name}_vs_{var2_name}.png"
                    filepath = contour_dir / filename
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    rank_zero_info(f"Saved final contour plot to {filepath}")
                plt.close(fig)
        
        except Exception as e:
            rank_zero_info(f"Failed to generate final contour plot: {e}")
    
    def _get_contour_pair_for_plotting(self) -> Optional[Tuple[str, str]]:
        """Get the contour pair to plot (either explicit or random from eval_contour_pairs)."""
        # If both var1_name and var2_name are specified, use them
        if self.var1_name is not None and self.var2_name is not None:
            if self.var1_name in self.registry and self.var2_name in self.registry:
                return (self.var1_name, self.var2_name)
            else:
                rank_zero_info(f"Warning: Specified variables ({self.var1_name}, {self.var2_name}) not in registry")
        
        # If only one is specified, try to find a pair from eval_contour_pairs
        if self.var1_name is not None or self.var2_name is not None:
            target_var = self.var1_name or self.var2_name
            for pair in self.eval_contour_pairs:
                if target_var in pair and all(v in self.registry for v in pair):
                    return pair
            rank_zero_info(f"Warning: Could not find pair containing {target_var} in eval_contour_pairs")
        
        # If eval_contour_pairs is provided, use it
        if self.eval_contour_pairs:
            # Filter pairs where both variables exist in registry and have bounds
            valid_pairs = [
                pair for pair in self.eval_contour_pairs 
                if all(v in self.registry and self.registry.get_bounds(v) is not None for v in pair)
            ]
            
            if valid_pairs:
                return random.choice(valid_pairs)
            else:
                rank_zero_info("Warning: No valid pairs with bounds found in eval_contour_pairs")
        else:
            # If eval_contour_pairs is not provided, randomly sample from all variables with bounds
            variables_with_bounds = [
                name for name in self.registry.variable_names
                if self.registry.get_bounds(name) is not None
            ]
            
            if len(variables_with_bounds) >= 2:
                # Randomly select two different variables
                var1_name, var2_name = random.sample(variables_with_bounds, 2)
                rank_zero_info(f"Auto-selected contour pair: {var1_name} vs {var2_name}")
                return (var1_name, var2_name)
            else:
                rank_zero_info(f"Warning: Not enough variables with bounds ({len(variables_with_bounds)}) for contour plotting")
        
        return None
    
    def _apply_random_subspace_perturbation(
        self, 
        variables: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Apply random subspace perturbations - key for variable-agnostic training.
        
        Randomly selects a subset of variables to perturb, keeping others fixed.
        This prevents the model from overfitting to specific variable pairs.
        """
        variable_names = list(variables.keys())
        
        # Randomly choose number of variables to perturb
        n_active = random.randint(
            min(self.min_active_variables, len(variable_names)),
            min(self.max_active_variables, len(variable_names))
        )
        
        # Randomly select which variables to perturb
        active_variables = random.sample(variable_names, n_active)
        
        # Apply small random perturbations only to active variables
        perturbed_variables = {}
        for name, value in variables.items():
            if name in active_variables and name in self.registry:
                var_info = self.registry.get_variable(name)
                
                # Use a percentage of the current value as noise scale (since we don't have bounds)
                # This is adaptive to the actual scale of the variable
                noise_scale = random.uniform(0.01, 0.05) * torch.abs(value).clamp(min=1e-6)
                noise = torch.randn_like(value) * noise_scale
                
                # Apply perturbation (no clamping since we don't have bounds)
                perturbed_value = value + noise
                perturbed_variables[name] = perturbed_value
            else:
                # Keep inactive variables unchanged
                perturbed_variables[name] = value
        
        return perturbed_variables, active_variables
    
    def _apply_coordinate_dropout(
        self,
        variables: Dict[str, torch.Tensor],
        active_variables: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply coordinate dropout by temporarily masking some variable tokens.
        
        This forces the model to be robust to missing variables.
        """
        dropped_variables = {}
        
        for name, value in variables.items():
            if name in active_variables and random.random() < self.coordinate_dropout_rate:
                # Replace with a generic default value (zero or current value mean)
                if name in self.registry:
                    var_info = self.registry.get_variable(name)
                    # Use zero as default or mean of current batch values
                    default_val = torch.mean(value).item() if value.numel() > 1 else 0.0
                    
                    dropped_variables[name] = torch.full_like(value, default_val)
                else:
                    dropped_variables[name] = value
            else:
                dropped_variables[name] = value
        
        return dropped_variables
    
    def _collect_step_outputs(
        self,
        stage: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor],
        variables: Dict[str, torch.Tensor],
        sequence_tokens: torch.Tensor,
        active_vars: List[str]
    ):
        """Collect outputs for epoch-end processing."""
        with torch.no_grad():
            # Compute errors
            errors = (predictions - targets).detach().cpu()
            
            step_output = {
                'predictions': predictions.detach().cpu(),
                'targets': targets.detach().cpu(),
                'errors': errors,
                'uncertainties': uncertainties.detach().cpu() if uncertainties is not None else None,
                'active_variables': active_vars
            }
            
            if stage == "train_":
                self.training_step_outputs.append(step_output)
            elif stage == "val":
                # Include additional info for validation (used for contour plots)
                step_output.update({
                    'variables': {k: v.detach().cpu() for k, v in variables.items()},
                    'sequence_tokens': sequence_tokens.detach().cpu()
                })
                self.validation_step_outputs.append(step_output)
    
    def _plot_contours_if_needed(
        self,
        stage: str,
        batch_idx: int,
        variables: Dict[str, torch.Tensor],
        sequence_tokens: torch.Tensor
    ):
        """Plot contours periodically during validation steps."""
        if not self.save_contour_plots or not self.logger or stage != "val":
            return
        
        # Only plot once per epoch (on batch_idx 0)
        if batch_idx != 0:
            return
        
        try:
            # Get fixed values (use first sample from batch)
            fixed_vars = {}
            for name, tensor in variables.items():
                if name in self.registry:
                    # Use the first sample's values as fixed values
                    fixed_vars[name] = tensor[0].item() if tensor.dim() > 0 else tensor.item()
            
            # Use the first sequence in the batch
            sequence = sequence_tokens[0] if sequence_tokens.dim() > 1 else sequence_tokens
            
            # Get contour pair to plot (either explicit or random from eval_contour_pairs)
            contour_pair = self._get_contour_pair_for_plotting()
            if contour_pair is None:
                return
            
            var1_name, var2_name = contour_pair
            
            try:
                # Generate contour plot for the selected pair
                fig = self._plot_step_contour(
                    var1_name=var1_name,
                    var2_name=var2_name,
                    fixed_variables=fixed_vars,
                    sequence_tokens=sequence,
                    stage=stage,
                    step=batch_idx
                )
                
                if fig is not None and self.logger:
                    # Log to tensorboard (following trace_ew_module pattern)
                    plot_name = f'contours_{stage}/{var1_name}_vs_{var2_name}'
                    self.logger.experiment.add_image(plot_name, image_to_buffer(fig), self.current_epoch)
                    plt.close(fig)
                    
            except Exception as e:
                rank_zero_info(f"Failed to generate step contour plot for {var1_name} vs {var2_name}: {e}")
                return
                    
        except Exception as e:
            rank_zero_info(f"Error in contour plotting during {stage} step {batch_idx}: {e}")
    
    def _plot_step_contour(
        self,
        var1_name: str,
        var2_name: str,
        fixed_variables: Dict[str, float],
        sequence_tokens: torch.Tensor,
        stage: str,
        step: int,
        resolution: int = 20  # Lower resolution for step plots
    ) -> Optional[plt.Figure]:
        """Generate a step-level contour plot using the visualization module."""
        try:
            # Get variable bounds
            var1_bounds = self.registry.get_bounds(var1_name)
            var2_bounds = self.registry.get_bounds(var2_name)
            
            # Skip plotting if either variable has no bounds (constant variable)
            if var1_bounds is None or var2_bounds is None:
                rank_zero_info(f"Skipping contour plot for {var1_name} vs {var2_name}: "
                              f"{var1_name} bounds={var1_bounds}, {var2_name} bounds={var2_bounds}")
                return None
            
            # Generate contour using model's predict_contour_2d method
            self.model.eval()
            with torch.no_grad():
                var1_grid, var2_grid, predictions = self.model.predict_contour_2d(
                    var1_name=var1_name,
                    var2_name=var2_name,
                    fixed_variables=fixed_variables,
                    sequence_tokens=sequence_tokens,
                    var1_range=var1_bounds,
                    var2_range=var2_bounds,
                    resolution=resolution,
                    device=self.device
                )
            
            # Prepare error data from validation outputs collected so far
            error_data = self._extract_error_data_for_variables(var1_name, var2_name)
            
            # Use visualization module to create plot
            fig = plot_contour_2d(
                var1_name=var1_name,
                var2_name=var2_name,
                var1_grid=var1_grid,
                var2_grid=var2_grid,
                predictions=predictions,
                spec_threshold=self.spec_threshold,
                error_data=error_data
            )
            
            return fig
            
        except Exception as e:
            rank_zero_info(f"Error generating step contour plot: {e}")
            return None
    
    def _extract_error_data_for_variables(self, var1_name: str, var2_name: str) -> Optional[Dict]:
        """Extract error data for specific variable pair from validation outputs."""
        try:
            if not self.validation_step_outputs:
                return None
            
            var1_values = []
            var2_values = []
            errors = []
            
            for output in self.validation_step_outputs:
                variables = output.get('variables', {})
                step_errors = output.get('errors', None)
                
                if (var1_name in variables and var2_name in variables and 
                    step_errors is not None):
                    
                    # Get variable values for this step
                    var1_vals = variables[var1_name]  # [batch_size]
                    var2_vals = variables[var2_name]  # [batch_size] 
                    step_errs = step_errors.squeeze()  # [batch_size]
                    
                    # Convert to lists and extend
                    var1_values.extend(var1_vals.numpy().flatten())
                    var2_values.extend(var2_vals.numpy().flatten())
                    errors.extend(step_errs.numpy().flatten())
            
            if len(errors) > 0:
                return {
                    'var1_values': var1_values,
                    'var2_values': var2_values, 
                    'errors': errors
                }
            else:
                return None
                
        except Exception as e:
            rank_zero_info(f"Error extracting error data: {e}")
            return None
    
    def generate_contour_evaluation_plots(self):
        """Generate 2D contour plots for evaluation pairs using visualization module."""
        if not self.validation_step_outputs:
            return
        
        # Get a representative sample for plotting
        sample = self.validation_step_outputs[0]
        variables = sample['variables']
        sequence_tokens = sample['sequence_tokens']
        
        # Create fixed variable values (use defaults)
        fixed_vars = self.registry.get_default_values()
        
        # Get contour pair to plot (either explicit or random from eval_contour_pairs)
        contour_pair = self._get_contour_pair_for_plotting()
        if contour_pair is None:
            return
        
        var1_name, var2_name = contour_pair
        
        try:
            # Get variable ranges
            var1_bounds = self.registry.get_bounds(var1_name)
            var2_bounds = self.registry.get_bounds(var2_name)
            
            # Generate contour using model
            self.model.eval()
            with torch.no_grad():
                var1_grid, var2_grid, predictions = self.model.predict_contour_2d(
                    var1_name=var1_name,
                    var2_name=var2_name,
                    fixed_variables=fixed_vars,
                    sequence_tokens=sequence_tokens[0],  # Use first sequence
                    var1_range=var1_bounds,
                    var2_range=var2_bounds,
                    resolution=self.eval_resolution,
                    device=self.device
                )
            
            # Prepare error data from validation outputs
            error_data = self._extract_error_data_for_variables(var1_name, var2_name)
            
            # Use visualization module to create plot
            fig = plot_contour_2d(
                var1_name=var1_name,
                var2_name=var2_name,
                var1_grid=var1_grid,
                var2_grid=var2_grid,
                predictions=predictions,
                spec_threshold=self.spec_threshold,
                error_data=error_data
            )
            
            # Log plot to tensorboard
            if fig is not None and self.logger:
                self.logger.experiment.add_image(
                    f'contours/{var1_name}_vs_{var2_name}',
                    image_to_buffer(fig),
                    self.current_epoch
                )
                plt.close(fig)
            
        except Exception as e:
            rank_zero_info(f"Failed to generate contour evaluation plot for {var1_name} vs {var2_name}: {e}")
    
    def predict_contour(
        self,
        var1_name: str,
        var2_name: str,
        sequence_tokens: torch.Tensor,
        fixed_variables: Optional[Dict[str, float]] = None,
        var1_range: Optional[Tuple[float, float]] = None,
        var2_range: Optional[Tuple[float, float]] = None,
        resolution: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate 2D contour for any pair of variables.
        
        Public interface for generating contours after training.
        """
        if fixed_variables is None:
            fixed_variables = self.registry.get_default_values()
        
        if var1_range is None:
            var1_range = self.registry.get_bounds(var1_name)
        if var2_range is None:
            var2_range = self.registry.get_bounds(var2_name)
        
        return self.model.predict_contour_2d(
            var1_name=var1_name,
            var2_name=var2_name,
            fixed_variables=fixed_variables,
            sequence_tokens=sequence_tokens,
            var1_range=var1_range,
            var2_range=var2_range,
            resolution=resolution,
            device=self.device
        )
    
    def set_contour_plot_config(
        self,
        eval_pairs: Optional[List[Tuple[str, str]]] = None,
        resolution: Optional[int] = None,
        enable_plotting: Optional[bool] = None
    ):
        """
        Configure contour plotting parameters.
        
        Args:
            eval_pairs: List of (var1, var2) pairs to plot
            resolution: Grid resolution for contour plots
            enable_plotting: Enable/disable contour plotting
        """
        if eval_pairs is not None:
            # Validate that all variables exist in registry
            valid_pairs = []
            for var1, var2 in eval_pairs:
                if var1 in self.registry and var2 in self.registry:
                    valid_pairs.append((var1, var2))
                else:
                    rank_zero_info(f"Warning: Variable pair ({var1}, {var2}) contains unknown variables. Skipping.")
            
            self.eval_contour_pairs = valid_pairs
            rank_zero_info(f"Updated contour evaluation pairs: {self.eval_contour_pairs}")
        
        if resolution is not None:
            self.eval_resolution = resolution
            rank_zero_info(f"Updated contour plot resolution: {resolution}")
        
        if enable_plotting is not None:
            self.save_contour_plots = enable_plotting
            status = "enabled" if enable_plotting else "disabled"
            rank_zero_info(f"Contour plotting {status}")
    
    def get_available_variable_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all possible variable pairs for contour plotting.
        
        Returns:
            List of (var1, var2) tuples for all registered variables
        """
        variables = list(self.registry.variable_names)
        pairs = []
        
        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                pairs.append((var1, var2))
        
        return pairs
    
    def get_variable_pairs_by_role(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Get variable pairs grouped by their semantic roles.
        
        Returns:
            Dict mapping role combinations to variable pairs
        """
        from itertools import combinations
        
        # Group variables by role
        role_groups = {}
        for var_name in self.registry.variable_names:
            variable = self.registry.get_variable(var_name)
            role = variable.role.value
            if role not in role_groups:
                role_groups[role] = []
            role_groups[role].append(var_name)
        
        paired_roles = {}
        
        # Same role pairs (e.g., height vs height)
        for role, vars_in_role in role_groups.items():
            if len(vars_in_role) >= 2:
                same_role_pairs = list(combinations(vars_in_role, 2))
                paired_roles[f"{role}_vs_{role}"] = same_role_pairs
        
        # Cross-role pairs (e.g., width vs height)
        role_names = list(role_groups.keys())
        for i, role1 in enumerate(role_names):
            for role2 in role_names[i+1:]:
                cross_pairs = [(v1, v2) for v1 in role_groups[role1] for v2 in role_groups[role2]]
                if cross_pairs:
                    paired_roles[f"{role1}_vs_{role2}"] = cross_pairs
        
        return paired_roles
    
    def metrics_factory(self):
        """Create metrics for contour prediction following trace_ew_module pattern."""
        metrics = {
            'loss': tm.MeanMetric,
            'mae': tm.MeanAbsoluteError,
            'mse': tm.MeanSquaredError,
            'rmse': lambda: tm.MeanSquaredError(squared=False),
            'r2': tm.R2Score,
        }
        metrics_dict = nn.ModuleDict()
        for k, metric in metrics.items():
            metrics_dict[k] = metric()
        return metrics_dict
    
    def update_metrics(
        self,
        stage: str,
        loss: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None
    ):
        """Update metrics for the given stage."""
        # Flatten tensors for metric computation
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Update all metrics
        for key, metric in self.metrics[stage].items():
            if 'loss' in key:
                metric.update(loss)
            else:
                metric.update(pred_flat, target_flat)
    
    def compute_metrics(self, stage: str):
        """Compute and log metrics for the given stage."""
        log_metrics = {}
        for key, metric in self.metrics[stage].items():
            log_metrics[f'{stage}/{key}'] = metric.compute()
            metric.reset()
        
        if self.logger is not None:
            self.logger.log_metrics(log_metrics, self.current_epoch)
        
        # Log validation metrics to progress bar
        if stage == "val":
            self.log(
                'hp_metric', log_metrics[f'{stage}/mae'],
                prog_bar=True, on_epoch=True, on_step=False, sync_dist=True
            )
        
        return log_metrics

class GaussianProcessModule(LightningModule):
    """
    Lightning module for Gaussian Process regression on eye width prediction.
    
    Predicts minimum eye width across parameter hyperspace with uncertainty quantification.
    This module is designed to work with the same data structure as ContourModule.
    """
    
    def __init__(
        self,
        # Model parameters
        variable_registry: Optional[VariableRegistry] = None,
        use_ard: bool = True,
        noise_constraint: float = 1e-4,
        # Evaluation
        eval_contour_pairs: Optional[List[Tuple[str, str]]] = None,
        eval_resolution: int = 25,
        save_contour_plots: bool = True,
        spec_threshold: Optional[float] = None,
        # Specific contour variables
        var1_name: Optional[str] = None,
        var2_name: Optional[str] = None,
        # Variable range overrides
        var_range: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        super().__init__()
        
        # Store hyperparameters
        self.save_hyperparameters()
        
        # Use manual optimization since GP model is initialized after first batch
        self.automatic_optimization = False
        
        self.registry = variable_registry or VariableRegistry()
        self.spec_threshold = spec_threshold
        self.use_ard = use_ard
        
        # Apply variable range overrides if provided
        if var_range:
            for var_name, bounds in var_range.items():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    self.registry.update_variable_bounds(var_name, tuple(bounds))
                    rank_zero_info(f"Applied var_range override for {var_name}: {bounds}")
        
        # GP model and likelihood will be initialized in setup()
        self.gp_model = None
        self.likelihood = None
        self.input_dim = None
        
        # Variable scaler for numerical stability (loaded from datamodule in setup)
        self.var_scaler = None
        
        # Store variable names for consistent ordering
        self.variable_names = None
        
        # Evaluation parameters
        self.eval_contour_pairs = eval_contour_pairs or []
        self.eval_resolution = eval_resolution
        self.save_contour_plots = save_contour_plots
        self.var1_name = var1_name
        self.var2_name = var2_name
        
        # Initialize metrics
        self.metrics = nn.ModuleDict({
            "train_": self.metrics_factory(),
            "val": self.metrics_factory(),
        })
        
        # Step outputs for plotting
        self.validation_step_outputs = []

    def _get_contour_output_dir(self) -> Optional[Path]:
        """Get the directory for saving contour plots."""
        if self.logger is None:
            return None
        
        # Get log directory from logger
        log_dir = Path(self.logger.log_dir) if hasattr(self.logger, 'log_dir') else None
        if log_dir is None:
            return None
        
        # Create contours subdirectory
        contour_dir = log_dir / "contours"
        contour_dir.mkdir(parents=True, exist_ok=True)
        return contour_dir
    
    def setup(self, stage: Optional[str] = None):
        """Initialize GP model and retrieve scaler from datamodule."""
        # Retrieve variable scaler from datamodule for inference
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'var_scaler'):
            self.var_scaler = self.trainer.datamodule.var_scaler
            rank_zero_info("Loaded variable scaler from datamodule for GP inference")
        else:
            self.var_scaler = None
            rank_zero_info("Warning: No variable scaler found in datamodule. GP predictions may be numerically unstable.")
    
    def _initialize_gp_model(self, x: torch.Tensor, y: torch.Tensor):
        """Initialize GP model with first batch of data."""
        if self.gp_model is not None:
            return  # Already initialized
        
        self.input_dim = x.shape[1]
        
        # Create likelihood with noise constraint
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(self.hparams.noise_constraint)
        )
        
        # Create GP model
        self.gp_model = ExactGPModel(
            train_x=x,
            train_y=y,
            likelihood=self.likelihood,
            input_dim=self.input_dim,
            use_ard=self.use_ard
        )
        
        # Move to correct device
        self.gp_model = self.gp_model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)
        
        rank_zero_info(f"Initialized GP model with input_dim={self.input_dim}, ARD={self.use_ard}")
        
        # Replace optimizer with real parameters
        opt = self.optimizers()
        opt.param_groups.clear()
        # Collect all unique parameters from model and likelihood
        all_params = list(self.gp_model.parameters()) + list(self.likelihood.parameters())
        opt.add_param_group({'params': all_params})
    
    def _variables_dict_to_tensor(self, variables: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert variables dictionary to tensor with consistent ordering.
        
        Args:
            variables: Dict of variable_name -> tensor [batch_size]
            
        Returns:
            Tensor of shape [batch_size, n_variables]
        """
        if self.variable_names is None:
            # Establish variable ordering on first call
            self.variable_names = sorted(variables.keys())
            rank_zero_info(f"Established variable ordering: {self.variable_names}")
        
        # Stack variables in consistent order
        var_tensors = [variables[name] for name in self.variable_names]
        return torch.stack(var_tensors, dim=1)  # [batch_size, n_variables]
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through GP.
        
        Args:
            x: Input features [batch_size, n_features]
            
        Returns:
            Predictive distribution
        """
        if self.gp_model is None:
            raise RuntimeError("GP model not initialized. Call training_step first.")
        
        return self.gp_model(x)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step - optimize marginal log likelihood."""
        return self._step(batch, batch_idx, "train_")
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._step(batch, batch_idx, "val")
    
    def _step(self, batch: Dict, batch_idx: int, stage: str) -> torch.Tensor:
        """Shared step function for training and validation."""
        # Unpack batch
        variables = batch['variables']
        targets = batch['targets'].squeeze()  # [batch_size]
        
        # Convert variables dict to tensor
        x = self._variables_dict_to_tensor(variables)  # [batch_size, n_variables]
        y = targets
        
        # Initialize GP on first training batch
        if stage == "train_" and self.gp_model is None:
            self._initialize_gp_model(x, y)
        
        if self.gp_model is None:
            # Skip if not initialized yet (shouldn't happen in normal training)
            return torch.tensor(0.0, device=self.device)
        
        if stage == "train_":
            # Get optimizer for manual optimization
            opt = self.optimizers()
            
            # Training mode: optimize marginal log likelihood
            self.gp_model.train()
            self.likelihood.train()
            
            # Update training data for this epoch (required for Exact GP)
            self.gp_model.set_train_data(inputs=x, targets=y, strict=False)
            
            # Forward pass
            output = self.gp_model(x)
            
            # Compute negative marginal log likelihood
            loss = -gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)(output, y)
            
            # Manual optimization
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            
            # Log loss
            self.log('train/mll_loss', loss, on_step=True, on_epoch=True)
            
            # Log GP hyperparameters to show training progression
            if hasattr(self.gp_model.covar_module, 'outputscale'):
                self.log('gp_params/outputscale', self.gp_model.covar_module.outputscale.item(), on_step=False, on_epoch=True)
            if hasattr(self.gp_model.covar_module, 'base_kernel') and hasattr(self.gp_model.covar_module.base_kernel, 'kernels'):
                # Log RBF lengthscale (first kernel in sum)
                rbf_kernel = self.gp_model.covar_module.base_kernel.kernels[0]
                if hasattr(rbf_kernel, 'lengthscale'):
                    lengthscales = rbf_kernel.lengthscale.squeeze()
                    if lengthscales.numel() > 1:  # ARD
                        for i, ls in enumerate(lengthscales):
                            self.log(f'gp_params/lengthscale_{i}', ls.item(), on_step=False, on_epoch=True)
                    else:
                        self.log('gp_params/lengthscale', lengthscales.item(), on_step=False, on_epoch=True)
            self.log('gp_params/noise', self.likelihood.noise.item(), on_step=False, on_epoch=True)
            
        else:
            # Validation mode: compute predictive performance
            self.gp_model.eval()
            self.likelihood.eval()
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                output = self.likelihood(self.gp_model(x))
                predictions = output.mean
                uncertainties = output.variance.sqrt()
                
                # Compute MSE loss
                loss = nn.functional.mse_loss(predictions, y)
                
                # Log metrics
                self.log('val/mse_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val/mean_uncertainty', uncertainties.mean(), on_step=False, on_epoch=True)
                
                # Update metrics
                self.update_metrics(stage, loss, predictions, y, uncertainties)
                
                # Store outputs for plotting
                self._collect_step_outputs(predictions, y, uncertainties, variables)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute epoch-level validation metrics and generate contour plots."""
        log_metrics = self.compute_metrics("val")
        
        # Generate contour plots
        if self.validation_step_outputs:
            # Use variables from first validation batch for fixed values
            first_batch_vars = self.validation_step_outputs[0]['variables']
            self._plot_contours_if_needed(first_batch_vars)
        
        # Clear outputs
        self.validation_step_outputs.clear()

    def on_fit_end(self):
        """Save final contour plots to disk when training is complete."""
        if not self.save_contour_plots:
            return
        
        rank_zero_info("Generating final GP contour plots...")
        
        # Get contour pair to plot
        contour_pair = self._get_contour_pair_for_plotting()
        if contour_pair is None:
            rank_zero_info("No valid contour pair found for final GP plotting")
            return
        
        var1_name, var2_name = contour_pair
        
        try:
            # Use default values for fixed variables
            fixed_vars = self.registry.get_default_values()
            
            # Generate final contour plot with high resolution
            fig = self._plot_contour(
                var1_name=var1_name,
                var2_name=var2_name,
                fixed_variables=fixed_vars,
                error_data=None,
                resolution=50  # Higher resolution for final plot
            )
            
            if fig is not None:
                # Save to disk
                contour_dir = self._get_contour_output_dir()
                if contour_dir is not None:
                    filename = f"final_gp_contour_{var1_name}_vs_{var2_name}.png"
                    filepath = contour_dir / filename
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    rank_zero_info(f"Saved final GP contour plot to {filepath}")
                plt.close(fig)
        
        except Exception as e:
            rank_zero_info(f"Failed to generate final GP contour plot: {e}")
    
    def _collect_step_outputs(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor,
        variables: Dict[str, torch.Tensor]
    ):
        """Collect outputs for epoch-end processing."""
        step_output = {
            'predictions': predictions.detach().cpu(),
            'targets': targets.detach().cpu(),
            'uncertainties': uncertainties.detach().cpu(),
            'variables': {k: v.detach().cpu() for k, v in variables.items()}
        }
        self.validation_step_outputs.append(step_output)
    
    def _plot_contours_if_needed(self, variables: Dict[str, torch.Tensor]):
        """Plot contours during validation."""
        if not self.save_contour_plots:
            return
        
        if not self.logger:
            rank_zero_info("Warning: No logger available for contour plotting")
            return
        
        try:
            # Get contour pair (reuse ContourModule logic)
            contour_pair = self._get_contour_pair_for_plotting()
            if contour_pair is None:
                rank_zero_info("Warning: No valid contour pair found for plotting")
                return
            
            var1_name, var2_name = contour_pair
            rank_zero_info(f"Plotting contour for {var1_name} vs {var2_name}")
            
            # Get fixed values (use first sample from batch or validation outputs)
            fixed_vars = {}
            for name, tensor in variables.items():
                if name in self.registry:
                    if isinstance(tensor, torch.Tensor):
                        fixed_vars[name] = tensor[0].item() if tensor.dim() > 0 else tensor.item()
                    else:
                        # Already a scalar value
                        fixed_vars[name] = tensor
            
            # Prepare error data from validation outputs
            error_data = self._prepare_error_data(var1_name, var2_name)
            
            # Generate contour plot with error analysis
            fig = self._plot_contour(var1_name, var2_name, fixed_vars, error_data=error_data)
            
            if fig is not None:
                plot_name = f'contours_gp/{var1_name}_vs_{var2_name}'
                self.logger.experiment.add_image(plot_name, image_to_buffer(fig), self.current_epoch)
                plt.close(fig)
                rank_zero_info(f"Successfully logged contour plot: {plot_name}")
            else:
                rank_zero_info(f"Warning: _plot_contour returned None for {var1_name} vs {var2_name}")
                
        except Exception as e:
            rank_zero_info(f"Error in GP contour plotting: {e}")
            import traceback
            rank_zero_info(traceback.format_exc())
    
    def _prepare_error_data(self, var1_name: str, var2_name: str) -> Optional[Dict]:
        """Prepare error data from validation outputs for error analysis plot."""
        if not self.validation_step_outputs:
            rank_zero_info(f"Warning: No validation_step_outputs available for error data")
            return None
        
        try:
            # Collect all validation data
            var1_values = []
            var2_values = []
            errors = []
            
            rank_zero_info(f"Preparing error data from {len(self.validation_step_outputs)} validation batches")
            
            for step_output in self.validation_step_outputs:
                predictions = step_output['predictions']
                targets = step_output['targets']
                variables = step_output['variables']
                
                # Check if both variables are present in this batch
                if var1_name not in variables or var2_name not in variables:
                    rank_zero_info(f"Warning: {var1_name} or {var2_name} not found in batch variables: {list(variables.keys())}")
                    continue
                
                # Extract variable values
                var1_vals = variables[var1_name].numpy()
                var2_vals = variables[var2_name].numpy()
                
                # Compute errors (predictions - targets)
                batch_errors = (predictions - targets).numpy()
                
                # Flatten and collect
                var1_values.extend(var1_vals.flatten())
                var2_values.extend(var2_vals.flatten())
                errors.extend(batch_errors.flatten())
            
            if len(errors) > 0:
                rank_zero_info(f"Collected {len(errors)} error samples for {var1_name} vs {var2_name}")
                return {
                    'var1_values': np.array(var1_values),
                    'var2_values': np.array(var2_values),
                    'errors': np.array(errors)
                }
            else:
                rank_zero_info(f"Warning: No error samples collected for {var1_name} vs {var2_name}")
                return None
                
        except Exception as e:
            rank_zero_info(f"Warning: Could not prepare error data: {e}")
            import traceback
            rank_zero_info(traceback.format_exc())
            return None
    
    def _prepare_train_data(self, var1_name: str, var2_name: str) -> Optional[Dict]:
        """Prepare training data points for visualization."""
        if not hasattr(self, 'training_data') or self.training_data is None:
            # Try to get from GP model's training data
            if self.gp_model is not None and hasattr(self.gp_model, 'train_inputs'):
                try:
                    # Get training inputs
                    train_x = self.gp_model.train_inputs[0].cpu().numpy()
                    
                    # Find indices for var1 and var2
                    if self.variable_names is None:
                        return None
                    
                    try:
                        var1_idx = self.variable_names.index(var1_name)
                        var2_idx = self.variable_names.index(var2_name)
                    except ValueError:
                        return None
                    
                    # Extract variable values
                    var1_vals = train_x[:, var1_idx]
                    var2_vals = train_x[:, var2_idx]
                    
                    # Inverse transform to get unscaled values
                    if self.var_scaler is not None:
                        # Create full tensor with all variables at zero, then set var1 and var2
                        full_data = np.zeros((len(var1_vals), len(self.variable_names)))
                        full_data[:, var1_idx] = var1_vals
                        full_data[:, var2_idx] = var2_vals
                        
                        # Inverse transform
                        unscaled = self.var_scaler.inverse_transform(torch.from_numpy(full_data)).numpy()
                        var1_vals = unscaled[:, var1_idx]
                        var2_vals = unscaled[:, var2_idx]
                    
                    return {
                        'var1_values': var1_vals,
                        'var2_values': var2_vals
                    }
                    
                except Exception as e:
                    rank_zero_info(f"Warning: Could not prepare training data: {e}")
                    return None
        
        return None
    
    def _get_contour_pair_for_plotting(self) -> Optional[Tuple[str, str]]:
        """Get contour pair for plotting (same logic as ContourModule)."""
        # If both var1_name and var2_name are specified, use them
        if self.var1_name is not None and self.var2_name is not None:
            if self.var1_name in self.registry and self.var2_name in self.registry:
                return (self.var1_name, self.var2_name)
        
        # If eval_contour_pairs is provided, use it
        if self.eval_contour_pairs:
            valid_pairs = [
                pair for pair in self.eval_contour_pairs 
                if all(v in self.registry and self.registry.get_bounds(v) is not None for v in pair)
            ]
            
            if valid_pairs:
                return random.choice(valid_pairs)
        else:
            # Auto-select variables: prefer spatial variables over boundary variables
            spatial_vars = []
            boundary_vars = []
            
            for name in self.registry.variable_names:
                if self.registry.get_bounds(name) is not None:
                    var_info = self.registry.get_variable(name)
                    if var_info and var_info.var_type == 'spatial':
                        spatial_vars.append(name)
                    else:
                        # Treat as boundary if not explicitly spatial
                        boundary_vars.append(name)
            
            # Prefer two spatial variables
            if len(spatial_vars) >= 2:
                var1_name, var2_name = random.sample(spatial_vars, 2)
                rank_zero_info(f"GP auto-selected spatial variables: {var1_name} vs {var2_name}")
                return (var1_name, var2_name)
            
            # Mix spatial and boundary if only one spatial
            if len(spatial_vars) == 1 and len(boundary_vars) >= 1:
                var1_name = spatial_vars[0]
                var2_name = random.choice(boundary_vars)
                rank_zero_info(f"GP auto-selected mixed variables: {var1_name} (spatial) vs {var2_name} (boundary)")
                return (var1_name, var2_name)
            
            # Fall back to boundary variables only
            if len(boundary_vars) >= 2:
                var1_name, var2_name = random.sample(boundary_vars, 2)
                rank_zero_info(f"GP auto-selected boundary variables: {var1_name} vs {var2_name}")
                return (var1_name, var2_name)
        
        return None
    
    def _plot_contour(
        self,
        var1_name: str,
        var2_name: str,
        fixed_variables: Dict[str, float],
        error_data: Optional[Dict] = None,
        resolution: int = 20
    ) -> Optional[plt.Figure]:
        """Generate contour plot using GP predictions with error analysis."""
        try:
            # Get variable bounds
            var1_bounds = self.registry.get_bounds(var1_name)
            var2_bounds = self.registry.get_bounds(var2_name)
            
            if var1_bounds is None or var2_bounds is None:
                return None
            
            # Generate 2D grid
            var1_grid = np.linspace(var1_bounds[0], var1_bounds[1], resolution)
            var2_grid = np.linspace(var2_bounds[0], var2_bounds[1], resolution)
            var1_mesh, var2_mesh = np.meshgrid(var1_grid, var2_grid)
            
            # Prepare input tensor
            n_points = resolution * resolution
            x_test = torch.zeros(n_points, len(self.variable_names), device=self.device)
            
            # Fill in fixed variables
            for i, var_name in enumerate(self.variable_names):
                if var_name == var1_name:
                    x_test[:, i] = torch.from_numpy(var1_mesh.flatten()).float()
                elif var_name == var2_name:
                    x_test[:, i] = torch.from_numpy(var2_mesh.flatten()).float()
                else:
                    x_test[:, i] = fixed_variables.get(var_name, 0.0)
            
            # Scale inputs for GP numerical stability
            if self.var_scaler is not None:
                x_test = self.var_scaler.transform(x_test)
            
            # Predict
            self.gp_model.eval()
            self.likelihood.eval()
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                output = self.likelihood(self.gp_model(x_test))
                predictions = output.mean.cpu().numpy().reshape(resolution, resolution)
                uncertainties = output.variance.sqrt().cpu().numpy().reshape(resolution, resolution)
            
            # Prepare training data for visualization
            train_data = self._prepare_train_data(var1_name, var2_name)
            
            # Create plot using visualization module with uncertainty and error analysis
            fig = plot_contour_2d(
                var1_name=var1_name,
                var2_name=var2_name,
                var1_grid=torch.from_numpy(var1_mesh),
                var2_grid=torch.from_numpy(var2_mesh),
                predictions=torch.from_numpy(predictions),
                spec_threshold=self.spec_threshold,
                error_data=error_data,
                uncertainty_grid=torch.from_numpy(uncertainties),
                train_data=train_data
            )
            
            return fig
            
        except Exception as e:
            rank_zero_info(f"Error generating GP contour plot: {e}")
            return None
    
    def predict(
        self,
        variables: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty.
        
        Args:
            variables: Dict of variable_name -> tensor
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        x = self._variables_dict_to_tensor(variables)
        
        # Scale inputs for GP numerical stability
        if self.var_scaler is not None:
            x = self.var_scaler.transform(x)
        
        self.gp_model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self.likelihood(self.gp_model(x))
            predictions = output.mean
            uncertainties = output.variance.sqrt()
        
        return predictions, uncertainties
    
    def configure_optimizers(self):
        """
        Configure optimizer for GP hyperparameters.
        
        Note: When using Lightning CLI with optimizer config in YAML, that config
        won't be automatically used because we override configure_optimizers.
        The learning rate is taken from init_args in the YAML or defaults to Adam(lr=0.1).
        """
        # For now, use Adam with a default learning rate
        # In _initialize_gp_model, we'll replace the param_groups with actual GP params
        dummy_param = torch.nn.Parameter(torch.zeros(1, device=self.device))
        optimizer = torch.optim.Adam([dummy_param], lr=0.1)
        return optimizer

    def metrics_factory(self):
        """Create metrics for GP prediction."""
        metrics = {
            'loss': tm.MeanMetric,
            'mae': tm.MeanAbsoluteError,
            'mse': tm.MeanSquaredError,
            'rmse': lambda: tm.MeanSquaredError(squared=False),
            'r2': tm.R2Score,
        }
        metrics_dict = nn.ModuleDict()
        for k, metric in metrics.items():
            metrics_dict[k] = metric()
        return metrics_dict
    
    def update_metrics(
        self,
        stage: str,
        loss: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None
    ):
        """Update metrics for the given stage."""
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        for key, metric in self.metrics[stage].items():
            if 'loss' in key:
                metric.update(loss)
            else:
                metric.update(pred_flat, target_flat)
    
    def compute_metrics(self, stage: str):
        """Compute and log metrics for the given stage."""
        log_metrics = {}
        for key, metric in self.metrics[stage].items():
            try:
                log_metrics[f'{stage}/{key}'] = metric.compute()
            except ValueError as e:
                # Handle cases where metric cannot be computed (e.g., insufficient samples during sanity check)
                rank_zero_info(f"Warning: Could not compute {stage}/{key}: {e}")
            metric.reset()
        
        if self.logger is not None and log_metrics:
            self.logger.log_metrics(log_metrics, self.current_epoch)
        
        if stage == "val":
            self.log(
                'hp_metric', log_metrics[f'{stage}/mae'],
                prog_bar=True, on_epoch=True, on_step=False, sync_dist=True
            )
        
        return log_metrics
