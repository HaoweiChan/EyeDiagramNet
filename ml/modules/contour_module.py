"""
Lightning Module for Variable-Agnostic Contour Training

Implements training loop with random subspace perturbations, multi-component losses,
and evaluation metrics for contour quality assessment.
"""

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

from ..models.contour_model import ContourPredictor
from ..utils.contour_losses import ContourLossFunction, AdversarialSmoothingLoss, MonotonicityLoss
from ..data.variable_registry import VariableRegistry
from ..utils.visualization import image_to_buffer


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
        spec_threshold: Optional[float] = None,
        # Training parameters
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",  # "cosine", "plateau", "step"
        # Random subspace training
        min_active_variables: int = 2,
        max_active_variables: int = 6,
        coordinate_dropout_rate: float = 0.1,
        # Evaluation
        eval_contour_pairs: Optional[List[Tuple[str, str]]] = None,
        eval_resolution: int = 25,
        save_contour_plots: bool = True
    ):
        super().__init__()
        
        # Store hyperparameters
        self.save_hyperparameters()
        
        self.registry = variable_registry or VariableRegistry()
        self.spec_threshold = spec_threshold
        
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
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.min_active_variables = min_active_variables
        self.max_active_variables = max_active_variables
        self.coordinate_dropout_rate = coordinate_dropout_rate
        
        # Evaluation parameters
        self.eval_contour_pairs = eval_contour_pairs or [
            ("W_wg1", "W_ws"),
            ("H_ubm", "H_cbd"),
            ("W_c", "L_l")
        ]
        self.eval_resolution = eval_resolution
        self.save_contour_plots = save_contour_plots
        
        # Contour plotting frequency (plot every N steps/epochs)
        self.contour_plot_every_n_train_steps = 100  # Plot during training every 100 steps
        self.contour_plot_every_n_val_steps = 10     # Plot during validation every 10 steps
        
        # Metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
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
        return self._step(batch, batch_idx, "train")
    
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
        if stage == "train":
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
        if stage == "train":
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
        
        # Store outputs for epoch-end processing
        self._collect_step_outputs(stage, predictions, targets, uncertainties, variables, 
                                 sequence_tokens, active_vars)
        
        # Plot contours periodically during training/validation
        self._maybe_plot_contours_during_step(stage, batch_idx, variables, sequence_tokens)
        
        return loss
    
    def on_training_epoch_end(self):
        """Compute epoch-level training metrics."""
        if not self.training_step_outputs:
            return
        
        # Collect all predictions and targets
        all_predictions = torch.cat([x['predictions'] for x in self.training_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.training_step_outputs])
        
        # Compute metrics
        mae = torch.mean(torch.abs(all_predictions - all_targets))
        rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2))
        
        # R² score
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        self.log('train/mae', mae, on_epoch=True)
        self.log('train/rmse', rmse, on_epoch=True)
        self.log('train/r2', r2, on_epoch=True)
        
        # Log variable usage statistics
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
        if not self.validation_step_outputs:
            return
        
        # Collect all predictions and targets
        all_predictions = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # Compute metrics
        mae = torch.mean(torch.abs(all_predictions - all_targets))
        rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2))
        
        # R² score
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        self.log('val/mae', mae, on_epoch=True)
        self.log('val/rmse', rmse, on_epoch=True)
        self.log('val/r2', r2, on_epoch=True, prog_bar=True)
        
        # Generate contour evaluation plots
        if self.save_contour_plots and len(self.validation_step_outputs) > 0:
            self.generate_contour_evaluation_plots()
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
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
            step_output = {
                'predictions': predictions.detach().cpu(),
                'targets': targets.detach().cpu(),
                'uncertainties': uncertainties.detach().cpu() if uncertainties is not None else None,
                'active_variables': active_vars
            }
            
            if stage == "train":
                self.training_step_outputs.append(step_output)
            elif stage == "val":
                # Include additional info for validation (used for contour plots)
                step_output.update({
                    'variables': {k: v.detach().cpu() for k, v in variables.items()},
                    'sequence_tokens': sequence_tokens.detach().cpu()
                })
                self.validation_step_outputs.append(step_output)
    
    def _maybe_plot_contours_during_step(
        self,
        stage: str,
        batch_idx: int,
        variables: Dict[str, torch.Tensor],
        sequence_tokens: torch.Tensor
    ):
        """Plot contours periodically during training/validation steps."""
        if not self.save_contour_plots or not self.logger:
            return
        
        # Determine if we should plot based on step frequency
        should_plot = False
        if stage == "train":
            should_plot = batch_idx % self.contour_plot_every_n_train_steps == 0
        elif stage == "val":
            should_plot = batch_idx % self.contour_plot_every_n_val_steps == 0
        
        if not should_plot:
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
            
            # Plot contours for each evaluation pair
            for i, (var1_name, var2_name) in enumerate(self.eval_contour_pairs):
                if var1_name not in self.registry or var2_name not in self.registry:
                    continue
                
                try:
                    # Generate contour plot
                    fig = self._plot_2d_contour(
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
                    continue
                    
        except Exception as e:
            rank_zero_info(f"Error in contour plotting during {stage} step {batch_idx}: {e}")
    
    def _plot_2d_contour(
        self,
        var1_name: str,
        var2_name: str,
        fixed_variables: Dict[str, float],
        sequence_tokens: torch.Tensor,
        stage: str,
        step: int,
        resolution: int = 20  # Lower resolution for step plots
    ) -> Optional[plt.Figure]:
        """Generate a 2D contour plot for two variables."""
        try:
            # Get variable bounds
            var1_bounds = self.registry.get_bounds(var1_name)
            var2_bounds = self.registry.get_bounds(var2_name)
            
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
            
            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            
            # Convert to numpy
            var1_np = var1_grid.cpu().numpy()
            var2_np = var2_grid.cpu().numpy()
            pred_np = predictions.cpu().numpy()
            
            # Contour plot
            cs = ax.contourf(var1_np, var2_np, pred_np.T, levels=15, cmap='viridis', alpha=0.8)
            fig.colorbar(cs, ax=ax, label='Predicted Eye Width')
            
            # Add contour lines
            cs_lines = ax.contour(var1_np, var2_np, pred_np.T, levels=8, colors='white', alpha=0.6, linewidths=0.8)
            
            # Add specification threshold if available
            if self.spec_threshold is not None:
                cs_spec = ax.contour(
                    var1_np, var2_np, pred_np.T, 
                    levels=[self.spec_threshold], 
                    colors='red', linewidths=2
                )
                ax.clabel(cs_spec, fmt='Spec', inline=True, fontsize=8)
            
            # Formatting
            var1_units = self.registry.get_variable(var1_name).units
            var2_units = self.registry.get_variable(var2_name).units
            
            ax.set_xlabel(f'{var1_name} ({var1_units})')
            ax.set_ylabel(f'{var2_name} ({var2_units})')
            
            # Title with step information
            epoch_info = f"Epoch {self.current_epoch}"
            step_info = f"Step {step}" if stage == "train" else f"Val Step {step}"
            ax.set_title(f'{var1_name} vs {var2_name}\n{epoch_info}, {step_info}', fontsize=10)
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            rank_zero_info(f"Error generating 2D contour plot: {e}")
            return None
    
    def generate_contour_evaluation_plots(self):
        """Generate 2D contour plots for evaluation pairs."""
        if not self.validation_step_outputs:
            return
        
        # Get a representative sample for plotting
        sample = self.validation_step_outputs[0]
        variables = sample['variables']
        sequence_tokens = sample['sequence_tokens']
        
        # Create fixed variable values (use medians)
        fixed_vars = self.registry.get_default_values()
        
        # Generate plots for each evaluation pair
        for var1_name, var2_name in self.eval_contour_pairs:
            if var1_name not in self.registry or var2_name not in self.registry:
                continue
            
            try:
                # Get variable ranges
                var1_bounds = self.registry.get_bounds(var1_name)
                var2_bounds = self.registry.get_bounds(var2_name)
                
                # Generate contour
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
                
                # Create plot
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                
                # Convert to numpy
                var1_np = var1_grid.cpu().numpy()
                var2_np = var2_grid.cpu().numpy()
                pred_np = predictions.cpu().numpy()
                
                # Contour plot
                cs = ax.contourf(var1_np, var2_np, pred_np.T, levels=20, cmap='viridis')
                fig.colorbar(cs, ax=ax, label='Predicted Eye Width')
                
                # Add contour lines
                cs_lines = ax.contour(var1_np, var2_np, pred_np.T, levels=10, colors='white', alpha=0.5)
                
                # Add specification threshold if available
                if self.spec_threshold is not None:
                    cs_spec = ax.contour(
                        var1_np, var2_np, pred_np.T, 
                        levels=[self.spec_threshold], 
                        colors='red', linewidths=2
                    )
                    ax.clabel(cs_spec, fmt='Spec', inline=True)
                
                ax.set_xlabel(f'{var1_name} ({self.registry.get_variable(var1_name).units})')
                ax.set_ylabel(f'{var2_name} ({self.registry.get_variable(var2_name).units})')
                ax.set_title(f'Contour: {var1_name} vs {var2_name} (Epoch {self.current_epoch})')
                
                # Log plot to tensorboard
                if self.logger:
                    self.logger.experiment.add_image(
                        f'contours/{var1_name}_vs_{var2_name}',
                        image_to_buffer(fig),
                        self.current_epoch
                    )
                
                plt.close(fig)
                
            except Exception as e:
                rank_zero_info(f"Failed to generate contour plot for {var1_name} vs {var2_name}: {e}")
                continue
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
            return [optimizer], [scheduler]
        elif self.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mse_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
            return [optimizer], [scheduler]
        else:
            return optimizer
    
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
        train_step_frequency: Optional[int] = None,
        val_step_frequency: Optional[int] = None,
        resolution: Optional[int] = None,
        enable_plotting: Optional[bool] = None
    ):
        """
        Configure contour plotting parameters.
        
        Args:
            eval_pairs: List of (var1, var2) pairs to plot
            train_step_frequency: Plot every N training steps (None to keep current)
            val_step_frequency: Plot every N validation steps (None to keep current)
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
        
        if train_step_frequency is not None:
            self.contour_plot_every_n_train_steps = train_step_frequency
            rank_zero_info(f"Updated training contour plot frequency: every {train_step_frequency} steps")
        
        if val_step_frequency is not None:
            self.contour_plot_every_n_val_steps = val_step_frequency  
            rank_zero_info(f"Updated validation contour plot frequency: every {val_step_frequency} steps")
        
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
            role = self.registry.get_role(var_name).value
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
