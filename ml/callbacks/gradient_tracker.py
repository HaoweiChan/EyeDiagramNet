"""
Gradient Tracking Callback

Usage example:
    from ml.callbacks.gradient_tracker import GradientTracker
    
    # Add to trainer callbacks
    trainer = pl.Trainer(
        callbacks=[
            GradientTracker(
                check_frequency=100,          # Check every 100 steps
                max_display_params=10,        # Show at most 10 unused params
                only_first_epoch=True,        # Only run checks in first epoch
                checkpointed_modules=[        # Modules that use gradient checkpointing
                    'trace_encoder', 
                    'snp_encoder', 
                    'signal_encoder'
                ]
            )
        ]
    )
    
    # The callback will automatically:
    # - Track which parameters receive gradients during training
    # - Report unused parameters that might indicate model issues
    # - Skip parameters from gradient-checkpointed modules
    # - Only run when advanced profiling is enabled to avoid overhead
"""

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class GradientTracker(Callback):
    """
    Callback to track gradient flow and identify parameters that don't receive gradients.
    
    This is useful for debugging model architecture issues, unused parameters, or
    gradient flow problems. Only runs when advanced profiling is enabled to avoid
    training overhead.
    
    Args:
        check_frequency: Check gradients every N steps (default: 100)
        max_display_params: Maximum number of unused parameters to display (default: 10)
        only_first_epoch: Only perform checks during the first epoch (default: True)
        checkpointed_modules: List of module names that use gradient checkpointing
    """
    
    def __init__(
        self,
        check_frequency: int = 100,
        max_display_params: int = 10,
        only_first_epoch: bool = True,
        checkpointed_modules: list[str] = None
    ):
        super().__init__()
        self.check_frequency = check_frequency
        self.max_display_params = max_display_params  
        self.only_first_epoch = only_first_epoch
        self.checkpointed_modules = checkpointed_modules or [
            'trace_encoder', 'snp_encoder', 'signal_encoder'
        ]
        
    def on_before_optimizer_step(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        optimizer
    ) -> None:
        """Track gradient flow and identify unused parameters."""
        
        # Only run on rank 0 for distributed training
        if trainer.global_rank != 0:
            return
            
        # Only run when profiling is enabled to avoid overhead during normal training
        profiling_enabled = (
            hasattr(trainer, 'profiler') and 
            trainer.profiler is not None and
            trainer.profiler.__class__.__name__ not in ['PassThroughProfiler', 'SimpleProfiler']
        )
        
        if not profiling_enabled:
            return
            
        # Check epoch and frequency constraints
        if self.only_first_epoch and trainer.current_epoch > 0:
            return
            
        if trainer.global_step % self.check_frequency != 0:
            return
            
        # Track gradient usage
        unused_params = []
        total_params = 0
        
        # When gradient checkpointing is enabled, it's expected that parameters
        # within the checkpointed modules will not have gradients available at this point
        grad_checkpointing_active = getattr(pl_module.model, 'use_gradient_checkpointing', False)

        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                total_params += 1

                # Skip parameters from gradient-checkpointed modules
                if grad_checkpointing_active and any(
                    mod_name in name for mod_name in self.checkpointed_modules
                ):
                    continue

                if param.grad is None:
                    unused_params.append(name)
        
        # Report findings
        if unused_params:
            rank_zero_info(
                f"Step {trainer.global_step} - Found {len(unused_params)}/{total_params} unused parameters:"
            )
            
            # Display limited number of parameter names to avoid spam
            for param_name in unused_params[:self.max_display_params]:
                rank_zero_info(f"  - {param_name}")
                
            if len(unused_params) > self.max_display_params:
                remaining = len(unused_params) - self.max_display_params
                rank_zero_info(f"  ... and {remaining} more")
                
            # Add warning if this might indicate a problem
            if len(unused_params) > total_params * 0.1:  # More than 10% unused
                rank_zero_info(
                    f"Warning: {len(unused_params)/total_params:.1%} of parameters "
                    f"are not receiving gradients - this might indicate model architecture issues"
                )
        else:
            rank_zero_info(f"Step {trainer.global_step} - All {total_params} parameters received gradients")
            
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log callback initialization."""
        if trainer.global_rank == 0:
            profiling_enabled = (
                hasattr(trainer, 'profiler') and 
                trainer.profiler is not None and
                trainer.profiler.__class__.__name__ not in ['PassThroughProfiler', 'SimpleProfiler']
            )
            
            if profiling_enabled:
                rank_zero_info(
                    f"GradientTracker enabled: checking every {self.check_frequency} steps"
                    + (f" (first epoch only)" if self.only_first_epoch else "")
                )
            else:
                rank_zero_info("GradientTracker disabled: advanced profiling not enabled")
