"""
Test Gaussian Process Module with Fake Data and Contour Plotting

Creates synthetic data compatible with ContourDataModule structure
and trains GaussianProcessModule to show contour plots.
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

# Check for gpytorch
try:
    import gpytorch
    GPYTORCH_AVAILABLE = True
except ImportError:
    print("WARNING: gpytorch not available. Install with 'pip install gpytorch'")
    print("Proceeding with visualization only...")
    GPYTORCH_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.variable_registry import VariableRegistry, VariableRole
from torch.utils.data import Dataset, DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

if GPYTORCH_AVAILABLE:
    from ml.modules.contour_module import GaussianProcessModule


class FakeContourDataset(Dataset):
    """
    Generate synthetic contour data for testing GP module.
    
    Simulates eye width as a function of design parameters with realistic patterns:
    - Non-linear dependencies
    - Parameter interactions
    - Noise
    """
    
    def __init__(
        self, 
        n_samples: int = 200,
        variable_registry: VariableRegistry = None,
        seed: int = 42
    ):
        super().__init__()
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.registry = variable_registry or self._create_default_registry()
        self.n_samples = n_samples
        
        # Generate synthetic variable data
        self.variables = self._generate_variables()
        
        # Generate synthetic eye widths based on parameter interactions
        self.eye_widths = self._generate_eye_widths()
        
        # Create case IDs
        self.case_ids = list(range(n_samples))
    
    def _create_default_registry(self) -> VariableRegistry:
        """Create default variable registry for testing."""
        registry = VariableRegistry()
        
        # Register boundary parameters (R, L, C)
        registry.register_variable('R_drv', role=VariableRole.BOUNDARY, bounds=(30.0, 70.0))
        registry.register_variable('R_odt', role=VariableRole.BOUNDARY, bounds=(30.0, 70.0))
        registry.register_variable('L_drv', role=VariableRole.BOUNDARY, bounds=(0.1, 0.5))
        registry.register_variable('C_pkg', role=VariableRole.BOUNDARY, bounds=(0.1, 0.8))
        
        # Register geometric parameters
        registry.register_variable('H_substrate', role=VariableRole.HEIGHT, bounds=(0.5, 2.0))
        registry.register_variable('W_trace', role=VariableRole.WIDTH, bounds=(0.1, 0.5))
        
        return registry
    
    def _generate_variables(self) -> Dict[str, np.ndarray]:
        """Generate random variable values within bounds."""
        variables = {}
        
        for var_name in self.registry.variable_names:
            var_info = self.registry.get_variable(var_name)
            if var_info.bounds is not None:
                min_val, max_val = var_info.bounds
                # Uniform sampling within bounds
                values = np.random.uniform(min_val, max_val, self.n_samples)
                variables[var_name] = values
        
        return variables
    
    def _generate_eye_widths(self) -> np.ndarray:
        """
        Generate synthetic eye widths with realistic parameter dependencies.
        
        Eye width model:
        - Increases with driver resistance (better impedance matching)
        - Decreases with package capacitance (signal degradation)
        - Non-linear interaction between trace width and substrate height
        - Gaussian noise
        """
        # Extract variables for computation
        R_drv = self.variables.get('R_drv', np.ones(self.n_samples) * 50.0)
        R_odt = self.variables.get('R_odt', np.ones(self.n_samples) * 50.0)
        L_drv = self.variables.get('L_drv', np.ones(self.n_samples) * 0.3)
        C_pkg = self.variables.get('C_pkg', np.ones(self.n_samples) * 0.4)
        H_sub = self.variables.get('H_substrate', np.ones(self.n_samples) * 1.0)
        W_tr = self.variables.get('W_trace', np.ones(self.n_samples) * 0.3)
        
        # Normalize to [0, 1] for easier modeling
        R_drv_norm = (R_drv - 30.0) / 40.0
        C_pkg_norm = (C_pkg - 0.1) / 0.7
        H_sub_norm = (H_sub - 0.5) / 1.5
        W_tr_norm = (W_tr - 0.1) / 0.4
        
        # Base eye width with parameter interactions
        eye_width = (
            0.5  # Base value
            + 0.3 * R_drv_norm  # Positive correlation with driver resistance
            - 0.2 * C_pkg_norm  # Negative correlation with capacitance
            + 0.15 * np.sin(2 * np.pi * H_sub_norm)  # Non-linear substrate effect
            + 0.1 * W_tr_norm * (1 - C_pkg_norm)  # Interaction: trace width helps when C is low
            - 0.05 * R_drv_norm * C_pkg_norm  # Negative interaction
        )
        
        # Add Gaussian noise
        noise = np.random.normal(0, 0.02, self.n_samples)
        eye_width = eye_width + noise
        
        # Clip to realistic range [0.1, 1.0]
        eye_width = np.clip(eye_width, 0.1, 1.0)
        
        return eye_width
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict:
        """Get sample compatible with GaussianProcessModule."""
        variables = {}
        for name, values in self.variables.items():
            variables[name] = torch.tensor(values[idx], dtype=torch.float32)
        
        return {
            'variables': variables,
            'targets': torch.tensor([self.eye_widths[idx]], dtype=torch.float32),
            'case_id': self.case_ids[idx],
            'active_variables': list(variables.keys())
        }

def create_fake_dataloader(
    n_samples: int = 200,
    batch_size: int = 32,
    registry: VariableRegistry = None
) -> DataLoader:
    """Create DataLoader with fake contour data."""
    dataset = FakeContourDataset(n_samples=n_samples, variable_registry=registry)
    
    def collate_fn(batch):
        """Custom collate function for variable batches."""
        all_variables = {}
        targets = []
        case_ids = []
        active_variables_list = []
        
        for item in batch:
            for var_name, var_value in item['variables'].items():
                if var_name not in all_variables:
                    all_variables[var_name] = []
                all_variables[var_name].append(var_value)
            
            targets.append(item['targets'])
            case_ids.append(item['case_id'])
            active_variables_list.append(item['active_variables'])
        
        # Stack variables
        batched_variables = {}
        for var_name, var_list in all_variables.items():
            batched_variables[var_name] = torch.stack(var_list, dim=0)
        
        batched_targets = torch.stack(targets, dim=0)
        
        return {
            'variables': batched_variables,
            'targets': batched_targets,
            'case_ids': case_ids,
            'active_variables': active_variables_list
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

def plot_ground_truth_contours(
    dataset: FakeContourDataset,
    var1_name: str = 'R_drv',
    var2_name: str = 'C_pkg',
    resolution: int = 30
) -> plt.Figure:
    """Plot ground truth contours for visual validation."""
    var1_bounds = dataset.registry.get_bounds(var1_name)
    var2_bounds = dataset.registry.get_bounds(var2_name)
    
    # Create grid
    var1_grid = np.linspace(var1_bounds[0], var1_bounds[1], resolution)
    var2_grid = np.linspace(var2_bounds[0], var2_bounds[1], resolution)
    var1_mesh, var2_mesh = np.meshgrid(var1_grid, var2_grid)
    
    # Get fixed values for other variables (use mean)
    fixed_vars = {}
    for var_name in dataset.registry.variable_names:
        if var_name not in [var1_name, var2_name]:
            fixed_vars[var_name] = np.mean(dataset.variables[var_name])
    
    # Compute eye width for each grid point
    eye_width_grid = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            # Create temporary dataset with single point
            temp_vars = fixed_vars.copy()
            temp_vars[var1_name] = var1_mesh[i, j]
            temp_vars[var2_name] = var2_mesh[i, j]
            
            # Compute eye width using the same formula
            R_drv = temp_vars.get('R_drv', 50.0)
            C_pkg = temp_vars.get('C_pkg', 0.4)
            H_sub = temp_vars.get('H_substrate', 1.0)
            W_tr = temp_vars.get('W_trace', 0.3)
            
            R_drv_norm = (R_drv - 30.0) / 40.0
            C_pkg_norm = (C_pkg - 0.1) / 0.7
            H_sub_norm = (H_sub - 0.5) / 1.5
            W_tr_norm = (W_tr - 0.1) / 0.4
            
            eye_width_grid[i, j] = (
                0.5
                + 0.3 * R_drv_norm
                - 0.2 * C_pkg_norm
                + 0.15 * np.sin(2 * np.pi * H_sub_norm)
                + 0.1 * W_tr_norm * (1 - C_pkg_norm)
                - 0.05 * R_drv_norm * C_pkg_norm
            )
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(var1_mesh, var2_mesh, eye_width_grid, levels=15, cmap='viridis')
    ax.contour(var1_mesh, var2_mesh, eye_width_grid, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Eye Width (UI)', fontsize=12)
    
    ax.set_xlabel(f'{var1_name}', fontsize=12)
    ax.set_ylabel(f'{var2_name}', fontsize=12)
    ax.set_title(f'Ground Truth: Eye Width Contour ({var1_name} vs {var2_name})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def test_gp_module():
    """Test GaussianProcessModule with fake data and generate contour plots."""
    print("=" * 80)
    print("Testing Gaussian Process Module with Fake Contour Data")
    print("=" * 80)
    
    # Create registry
    registry = VariableRegistry()
    registry.register_variable('R_drv', role=VariableRole.BOUNDARY, bounds=(30.0, 70.0))
    registry.register_variable('R_odt', role=VariableRole.BOUNDARY, bounds=(30.0, 70.0))
    registry.register_variable('L_drv', role=VariableRole.BOUNDARY, bounds=(0.1, 0.5))
    registry.register_variable('C_pkg', role=VariableRole.BOUNDARY, bounds=(0.1, 0.8))
    registry.register_variable('H_substrate', role=VariableRole.HEIGHT, bounds=(0.5, 2.0))
    registry.register_variable('W_trace', role=VariableRole.WIDTH, bounds=(0.1, 0.5))
    
    # Create data
    print("\n1. Creating fake dataset...")
    dataset = FakeContourDataset(n_samples=150, variable_registry=registry)
    print(f"   Generated {len(dataset)} samples with {len(dataset.variables)} variables")
    
    # Create dataloaders
    # Note: GP training requires full-batch, so batch_size = n_samples
    train_loader = create_fake_dataloader(n_samples=120, batch_size=120, registry=registry)  
    val_loader = create_fake_dataloader(n_samples=30, batch_size=30, registry=registry)
    
    # Plot ground truth contours for all pairs
    print("\n2. Plotting ground truth contours for all pairs...")
    contour_pairs = [('R_drv', 'C_pkg'), ('H_substrate', 'W_trace'), ('L_drv', 'R_odt')]
    ground_truth_dir = Path('saved/test_gp_contour/ground_truth')
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    for var1_name, var2_name in contour_pairs:
        gt_fig = plot_ground_truth_contours(dataset, var1_name=var1_name, var2_name=var2_name)
        gt_save_path = ground_truth_dir / f'ground_truth_{var1_name}_vs_{var2_name}.png'
        gt_fig.savefig(gt_save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {gt_save_path}")
        plt.close(gt_fig)
    
    # If gpytorch not available, return early
    if not GPYTORCH_AVAILABLE:
        print("\n" + "=" * 80)
        print("⚠️  GPyTorch not available - skipping GP training")
        print("✓  Ground truth contour plot generated successfully")
        print(f"   Saved at: {gt_save_path}")
        print("\nTo test GP module, install gpytorch:")
        print("   pip install gpytorch")
        print("=" * 80)
        return None, dataset
    
    # Create GP module
    print("\n3. Creating Gaussian Process module...")
    # Define multiple contour pairs to evaluate
    contour_pairs = [('R_drv', 'C_pkg'), ('H_substrate', 'W_trace'), ('L_drv', 'R_odt')]
    
    gp_module = GaussianProcessModule(
        variable_registry=registry,
        use_ard=True,
        eval_contour_pairs=contour_pairs,
        eval_resolution=25,
        save_contour_plots=True,
        spec_threshold=0.4,
        var1_name=None,  # Let it randomly select from eval_contour_pairs
        var2_name=None
    )
    print("   GP module initialized")
    print(f"   Will plot contours for pairs: {contour_pairs}")
    
    # Create trainer
    print("\n4. Setting up Lightning Trainer...")
    logger = TensorBoardLogger(
        save_dir='saved',
        name='test_gp_contour'
    )
    
    trainer = Trainer(
        max_epochs=200,  # Test with more epochs to check convergence
        logger=logger,
        accelerator='cpu',
        devices=1,
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        num_sanity_val_steps=0  # Disable sanity check to avoid R2 metric error
    )
    
    # Train
    print("\n5. Training Gaussian Process model...")
    print("   (This will optimize GP hyperparameters)")
    trainer.fit(gp_module, train_loader, val_loader)
    
    # Get version folder for saving plots
    version_folder = Path(logger.log_dir)
    print(f"\n6. Saving results to: {version_folder}")
    
    print("\n7. Generating GP contour predictions for all pairs...")
    from ml.utils.visualization import plot_contour_2d
    
    resolution = 30
    gp_module.eval()
    
    # Generate plots for all contour pairs
    for var1_name, var2_name in contour_pairs:
        print(f"   Generating contours for {var1_name} vs {var2_name}...")
        
        # Get variable bounds
        var1_bounds = registry.get_bounds(var1_name)
        var2_bounds = registry.get_bounds(var2_name)
        
        # Create grid
        var1_grid = np.linspace(var1_bounds[0], var1_bounds[1], resolution)
        var2_grid = np.linspace(var2_bounds[0], var2_bounds[1], resolution)
        var1_mesh, var2_mesh = np.meshgrid(var1_grid, var2_grid)
        
        # Get fixed values for other variables (use dataset mean)
        fixed_vars = {}
        for var_name in registry.variable_names:
            if var_name not in [var1_name, var2_name]:
                fixed_vars[var_name] = np.mean(dataset.variables[var_name])
        
        # Prepare test data
        test_variables = {}
        for var_name in registry.variable_names:
            if var_name == var1_name:
                test_variables[var_name] = torch.from_numpy(var1_mesh.flatten()).float()
            elif var_name == var2_name:
                test_variables[var_name] = torch.from_numpy(var2_mesh.flatten()).float()
            else:
                test_variables[var_name] = torch.full((resolution * resolution,), fixed_vars[var_name])
        
        # Predict
        predictions, uncertainties = gp_module.predict(test_variables)
        pred_grid = predictions.detach().cpu().numpy().reshape(resolution, resolution)
        unc_grid = uncertainties.detach().cpu().numpy().reshape(resolution, resolution)
        
        # Save uncertainty plot
        fig_unc, axes_unc = plt.subplots(1, 2, figsize=(16, 6))
        
        # Predictions
        ax = axes_unc[0]
        contour = ax.contourf(var1_mesh, var2_mesh, pred_grid, levels=15, cmap='viridis')
        ax.contour(var1_mesh, var2_mesh, pred_grid, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Predicted Eye Width (UI)', fontsize=11)
        ax.set_xlabel(var1_name, fontsize=12)
        ax.set_ylabel(var2_name, fontsize=12)
        ax.set_title(f'GP Predictions: {var1_name} vs {var2_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Uncertainty
        ax = axes_unc[1]
        contour = ax.contourf(var1_mesh, var2_mesh, unc_grid, levels=15, cmap='plasma')
        ax.contour(var1_mesh, var2_mesh, unc_grid, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Prediction Uncertainty (std)', fontsize=11)
        ax.set_xlabel(var1_name, fontsize=12)
        ax.set_ylabel(var2_name, fontsize=12)
        ax.set_title('GP Uncertainty Quantification', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        gp_unc_path = version_folder / f'gp_uncertainty_{var1_name}_vs_{var2_name}.png'
        fig_unc.savefig(gp_unc_path, dpi=150, bbox_inches='tight')
        plt.close(fig_unc)
        
        # Get actual data points for error calculation
        actual_var1 = dataset.variables[var1_name]
        actual_var2 = dataset.variables[var2_name]
        actual_ew = dataset.eye_widths
        
        # Prepare actual test variables
        actual_test_vars = {}
        for var_name in registry.variable_names:
            if var_name == var1_name:
                actual_test_vars[var_name] = torch.from_numpy(actual_var1).float()
            elif var_name == var2_name:
                actual_test_vars[var_name] = torch.from_numpy(actual_var2).float()
            else:
                actual_test_vars[var_name] = torch.full((len(actual_var1),), fixed_vars[var_name])
        
        # Predict on actual data points
        actual_predictions, _ = gp_module.predict(actual_test_vars)
        errors = np.abs(actual_predictions.detach().cpu().numpy() - actual_ew)
        
        # Prepare error data for visualization
        error_data = {
            'var1_values': actual_var1,
            'var2_values': actual_var2,
            'errors': errors
        }
        
        # Create the dual-plot using plot_contour_2d
        fig_dual = plot_contour_2d(
            var1_name=var1_name,
            var2_name=var2_name,
            var1_grid=torch.from_numpy(var1_mesh),
            var2_grid=torch.from_numpy(var2_mesh),
            predictions=torch.from_numpy(pred_grid),
            spec_threshold=0.4,
            error_data=error_data
        )
        
        gp_dual_path = version_folder / f'gp_predictions_{var1_name}_vs_{var2_name}.png'
        fig_dual.savefig(gp_dual_path, dpi=150, bbox_inches='tight')
        print(f"     - Uncertainty: {gp_unc_path.name}")
        print(f"     - Predictions: {gp_dual_path.name}")
        plt.close(fig_dual)
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print(f"Results saved in: {version_folder}")
    print("=" * 80)
    
    return gp_module, dataset

if __name__ == '__main__':
    # Run test
    gp_module, dataset = test_gp_module()
    
    print("\n✓ Gaussian Process module works smoothly with fake contour data")
    print("✓ Contour plots generated successfully")
    print("\nCheck the saved plots at: saved/test_gp_contour/")

