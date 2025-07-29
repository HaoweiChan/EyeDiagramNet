from typing import Any
import torch
import numpy as np
import random
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from ..utils.visualization import plot_sparam_reconstruction, image_to_tensor

class SParameterVisualizer(Callback):
    """
    Callback to visualize S-parameter reconstruction at the end of each epoch.
    """
    def __init__(self, num_samples: int = 1):
        super().__init__()
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Logs a plot of a random sample's S-parameter reconstruction.
        """
        if not hasattr(trainer.datamodule, 'train_dataset'):
            return

        # Get a random sample from the training dataset
        dataset = trainer.datamodule.train_dataset
        sample_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[sample_idx]
        
        # Move sample to the model's device and add a batch dimension
        true_snp_tensor = sample['snp_vert'].to(pl_module.device).unsqueeze(0)
        
        # Get the model's reconstruction
        pl_module.eval()
        with torch.no_grad():
            with torch.autocast(device_type=pl_module.device.type, enabled=True):
                recon_snp_tensor, _ = pl_module(true_snp_tensor)
        pl_module.train()
        
        # Convert tensors to numpy for plotting
        true_snp = true_snp_tensor.squeeze(0).cpu().numpy()
        recon_snp = recon_snp_tensor.squeeze(0).cpu().numpy()
        
        # Determine frequency points (assuming linear space for visualization)
        num_freq_points = true_snp.shape[0]
        freqs = np.linspace(1e6, 10e9, num_freq_points) # Placeholder frequencies

        # Select two random ports to plot
        num_ports = true_snp.shape[1]
        port1, port2 = random.sample(range(num_ports), 2)
        
        # Generate the plot
        image = plot_sparam_reconstruction(
            freqs=freqs,
            true_sparam=true_snp[:, port1, port2],
            recon_sparam=recon_snp[:, port1, port2],
            port1=port1,
            port2=port2,
            title=f"Epoch {trainer.current_epoch}"
        )
        
        # Log the image to TensorBoard
        tensorboard_logger = trainer.logger.experiment
        tensorboard_logger.add_image(
            f"S-Parameter Reconstruction",
            image_to_tensor(image),
            global_step=trainer.current_epoch
        ) 