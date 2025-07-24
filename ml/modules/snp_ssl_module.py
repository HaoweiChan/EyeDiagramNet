import torch
import torch.nn as nn
from lightning import LightningModule

from ..models.snp_model import SNPEmbedding, OptimizedSNPEmbedding

class SNPSelfSupervisedModule(LightningModule):
    """Lightning module for self-supervised SNP embedding pretraining."""
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss_fn: nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'decoder', 'loss_fn'])
        
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        
        self.best_val_loss = float('inf')

    def forward(self, snp_vert):
        """
        Encodes and decodes the S-parameter tensor.
        The output is resized to match the input frequency dimension.
        """
        output_freq_length = snp_vert.shape[1]
        hidden_states = self.encoder(snp_vert)
        reconstructed = self.decoder(hidden_states, output_freq_length=output_freq_length)
        if reconstructed.dim() == 5 and snp_vert.dim() == 4:
            reconstructed = reconstructed.squeeze(1)
        return reconstructed, hidden_states

    def training_step(self, batch, batch_idx):
        snp_vert = batch['snp_vert'] if isinstance(batch, dict) else batch
        reconstructed, hidden_states = self(snp_vert)
        loss, loss_dict = self.loss_fn(reconstructed, snp_vert, hidden_states)
        
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('train/magnitude_loss', loss_dict.get('log_magnitude_loss', 0), on_epoch=True, on_step=False, sync_dist=True)
        self.log('train/phase_loss', loss_dict.get('phase_loss', 0), on_epoch=True, on_step=False, sync_dist=True)
        
        if 'complex_cosine_loss' in loss_dict:
            self.log('train/complex_cosine_loss', loss_dict['complex_cosine_loss'], on_epoch=True, on_step=False, sync_dist=True)
        if 'spectral_loss' in loss_dict:
            self.log('train/spectral_loss', loss_dict['spectral_loss'], on_epoch=True, on_step=False, sync_dist=True)
        if 'gradient_penalty' in loss_dict:
            self.log('train/gradient_penalty', loss_dict['gradient_penalty'], on_epoch=True, on_step=False, sync_dist=True)
        if 'regularization' in loss_dict:
            self.log('regularization_loss', loss_dict['regularization'], on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint['encoder_state_dict'] = self.encoder.state_dict()
        encoder_config = {
            'encoder_type': self.encoder.__class__.__name__
        }
        if hasattr(self.encoder, 'model_dim'):
            encoder_config['model_dim'] = self.encoder.model_dim
        if hasattr(self.encoder, 'freq_length'):
            encoder_config['freq_length'] = self.encoder.freq_length
        checkpoint['encoder_config'] = encoder_config

    @classmethod
    def load_encoder_from_checkpoint(cls, checkpoint_path, map_location=None):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        config = checkpoint['encoder_config']
        
        model_dim = config.get('model_dim')
        freq_length = config.get('freq_length')
        encoder_type = config.get('encoder_type')

        if not all([model_dim, freq_length, encoder_type]):
            raise ValueError("Checkpoint config is missing required keys ('model_dim', 'freq_length', 'encoder_type').")

        if encoder_type == 'SNPEmbedding':
            encoder = SNPEmbedding(model_dim=model_dim, freq_length=freq_length)
        elif encoder_type == 'OptimizedSNPEmbedding':
            encoder = OptimizedSNPEmbedding(model_dim=model_dim, freq_length=freq_length)
        else:
            raise ValueError(f"Unknown encoder type in checkpoint: {encoder_type}")
            
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        return encoder 