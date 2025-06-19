import torch
from lightning import LightningModule

from ..models.snp_model import SNPEmbedding, OptimizedSNPEmbedding, SNPDecoder, ImprovedSNPDecoder
from ..utils.snp_losses import SNPReconstructionLoss, ImprovedSNPReconstructionLoss

class SNPSelfSupervisedModule(LightningModule):
    """Lightning module for self-supervised SNP embedding pretraining."""
    
    def __init__(
        self,
        model_dim: int = 768,
        freq_length: int = 201,
        encoder_type: str = 'OptimizedSNPEmbedding',
        decoder_type: str = 'SNPDecoder',
        decoder_hidden_ratio: int = 2,
        num_decoder_layers: int = 3,
        use_skip_connections: bool = True,
        use_separate_phase_mag: bool = True,
        dropout_rate: float = 0.1,
        reconstruction_loss: SNPReconstructionLoss = None,
        loss_type: str = 'SNPReconstructionLoss',
        magnitude_weight: float = 1.0,
        phase_weight: float = 0.5,
        complex_weight: float = 0.5,
        spectral_weight: float = 0.2,
        use_unwrapped_phase: bool = True,
        gradient_penalty_weight: float = 0.0,
        latent_regularization_type: str = 'l2',
        latent_regularization_weight: float = 0.01,
        use_gradient_checkpointing: bool = False,
        use_mixed_precision: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if encoder_type == 'SNPEmbedding':
            self.encoder = SNPEmbedding(model_dim=model_dim, freq_length=freq_length, use_tx_rx_tokens=False)
        elif encoder_type == 'OptimizedSNPEmbedding':
            self.encoder = OptimizedSNPEmbedding(
                model_dim=model_dim, freq_length=freq_length,
                use_checkpointing=use_gradient_checkpointing,
                use_mixed_precision=use_mixed_precision,
                use_tx_rx_tokens=False
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        if decoder_type == 'SNPDecoder':
            self.decoder = SNPDecoder(
                model_dim=model_dim, freq_length=freq_length,
                decoder_hidden_ratio=decoder_hidden_ratio,
                use_checkpointing=use_gradient_checkpointing,
                use_mixed_precision=use_mixed_precision
            )
        elif decoder_type == 'ImprovedSNPDecoder':
            self.decoder = ImprovedSNPDecoder(
                model_dim=model_dim, 
                freq_length=freq_length,
                decoder_hidden_ratio=decoder_hidden_ratio,
                num_decoder_layers=num_decoder_layers,
                use_skip_connections=use_skip_connections,
                use_separate_phase_mag=use_separate_phase_mag,
                dropout_rate=dropout_rate,
                use_checkpointing=use_gradient_checkpointing,
                use_mixed_precision=use_mixed_precision
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        
        if reconstruction_loss is not None:
            self.loss_fn = reconstruction_loss
        elif loss_type == 'SNPReconstructionLoss':
            self.loss_fn = SNPReconstructionLoss(
                magnitude_weight=magnitude_weight,
                phase_weight=phase_weight,
                latent_reg_type=latent_regularization_type,
                latent_reg_weight=latent_regularization_weight
            )
        elif loss_type == 'ImprovedSNPReconstructionLoss':
            self.loss_fn = ImprovedSNPReconstructionLoss(
                magnitude_weight=magnitude_weight,
                phase_weight=phase_weight,
                complex_weight=complex_weight,
                spectral_weight=spectral_weight,
                use_unwrapped_phase=use_unwrapped_phase,
                latent_reg_type=latent_regularization_type,
                latent_reg_weight=latent_regularization_weight,
                gradient_penalty_weight=gradient_penalty_weight
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
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
        
        # Log training loss for checkpointing and visualization
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('train/magnitude_loss', loss_dict['log_magnitude_loss'], on_epoch=True, on_step=False, sync_dist=True)
        self.log('train/phase_loss', loss_dict['phase_loss'], on_epoch=True, on_step=False, sync_dist=True)
        
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
        checkpoint['encoder_config'] = {
            'model_dim': self.hparams.model_dim,
            'freq_length': self.hparams.freq_length,
            'encoder_type': self.hparams.encoder_type
        }

    @classmethod
    def load_encoder_from_checkpoint(cls, checkpoint_path, map_location=None):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        config = checkpoint['encoder_config']
        if config['encoder_type'] == 'SNPEmbedding':
            encoder = SNPEmbedding(model_dim=config['model_dim'], freq_length=config['freq_length'])
        else:
            encoder = OptimizedSNPEmbedding(model_dim=config['model_dim'], freq_length=config['freq_length'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        return encoder 