import torch
from lightning import LightningModule

from ..models.snp_model import SNPEmbedding, OptimizedSNPEmbedding, SNPDecoder
from ..utils.snp_losses import SNPReconstructionLoss

class SNPSelfSupervisedModule(LightningModule):
    """Lightning module for self-supervised SNP embedding pretraining."""
    
    def __init__(
        self,
        model_dim: int = 768,
        freq_length: int = 201,
        encoder_type: str = 'OptimizedSNPEmbedding',
        decoder_hidden_ratio: int = 2,
        reconstruction_loss: str = 'complex_mse',
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
            
        self.decoder = SNPDecoder(
            model_dim=model_dim, freq_length=freq_length,
            decoder_hidden_ratio=decoder_hidden_ratio,
            use_checkpointing=use_gradient_checkpointing,
            use_mixed_precision=use_mixed_precision
        )
        
        self.loss_fn = SNPReconstructionLoss(
            loss_type=reconstruction_loss,
            latent_reg_type=latent_regularization_type,
            latent_reg_weight=latent_regularization_weight
        )
        
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
        
        # Log training loss for checkpointing
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        
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