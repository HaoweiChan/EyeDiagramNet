import traceback

import torch
import torch.nn as nn
from lightning import LightningModule

from common import utils, losses
from common.init_weights import init_weights

class TraceEWModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        ckpt_path: str = None,
        strict: bool = False,
        ew_scaler: int = 50,
        ew_threshold: float = 0.3
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model

    def setup(self, stage=None):
        # Warm up the model by performing a dummy forward pass
        if stage in ('fit', None):
            loader = self.trainer.datamodule.train_dataloader()
        else:
            loader = self.trainer.datamodule.predict_dataloader()

        dummy_batch, *_ = next(iter(loader))
        if stage in ('fit', None):
            key = next(iter(dummy_batch))
            inputs = dummy_batch[key]
            forward_args = inputs[-1:]
        else:
            forward_args = dummy_batch

        try:
            with torch.no_grad():
                self(*forward_args)
        except (ValueError, RuntimeError) as e:
            utils.log_info(traceback.format_exc())
            self.apply(init_weights('xavier'))

        # Load model checkpoint
        if self.hparams.ckpt_path is not None:
            utils.log_info(f'Loading model checkpoint: {self.hparams.ckpt_path}')
            ckpt = torch.load(self.hparams.ckpt_path, map_location=self.device)
            self.load_state_dict(ckpt['state_dict'], strict=self.hparams.strict)

    def forward(self, *inputs):
        return self.model(*inputs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred_ew, pred_prob = self(*batch)
        pred_ew *= self.hparams.ew_scaler
        pred_ew[pred_prob < self.hparams.ew_threshold] -= 0.1
        mask = pred_prob < self.hparams.ew_threshold
        return pred_ew, pred_prob, mask