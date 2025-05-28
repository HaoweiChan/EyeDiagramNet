import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from lightning import LightningModule

from common import utils, losses
from common.init_weights import init_weights
from models.layers import UncertaintyWeightedLoss, LearnableLossWeighting, GradNormLossBalancer

class TraceEWModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        ckpt_path: str = None,
        strict: bool = False,
        bce_weight: int = 10,
        ew_scaler: int = 50,
        ew_threshold: float = 0.3,
        mc_samples: int = 50,
        use_mc_validation: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.train_step_outputs = {}
        self.val_step_outputs = {}

        self.metrics = nn.ModuleDict({
            'train': self.metrics_factory(),
            'val': self.metrics_factory(),
        })
        self.ew_scaler = torch.tensor(self.hparams.ew_scaler)
        # self.weighted_loss = UncertaintyWeightedLoss(['nll', 'bce'])
        # self.weighted_loss = LearnableLossWeighting(['nll', 'bce'])
        self.weighted_loss = GradNormLossBalancer(['nll', 'bce'])

    def setup(self, stage=None):
        # Warm up the model by performing a dummy forward pass
        if stage in ('fit', None):
            loader = self.trainer.datamodule.train_dataloader()
        else:
            loader = self.trainer.datamodule.predict_dataloader()
        
        dummy_batch = next(iter(loader))
        if stage in ('fit', None):
            key = next(iter(dummy_batch.keys()))
            inputs = dummy_batch[key]
            forward_args = inputs[:-1]
        else:
            forward_args = dummy_batch

        try:
            with torch.no_grad():
                self(*forward_args)
        except (ValueError, RuntimeError) as e:
            self.utils.log.info(traceback.format_exc())
            raise
        self.apply(init_weights('xavier'))

        # load model checkpoint
        if self.hparams.ckpt_path is not None:
            self.utils.log.info(f'Loading model checkpoint: {self.hparams.ckpt_path}')
            ckpt = torch.load(self.hparams.ckpt_path, map_location=self.device)
            self.load_state_dict(ckpt['state_dict'], strict=self.hparams.strict)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    ############################ TRAIN & VALIDATION ############################

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, batch_idx, "train", dataloader_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, batch_idx, "val", dataloader_idx)

    def on_train_epoch_end(self):
        log_metrics = self.compute_metrics("train")
        if self.current_epoch % self.trainer.check_val_every_n_epoch == 0:
            for dataloader_idx in self.train_step_outputs.keys():
                self.plot_sparam_curve("train", log_metrics, self.train_step_outputs[dataloader_idx][0], dataloader_idx)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        log_metrics = self.compute_metrics("val")
        for dataloader_idx in self.val_step_outputs.keys():
            self.plot_sparam_curve("val", log_metrics, self.val_step_outputs[dataloader_idx][0], dataloader_idx)
        self.val_step_outputs.clear()

    ############################ INFERENCE ############################

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if hasattr(self.model, 'predict_with_uncertainty'):
            pred_ew, total_var, aleatoric_var, epistemic_var, pred_logits = self.model.predict_with_uncertainty(
                *batch, n_samples=self.hparams.mc_samples
            )
            pred_prob = torch.sigmoid(pred_logits)
        else:
            pred_ew, pred_logvar, pred_logits = self(*batch)
            pred_prob = torch.sigmoid(pred_logits)
        
        pred_ew = pred_ew * self.ew_scaler
        pred_ew[pred_prob < self.hparams.ew_threshold] = -0.1
        return pred_ew

    ############################ PRIVATE METHODS ############################

    def convert_metric_name(self, stage):
        if stage == "train":
            return "train"
        return stage

    def metrics_factory(self):
        metrics = {
            'loss': tm.MeanMetric,
            'mae': tm.MeanAbsoluteError,
            'mape': tm.WeightedMeanAbsolutePercentageError,
            'r2': tm.R2Score,
            'accuracy': tm.classification.BinaryAccuracy,
            'f1': tm.classification.BinaryF1Score,
            'cov': tm.classification.BinaryAccuracy,
            'cov2s': tm.classification.BinaryAccuracy,
        }
        metrics_dict = nn.ModuleDict()
        for k, metric in metrics.items():
            if 'accuracy' in k or 'f1' in k:
                metrics_dict[k] = metric(threshold=self.hparams.ew_threshold)
            else:
                metrics_dict[k] = metric()
        return metrics_dict

    def get_output_steps(self, stage):
        if stage == "train":
            max_batches = self.trainer.num_training_batches
        else:
            max_batches = getattr(self.trainer, f'num_{stage}_batches')[0]
        return (self.current_epoch * max_batches, max_batches - 1)

    def augment_input_sequence(self, seq):
        insert_len = torch.randint(0, seq.size(1) // 10, (1,)).item()
        insert_idx = torch.randperm(seq.size(1) + insert_len)[:insert_len]
        orig_idx = torch.sort(insert_idx)[1].values
        extended_seq = (-1) * torch.ones(seq.size(0), seq.size(1) + insert_len, seq.size(-1), device=seq.device, dtype=seq.dtype)
        orig_idx = torch.arange(seq.size(1) + insert_len)
        extended_seq[:, orig_idx[torch.isin(orig_idx, insert_idx)], :] = seq
        return extended_seq

    def step(self, batch, batch_idx, stage, dataloader_idx):
        """
        Arguments:
            batch:
                input: (B, E)
                true: (B, F, P, P)
                snp_ports: (B, E)
        """
        loss = 0
        for name, batch_one in batch.items():
            trace_seq, direction, boundary, snp_vert, true_ew = batch_one
            true_ew = true_ew / self.ew_scaler

            if stage == "train":
                trace_seq = self.augment_input_sequence(trace_seq)

            # Make the true probability for non-closed eye
            true_prob = true_ew.clone()
            true_prob[true_prob > 0] = 1
            weight_prob = true_prob.clone()
            weight_prob[weight_prob == 0] = 10
            weight_prob = weight_prob / weight_prob.sum()

            # Use Monte Carlo inference for validation if enabled
            if stage == "val" and self.hparams.use_mc_validation and hasattr(self.model, 'predict_with_uncertainty'):
                # Use MC inference for better uncertainty estimation
                pred_ew, total_var, aleatoric_var, epistemic_var, pred_logits = self.model.predict_with_uncertainty(
                    trace_seq, direction, boundary, snp_vert, n_samples=self.hparams.mc_samples
                )
                pred_prob = torch.sigmoid(pred_logits)
                
                # Use total variance for loss computation
                pred_logvar = torch.log(total_var + 1e-8)
                
                # Forward the model normally for loss computation (to get gradients)
                pred_ew_train, pred_logvar_train, pred_prob_train, hidden_states = self(
                    trace_seq, direction, boundary, snp_vert, output_hidden_states=True
                )
                
                loss += self.weighted_loss({
                    'nll': losses.gaussian_nll_loss(pred_ew_train, true_ew, pred_logvar_train, mask=true_prob),
                    'bce': F.binary_cross_entropy_with_logits(pred_prob_train, true_prob, weight=weight_prob)
                }, hidden_states)
            else:
                # Standard forward pass for training
                pred_ew, pred_logvar, pred_prob, hidden_states = self(trace_seq, direction, boundary, snp_vert, output_hidden_states=True)
                pred_prob = torch.sigmoid(pred_prob)
                
                loss += self.weighted_loss({
                    'nll': losses.gaussian_nll_loss(pred_ew, true_ew, pred_logvar, mask=true_prob),
                    'bce': F.binary_cross_entropy_with_logits(pred_prob, true_prob, weight=weight_prob)
                }, hidden_states)

            pred_ew = pred_ew * self.ew_scaler
            true_ew = true_ew * self.ew_scaler
            pred_logvar = pred_logvar + 2 + torch.log(self.ew_scaler)
            pred_sigma = torch.exp(0.5 * pred_logvar)
            self.update_metrics(stage, loss.detach(), pred_ew, true_ew, pred_prob, pred_sigma)
            
            if batch_idx in self.get_output_steps(stage):
                idx = torch.randint(len(pred_ew), (1,)).item()
                step_output = {
                    'pred_ew': pred_ew[idx],
                    'true_ew': true_ew[idx],
                    'pred_prob': pred_prob[idx],
                    'true_prob': true_prob[idx],
                    'pred_sigma': pred_sigma[idx]
                }
                if stage == "train":
                    self.train_step_outputs.setdefault(name, []).append(step_output)
                else:
                    self.val_step_outputs.setdefault(name, []).append(step_output)

        prog_bar = True if stage == "train" else False
        loss = loss / len(batch)
        self.log("loss", loss, on_step=True, prog_bar=prog_bar, logger=False, sync_dist=True)

        return {"loss": loss}

    def update_metrics(
        self,
        stage: str,
        loss: torch.Tensor,
        pred_ew: torch.Tensor,
        true_ew: torch.Tensor,
        pred_prob: torch.Tensor,
        true_prob: torch.Tensor,
        pred_sigma: torch.Tensor
    ):
        stage = self.convert_metric_name(stage)
        mask = true_prob.bool()
        for key, metric in self.metrics[stage].items():
            if 'loss' in key:
                metric.update(loss)
            elif 'f1' in key or 'accuracy' in key:
                metric.update(pred_prob.flatten(), true_prob.flatten())
            elif 'cov' in key:
                sigma_multiplier = 1.0 if '1s' in key else 2.0
                lower = pred_ew - sigma_multiplier * pred_sigma
                upper = pred_ew + sigma_multiplier * pred_sigma
                in_range_mask = ((true_ew >= lower) & (true_ew <= upper)).float()
                true_mask = torch.ones_like(in_range_mask)
                metric.update(in_range_mask[mask].flatten(), true_mask[mask].flatten())
            else:
                metric.update(pred_ew[mask].flatten(), true_ew[mask].flatten())

    def compute_metrics(self, stage):
        stage = self.convert_metric_name(stage)
        log_metrics = {}
        for key, metric in self.metrics[stage].items():
            log_metrics[f'{stage}/{key}'] = metric.compute()
            metric.reset()
        if stage in ('train', 'val') and self.logger is not None:
            self.logger.log_metrics(log_metrics, self.current_epoch)

        if stage in ('val', 'test'):
            self.log(
                'hp_metric', log_metrics[f'{stage}/mae'],
                prog_bar=True, on_epoch=True, on_step=False, sync_dist=True
            )

        if stage == 'test':
            for key, metric in log_metrics.items():
                self.log(key, metric, sync_dist=True)

        return log_metrics

    def plot_sparam_curve(self, stage, log_metrics, outputs, dataloader_idx):
        tag = self.convert_metric_name(stage)
        fig = utils.plot_ew_curve(outputs, log_metrics, self.hparams.ew_threshold)
        if self.logger:
            tag = "_".join(['sparam', str(dataloader_idx)])
            self.logger.experiment.add_image(f'{stage}/{tag}', utils.plot_to_image(fig), self.current_epoch)