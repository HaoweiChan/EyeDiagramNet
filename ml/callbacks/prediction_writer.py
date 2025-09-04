import pprint
from pathlib import Path

import torch
import torch.distributed as dist
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter


class EWPredictionWriter(BasePredictionWriter):
    """Writes EW predictions and metadata once per epoch."""

    def __init__(self, file_prefix: str, write_interval: str = "epoch", output_dir: str = None):
        super().__init__(write_interval=write_interval)
        self.file_prefix = file_prefix
        self.file_dir = Path(output_dir) if output_dir else None

    def set_output_dir(self, output_dir: str):
        """Set the output directory for predictions (typically from logger directory)."""
        self.file_dir = Path(output_dir)

    def write_on_batch_end(self, trainer, pl_module, predictions, batch_indices, batch, batch_idx, dataloader_idx):
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Ensure output directory is set (from logger directory)
        if self.file_dir is None:
            if trainer.logger and hasattr(trainer.logger, 'log_dir'):
                self.set_output_dir(trainer.logger.log_dir)
            else:
                raise RuntimeError("Output directory not set and no logger directory available")
        
        # ensure output dir exists
        self.file_dir.mkdir(parents=True, exist_ok=True)
        txt_path = self.file_dir / f"{self.file_prefix}.txt"
        csv_path = self.file_dir / f"{self.file_prefix}.csv"

        # build DataFrame of EW values
        df = self._build_dataframe(predictions, batch_indices)

        # 1) dump metadata + DataFrame snapshot to .txt
        with txt_path.open("w") as f:
            f.write("boundary:\n")
            
            # Handle boundary attribute safely
            if hasattr(trainer.datamodule, 'boundary'):
                pprint.pprint(trainer.datamodule.boundary, stream=f)
            else:
                f.write("Not available\n")
            
            # Handle SNP attributes - different datamodules use different naming
            drv_snp_path = getattr(trainer.datamodule, 'drv_snp', 
                                 getattr(trainer.datamodule, 'tx_snp', 'Not available'))
            odt_snp_path = getattr(trainer.datamodule, 'odt_snp', 
                                 getattr(trainer.datamodule, 'rx_snp', 'Not available'))
            
            f.write(f"drv_snp: {drv_snp_path}\n")
            f.write(f"odt_snp: {odt_snp_path}\n")
            f.write(df.to_string())

        # 2) write full CSV
        df.to_csv(csv_path, index=False)

    def _build_dataframe(self, predictions, batch_indices) -> pd.DataFrame:
        """
        Gather across ranks, flatten & sort by batch index,
        then return a DataFrame of EW values.
        """
        gathered_preds, gathered_idxs = self._safe_gather(predictions, batch_indices)

        # flatten first dataloader's indices
        flat_idxs = [i for sub in gathered_idxs[0] for i in sub]

        # Handle case where gathered_preds contains a single 2D tensor
        # Extract individual prediction tensors (rows) from the 2D tensor
        if len(gathered_preds) == 1 and isinstance(gathered_preds[0], torch.Tensor) and gathered_preds[0].dim() >= 2:
            # gathered_preds contains a single 2D tensor with shape (num_samples, num_features)
            pred_tensor = gathered_preds[0]
            # Extract individual rows (predictions) for each sample
            individual_preds = [pred_tensor[i] for i in range(pred_tensor.shape[0])]
        else:
            # Handle case where predictions might be structured differently
            individual_preds = []
            for pred in gathered_preds:
                if isinstance(pred, (tuple, list)):
                    # Extract EW tensor from tuple (ew, prob, mask, etc.)
                    individual_preds.append(pred[0])  
                else:
                    individual_preds.append(pred)

        # sorted pairs: (idx, prediction)
        sorted_pairs = sorted(zip(flat_idxs, individual_preds), key=lambda x: x[0])

        # extract just the prediction tensors
        pred_tensors = [pred for _, pred in sorted_pairs]

        # concatenate and wrap in DataFrame
        if pred_tensors:
            pred_cat = torch.stack(pred_tensors, dim=0)  # Use stack for 2D result
            return pd.DataFrame(pred_cat.float().cpu().numpy())
        else:
            return pd.DataFrame()

    def _safe_gather(self, predictions, batch_indices):
        """
        All-gather both predictions and batch_indices across ranks.
        On single-GPU or on error, fall back to the raw lists.
        """
        try:
            world_size = dist.get_world_size()

            # prepare containers
            preds_container = [None] * world_size
            idxs_container = [None] * world_size

            # gather
            dist.all_gather_object(preds_container, predictions)
            dist.all_gather_object(idxs_container, batch_indices)

            # use rank-0 slice (its lists already contain all samples)
            gathered_preds = preds_container[0]
            gathered_idxs = idxs_container[0]

        except (ValueError, RuntimeError, AttributeError):
            # fallback for non-distributed / on failure
            gathered_preds = predictions
            gathered_idxs = batch_indices

        return gathered_preds, gathered_idxs
