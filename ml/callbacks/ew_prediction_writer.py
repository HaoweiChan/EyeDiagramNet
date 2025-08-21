import pprint
from pathlib import Path

import torch
import torch.distributed as dist
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter


class EWPredictionWriter(BasePredictionWriter):
    """Writes EW predictions and metadata once per epoch."""

    def __init__(self, file_prefix: str, write_interval: str = "epoch"):
        super().__init__(write_interval=write_interval)
        self.file_prefix = file_prefix
        self.file_dir: Path | None = None

    def set_output_paths(self, file_dir: Path):
        self.file_dir = file_dir

    def write_on_batch_end(self, trainer, pl_module, predictions, batch_indices, batch, batch_idx, dataloader_idx):
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        assert self.file_dir is not None, "file_dir not set — did you call set_output_paths?"

        # ensure output dir exists
        self.file_dir.mkdir(parents=True, exist_ok=True)
        txt_path = self.file_dir / f"{self.file_prefix}.txt"
        csv_path = self.file_dir / f"{self.file_prefix}.csv"

        # build DataFrame of EW values
        df = self._build_dataframe(predictions, batch_indices)

        # 1) dump metadata + DataFrame snapshot to .txt
        with txt_path.open("w") as f:
            f.write("boundary:\n")
            pprint.pprint(trainer.datamodule.boundary, stream=f)
            f.write(f"tx_snp: {trainer.datamodule.tx_snp}\n")
            f.write(f"rx_snp: {trainer.datamodule.rx_snp}\n")
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

        # sorted pairs: (idx, ew, prob, mask)
        sorted_pairs = sorted(zip(flat_idxs, gathered_preds), key=lambda x: x[0])

        # extract just the EW tensors
        ew_tensors = [ew for _, (ew, *_rest) in sorted_pairs]

        # concatenate and wrap in DataFrame
        ew_cat = torch.cat(ew_tensors, dim=0)
        return pd.DataFrame(ew_cat.float())

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
