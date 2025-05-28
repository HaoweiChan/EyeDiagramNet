import json
import torch
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from pathlib import Path
from typing import Union, Dict, List, Tuple
from lightning.pytorch.utilities import CombinedLoader

from common.utils import log_info, read_snp, read_boundary
from .dataset import InferenceTraceEWDataset, get_loader_from_dataset

class CSVProcessor:
    def __init__(self, patterns: List[str] = None, padding_value: int = -1):
        self._patterns = patterns or [
            "**/input for AI*.csv",
            "**/AI input data*.csv",
            "**/AI input.csv",
            "**/saj.csv",
        ]
        self._padding = padding_value

    def locate(self, data_dirs: Union[Dict[str, str], List[str]]) -> Union[Dict[str, Path], List[Path]]:
        is_dict = isinstance(data_dirs, dict)
        out = {} if is_dict else []
        items = data_dirs.items() if is_dict else enumerate(data_dirs)

        for key, dir_path in items:
            log_info(f"Parsing data from {dir_path}")
            matches = [
                p
                for pat in self._patterns
                for p in Path(dir_path).glob(pat)
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one match in {dir_path}, got {matches}"
                )
            if is_dict:
                out[key] = matches[0]
            else:
                out.append(matches[0])
        return out

    def parse(self, csv_path: Union[Path, str]) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_path, index_col=0, header=0)
        types = ("S", "G", "D")
        type_cols = df.columns[df.columns.str.contains("Type_")]
        df[type_cols] = df[type_cols].replace(
            {t: i for i, t in enumerate(types)}
        )

        mats = [self._spatial_feats(row) for _, row in df.iterrows()]

        max_len = max(m.shape[0] for m in mats)
        padded = np.stack([
                np.pad(
                    m,
                    ((0, max_len - m.shape[0]), (0, 0)),
                    constant_values=self._padding
                )
            for m in mats
        ])
        return df.index.values, padded

    def _spatial_feats(self, case: pd.Series) -> np.ndarray:
        idx = case.index
        layer_mask = idx.str.contains("Layer_")
        width_mask = idx.str.contains("W")
        height_mask = idx.str.contains("H_")

        layers = case[layer_mask].astype(int).values
        widths = case[width_mask].values
        heights = case[height_mask].values

        # per-trace fields
        layer_change = np.r_[True, np.diff(layers) != 0]
        _, layer_count = np.unique(layers, return_counts=True)

        # x coordinate: cumulative widths (shifted by 1)
        cum_x = np.r_[0, widths.cumsum()[:-1]]
        x_dim = cum_x - np.repeat(cum_x[layer_change], layer_count)

        # z coordinate: bottom height of each layer
        cum_h = heights[layer_change].cumsum()
        cum_h = np.roll(cum_h, 1)
        cum_h[0] = 0
        z_dim = np.repeat(cum_h, layer_count)

        # original feature block
        layer_idx = np.flatnonzero(layer_mask)
        feat_dim = layer_idx[l] - layer_idx[0]
        data_col = case.values.reshape(-1, feat_dim)
        
        return np.hstack([data_col, x_dim[:, None], z_dim[:, None]])

class TraceSeqEWDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dirs: Dict[str, str],
        label_dir: str,
        batch_size: int,
        test_size: float = 0.2,
        scaler_path: str = None
    ):
        super().__init__()
        
        self._label_dir = label_dir
        self._batch_size = batch_size
        self._test_size = test_size
        self._scaler_path = scaler_path

        self._input_csv_dir = locate_input_csv_files(data_dirs)

        self._train_dataset = {}
        self._val_dataset = {}

    def setup(self, stage=None, padding_value=1):
        fit_scaler = True
        try:
            self._seq_scaler, self._fix_scaler = torch.load(self._scaler_path)
            log_info(f'Loaded scaler object from {self._scaler_path}')
            fit_scaler = False
        except (FileNotFoundError, AttributeError):
            self._seq_scaler = MinMaxScaler(ignore_value=padding_value)
            self._fix_scaler = MinMaxScaler(ignore_value=padding_value)
            log_info(f'Could not find scaler file in {self._scaler_path}, initiating new scaler object.')

        for name, csv_dir in self._input_csv_dir.items():
            case_id, input_arr = parse_input_csv(csv_dir, padding_value)

            # Load the label file with data rate and boundary inputs and EW outputs
            labels = {}
            for pkl_file in Path(self._label_dir, name).glob("*.pkl"):
                with open(pkl_file, 'rb') as f:
                    loaded = pickle.load(f)
                    snp_file = Path(loaded['snp horiz']).stem.replace('_', ' ')
                    snp_vert = tuple(zip(loaded['snp txs'], loaded['snp rxs']))
                    labels[snp_file] = (loaded['directions'], loaded['line ews'], snp_vert)

            # Remove indices not exist in the labels
            labels = {int(key.split('_')[1].split('.')[0]): val for key, val in labels.items()}
            res_set = set(labels.keys())
            act_set = set(case_id)
            missing_indices = act_set - res_set
            arr = np.delete(input_arr, list(missing_indices), axis=0)

            # Obtain min length from all dataset such that they can be combined as an array
            sorted_keys = sorted(labels.keys())
            sorted_vals = [labels[key] for key in sorted_keys]
            min_len = min(len(arr[0]) for arr in sorted_vals)

            # Take intersected amount from all inputs
            boundary_inputs, direction_inputs, snp_vert = map(
                lambda x: np.array([arr[:min_len] for arr in x]), zip(*sorted_vals))
            eye_widths = [eye_widths * 0]  # Make eye_widths to 0

            log_info(f'Name: {name}, '
                     f'Input array: {input_arr.shape}, '
                     f'Boundary inputs: {boundary_inputs.shape}, '
                     f'Directions inputs: {direction_inputs.shape}, '
                     f'Eye widths label: {eye_widths.shape}')

            # Split train and validation set
            train_size = int((1 - self._test_size) * len(input_arr))
            test_size = len(input_arr) - train_size
            train_idx, val_idx = train_test_split(range(len(input_arr)), test_size=test_size, shuffle=True)

            split_func = lambda x: (x[train_idx], x[val_idx])
            x_seq_train, x_seq_val = split_func(input_arr)
            x_tok_train, x_tok_val = split_func(direction_inputs)
            x_fix_train, x_fix_val = split_func(boundary_inputs)
            x_vert_train, x_vert_val = split_func(snp_vert)
            y_train, y_val = split_func(eye_widths)

            # Scale input but ignore layer, type, x_dim, z_dim features
            if fit_scaler:
                self._seq_scaler.partial_fit(x_seq_train[:, :, 2:-2].reshape(-1, input_arr.shape[-1] - 4))
                self._fix_scaler.partial_fit(x_fix_train.reshape(-1, x_fix_train.shape[-1]))

            # store dataset
            self._train_dataset[name] = TraceEWDataset(
                x_seq_train, x_tok_train, x_fix_train, x_vert_train, y_train, train=True
            )
            self._val_dataset[name] = TraceEWDataset(
                x_seq_val, x_tok_val, x_fix_val, x_vert_val, y_val
            )

        log_info(f'\n------------ Sequence Scaler ------------\n'
                 f'Samples seen: {self._seq_scaler.n_samples_seen_}\n'
                 f'Minimum: {self._seq_scaler.min_.round(decimals=2).tolist()}\n'
                 f'Maximum: {self._seq_scaler.max_.round(decimals=2).tolist()}\n')

        log_info(f'------------- Fix Scaler ---------------\n'
                 f'Samples seen: {self._fix_scaler.n_samples_seen_}\n'
                 f'Minimum: {self._fix_scaler.min_.round(decimals=2).tolist()}\n'
                 f'Maximum: {self._fix_scaler.max_.round(decimals=2).tolist()}\n')

        for name in list(self._train_dataset.keys()):
            self._train_dataset[name] = self._train_dataset[name].transform(self._seq_scaler, self._fix_scaler)
            self._val_dataset[name] = self._val_dataset[name].transform(self._seq_scaler, self._fix_scaler)

        torch.save((self._seq_scaler, self._fix_scaler), Path(self.trainer.logger.log_dir).joinpath("scaler.pth"))

    def train_dataloader(self):
        return CombinedLoader({
            name: get_loader_from_dataset(
                dataset=dataset,
                batch_size=self._batch_size // len(self._train_dataset),
                shuffle=True,
            )
            for name, dataset in self._train_dataset.items()
        }, 'min_size')

    def val_dataloader(self):
        return CombinedLoader({
            name: get_loader_from_dataset(
                dataset=dataset,
                batch_size=int(self._batch_size * 1.6 // len(self._val_dataset)),
            )
            for name, dataset in self._val_dataset.items()
        }, 'min_size')

class InferenceTraceSeqEWDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dirs: List[str],
        tx_snp: str,
        rx_snp: str,
        batch_size: int,
        bound_path: str = None,
        scaler_path: str = None
    ):
        super().__init__()

        self._data_dirs = data_dirs
        self._tx_snp = tx_snp
        self._rx_snp = rx_snp
        self._batch_size = batch_size
        self._bound_path = bound_path
        self._scaler_path = scaler_path

    def setup(self, stage=None):
        # Initialize processor and locate CSV files
        processor = CSVProcessor()
        csv_paths = processor.locate(self._data_dirs)

        # Load scaler
        scalers = torch.load(self._scaler_path)
        log_info(f'Loaded scaler object from {self._scaler_path}')

        tx = read_snp(Path(self._tx_snp))
        rx = read_snp(Path(self._rx_snp))
        assert tx.shape[-1] == rx.shape[-1], \
            f"TX {self._tx_snp} and RX {self._rx_snp} must match ports."

        # Load boundary JSON
        self._directions, self._boundary = read_boundary(self._bound_path)

        self._predict_dataset = []
        for csv_path in csv_paths:
            case_id, input_arr = processor.parse(csv_path)
            log_info(f'Input array: {input_arr.shape}')

            ds = InferenceTraceEWDataset(input_arr, self._directions, self._boundary.to_array(), tx, rx)
            self._predict_dataset.append(ds.transform(*scalers))
            
    def predict_dataloader(self):
        return CombinedLoader({
            i: get_loader_from_dataset(dataset=ds, batch_size=self._batch_size)
            for i, ds in enumerate(self._predict_dataset)
        }, 'sequential')