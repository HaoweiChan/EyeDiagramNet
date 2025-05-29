import json
import torch
import pickle
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from pathlib import Path
from typing import Union, Dict, List, Tuple
from sklearn.model_selection import train_test_split
from lightning.pytorch.utilities import CombinedLoader

from .dataset import TraceDataset, TraceEWDataset, InferenceTraceDataset, InferenceTraceEWDataset, get_loader_from_dataset
from .bound_param import SampleResult
from common.scaler import MinMaxScaler
from common.utils import log_info, parse_snps, read_snp

class CSVProcessor:
    def __init__(self, patterns: List[str] = None, padding_value: int = -1):
        self.patterns = patterns or [
            "input_for_AI*.csv",
            "*AI_input_data*.csv",
            "*AI_input*.csv",
            "*ai*.csv",
        ]
        self.padding = padding_value

    def locate(self, data_dirs: Union[Dict[str, str], List[str]]) -> Union[Dict[str, Path], List[Path]]:
        is_dict = isinstance(data_dirs, dict)
        out = {} if is_dict else []
        items = data_dirs.items() if is_dict else enumerate(data_dirs)

        for key, dir_path in items:
            log_info(f"Parsing data from {dir_path}")
            matches = [
                p
                for pat in self.patterns
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
        df[type_cols] = df[type_cols].replace({t: i for i, t in enumerate(types)})

        mats = [self._spatial_feats(row.dropna()) for _, row in df.iterrows()]
        max_len = max(m.shape[0] for m in mats)
        padded = np.stack(
            [
                np.pad(
                    m,
                    ((0, max_len - m.shape[0]), (0, 0)),
                    constant_values=self.padding
                )
                for m in mats
            ]
        )
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

class TraceSeqEWDataloader(pl.LightningDataModule):
    def __init__(
        self,
        data_dirs: Dict[str, str],
        label_dir: str,
        batch_size: int,
        test_size: float = 0.2,
        scaler_path: str | None = None,
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.label_dir = Path(label_dir)
        self.batch_size = batch_size
        self.test_size = test_size
        self.scaler_path = scaler_path

        # containers for datasets per "name"
        self.train_dataset: Dict[str, TraceEWDataset] = {}
        self.val_dataset: Dict[str, TraceEWDataset] = {}

    def setup(self, stage: str | None = None, padding_value: int = -1):
        # Scalers
        fit_scaler = True
        try:
            self.seq_scaler, self.fix_scaler = torch.load(self.scaler_path)
            log_info(f"Loaded scalers from {self.scaler_path}")
            fit_scaler = False
        except (FileNotFoundError, AttributeError):
            self.seq_scaler = MinMaxScaler(ignore_value=padding_value)
            self.fix_scaler = MinMaxScaler(ignore_value=padding_value)
            log_info("Could not find scalers on disk, creating new ones.")

        # locate every CSV once via processor
        processor = CSVProcessor()
        csv_paths = processor.locate(self.data_dirs) # dict[str, Path]

        # iterate over each named dataset
        for name, csv_path in csv_paths.items():
            case_ids, input_arr = processor.parse(csv_path)

            # Load labels
            labels: dict[str, tuple] = {}
            for pkl_file in Path(self.label_dir, name).glob("*.pkl"):
                with open(pkl_file, "rb") as f:
                    loaded = pickle.load(f)
                snp_file = Path(loaded["snp_horiz"]).stem.replace("-", "_")
                snp_vert = tuple(zip(loaded["snp_txs"], loaded["snp_rxs"]))
                key = (snp_file.split("_")[-1].split(".")[0])
                labels[key] = {
                    loaded["configs"],
                    loaded["directions"],
                    loaded["line_ews"],
                    snp_vert,
                }

            # keep only indices present in labels
            label_keys = set(labels.keys())
            keep_idx = [i for i, cid in enumerate(case_ids) if cid in label_keys]
            input_arr = input_arr[keep_idx]
            sorted_keys = [case_ids[i] for i in keep_idx]
            sorted_vals = [labels[k] for k in sorted_keys]

            # all tensors must share same length along trace dim
            min_len = min(len(v[0]) for v in sorted_vals)
            boundary_inputs, direction_inputs, eye_widths, snp_vert = map(
                lambda arrs: np.array([a[:min_len] for a in arrs]),
                zip(*sorted_vals),
            )
            eye_widths = np.asarray(eye_widths)
            eye_widths[eye_widths < 0] = 0 # Make -0.1 eye_widths to 0

            log_info(f"{name}| input_seq {input_arr.shape} | eye_width {eye_widths.shape}")

            # train/val split
            indices = np.arange(len(input_arr))
            train_idx, val_idx = train_test_split(indices, test_size=self.test_size, shuffle=True)

            def _split(arr):
                return arr[train_idx], arr[val_idx]

            x_seq_tr, x_seq_val = _split(input_arr)
            x_tok_tr, x_tok_val = _split(direction_inputs)
            x_fix_tr, x_fix_val = _split(boundary_inputs)
            x_vert_tr, x_vert_val = _split(snp_vert)
            y_tr, y_val = _split(eye_widths)

            # fit scalers once on training data
            if fit_scaler:
                seq_feats = x_seq_tr[:, :, 2:-2].reshape(-1, input_arr.shape[-1] - 4)
                self.seq_scaler.partial_fit(seq_feats)
                self.fix_scaler.partial_fit(x_fix_tr.reshape(-1, x_fix_tr.shape[-1]))

            # build datasets
            self.train_dataset[name] = TraceEWDataset(
                x_seq_tr, x_tok_tr, x_fix_tr, x_vert_tr, y_tr, train=True
            )
            self.val_dataset[name] = TraceEWDataset(
                x_seq_val, x_tok_val, x_fix_val, x_vert_val, y_val
            )

        # final transform with fitted scalers
        for name in list(self.train_dataset.keys()):
            self.train_dataset[name] = self.train_dataset[name].transform(
                self.seq_scaler, self.fix_scaler
            )
            self.val_dataset[name] = self.val_dataset[name].transform(
                self.seq_scaler, self.fix_scaler
            )

        # persist scalers for future runs
        if self.trainer and self.trainer.logger:
            save_path = Path(self.trainer.logger.log_dir) / "scaler.pth"
            torch.save((self.seq_scaler, self.fix_scaler), save_path)

    def train_dataloader(self):
        per_loader_bs = self.batch_size // max(1, len(self.train_dataset))
        loaders = {
            name: get_loader_from_dataset(ds, batch_size=per_loader_bs, shuffle=True)
            for name, ds in self.train_dataset.items()
        }
        return CombinedLoader(loaders, mode="min_size")

    def val_dataloader(self):
        per_loader_bs = int(self.batch_size * 1.6 / max(1, len(self.val_dataset)))
        loaders = {
            name: get_loader_from_dataset(ds, batch_size=per_loader_bs, shuffle=False)
            for name, ds in self.val_dataset.items()
        }
        return CombinedLoader(loaders, mode="min_size")

class InferenceTraceSeqEWDataloader(pl.LightningDataModule):
    def __init__(
        self,
        data_dirs: List[str],
        tx_snp: str,
        rx_snp: str,
        batch_size: int,
        bound_path: str = None,
        scaler_path: str = None,
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.tx_snp = tx_snp
        self.rx_snp = rx_snp
        self.batch_size = batch_size
        self.bound_path = bound_path
        self.scaler_path = scaler_path

    def setup(self, stage=None):
        # Initialize processor and locate CSV files
        processor = CSVProcessor()
        csv_paths = processor.locate(self.data_dirs)

        # Load scaler
        scalers = torch.load(self.scaler_path)
        log_info(f"Loaded scaler object from {self.scaler_path}")

        tx = read_snp(Path(self.tx_snp))
        rx = read_snp(Path(self.rx_snp))
        assert tx.shape[-1] == rx.shape[-1], \
            f"TX {self.tx_snp} and RX {self.rx_snp} must match ports."

        # Load boundary JSON
        with open(self.bound_path, 'r') as f:
            loaded = json.load(f)
            directions = np.array(loaded['directions']) if 'directions' in loaded else np.ones(tx.shape[-1] // 2, dtype=int)
            ctle = loaded.get('CTLE', {"AC_gain": np.nan, "DC_gain": np.nan, "fp1": np.nan, "fp2": np.nan})
            boundary = loaded['boundary'] | ctle
            self.boundary = SampleResult(**boundary)

        self.predict_dataset = []
        for csv_path in csv_paths:
            case_id, input_arr = processor.parse(csv_path)
            log_info(f"Input array: {input_arr.shape}")
            ds = InferenceTraceEWDataset(input_arr, directions, self.boundary.to_array(), tx, rx)
            self.predict_dataset.append(ds.transform(*scalers))

    def predict_dataloader(self):
        return CombinedLoader(
            get_loader_from_dataset(dataset=ds, batch_size=self.batch_size)
            for ds in self.predict_dataset
        )

class TraceSeqDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Dict[str, str],
        batch_size: int,
        test_size: float = 0.2,
        max_ports: int = 8,
        scaler_path: str = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.test_size = test_size
        self.locate_input_csv_files(data_dir)
        self.max_ports = max_ports
        self.scaler_path = scaler_path
        self.train_dataset = {}
        self.val_dataset = {}

    def locate_input_csv_files(self, data_dir):
        self.input_csv_dir = {}
        for name, csv_dir in data_dir.items():
            log_info(f"Parsing data from {csv_dir}")
            patterns = ["input for AI*.csv", "*AI input data*.csv"]
            input_csv_dir = [
                d for pattern in patterns for d in Path(csv_dir).glob(pattern)
            ]
            assert len(input_csv_dir) == 1, f"Cannot find input file in {csv_dir}"
            self.input_csv_dir[name] = input_csv_dir[0]

    def replace_zero_length_with_max(self, row):
        max_val = row[row != 0].max()
        row[row == 0] = max_val
        return row

    def parse_input_csv(self, csv_dir):
        # Cast line types to integers
        unique_types = ('S', 'G', 'D')
        df = pd.read_csv(input_csv_dir, index_col=0, header=0)
        df = df.replace({col: {v: i for i, v in enumerate(unique_types)} for col in df.columns[df.columns.str.contains('Type_')]})

        def replace_zero_with_max(row):
            max_val = row[row != 0].max()
            row[row == 0] = max_val
            return row

        def load_and_preprocess_data(input_csv_dir):
            df = pd.read_csv(input_csv_dir, index_col=0, header=0)
            df = df.replace({col: {v: i for i, v in enumerate(unique_types)} for col in df.columns[df.columns.str.contains('Type_')]})
            return df

        def extract_features(df):
            layer_idx = df.columns.str.contains('Layer_').nonzero()[0]
            feat_dim = layer_idx[1] - layer_idx[0] - 3

            width_idx = df.columns.str.contains('W_').nonzero()[0]
            widths = df.iloc[:, width_idx]

            height_idx = df.columns.str.contains('H_').nonzero()[0]
            heights = df.iloc[:, height_idx]

            length_idx = df.columns.str.contains('L_').nonzero()[0]
            lengths = df.iloc[:, length_idx].apply(replace_zero_with_max, axis=1)

            df.drop(df.columns[np.concatenate((width_idx, height_idx, length_idx))], axis=1, inplace=True)

            return df, widths, heights, lengths, feat_dim

        def calculate_dimensions(df, widths, heights, lengths):
            layer_idx = df.columns.str.contains('Layer_').nonzero()[0]
            layer_change = df.iloc[:, layer_idx].diff(axis=1).fillna(1).values.astype(bool)
            layer_count = np.unique(df.iloc[:, layer_idx], return_counts=True)[1]
            layer_count = layer_count / len(df).astype(int)

            signal_layer_start = layer_count.cumsum()[np.where(layer_count != 1)[0] - 1][0]
            signal_layer_end = layer_count.cumsum()[np.where(layer_count != 1)[0]][0]
            signal_total_widths = widths.values.copy()
            new_widths = signal_total_widths[..., None]
            new_widths[:, signal_layer_end:] = signal_total_widths[..., None]
            new_widths[:, signal_layer_end:] = signal_total_widths[..., None]
            new_widths[:, signal_layer_end:] = signal_total_widths[..., None]
            cum_widths = widths.cumsum(1, axis=1, fill_value=0).values
            x_dim = cum_widths = np.repeat(cum_widths[layer_change].reshape((len(widths), -1)), layer_count, axis=1)

            cum_heights = heights.values[layer_change].reshape((len(heights), -1)).cumsum(1)
            cum_heights = np.roll(cum_heights, shift=1, axis=1)
            cum_heights[:, 0] = 0
            z_dim = np.repeat(cum_heights, layer_count, axis=1)

            y_dim = np.dstack([np.zeros_like(lengths), lengths]).reshape((len(lengths), -1))

            spatial_feats = np.dstack([new_widths, heights, lengths, x_dim, z_dim])

            return spatial_feats, y_dim

        df = load_and_preprocess_data(csv_dir)
        df, widths, heights, lengths, feat_dim = extract_features(df)
        spatial_feats, y_dim = calculate_dimensions(df, widths, heights, lengths)

        new_input_arr = np.concatenate([df.values.reshape(len(df), -1, feat_dim), spatial_feats], axis=2)
        new_input_arr = np.concatenate([np.repeat(new_input_arr, 2, axis=1), y_dim[..., None]], axis=2)

        return new_input_arr, df.index

    def setup(self, stage=None):
        fit_scaler = True
        try:
            self.seq_scaler = torch.load(self.scaler_path)
            log_info(f'Loaded scaler object from {self.scaler_path}')
            fit_scaler = False
        except (FileNotFoundError, AttributeError):
            self.seq_scaler = MinMaxScaler()
            log_info('Could not find scaler file in {self.scaler_path}, initiating new scaler object.')

        for name, csv_dir in self.input_csv_dir.items():
            input_arr, sample_idx = self.parse_input_csv(csv_dir)

            # Read output snps
            snp_dir = Path(csv_dir).parent
            if Path(snp_dir, 'npz').exists():
                snp_dir = snp_dir.joinpath("npz")
            elif Path(snp_dir, 'snp').exists():
                snp_dir = snp_dir.joinpath("snp")
            snps, missing_idx = parse_snps(snp_dir, sample_idx.astype(np.float32))
            for i in missing_idx:
                input_arr = np.delete(input_arr, i, axis=0)

            # Get number of ports
            one_snp = read_snp(snps[0])

            # Random choice N elements from input and snp
            # seed = sum(ord(char) for char in str(csv_dir))
            # np.random.seed(seed)
            # indices = np.random.choice(input_arr.shape[0], 10, replace=False)
            indices = np.arange(125)
            # indices = np.arange(1000)
            snps = [snps[i] for i in indices]
            input_arr = input_arr[indices]
            log_info(f'{name}| input_arr: {input_arr.shape}, SNP: ({len(snps)}*{one_snp.shape})')

            # Split train and validation set
            x_train, x_val, y_train, y_val = train_test_split(input_arr, snps, test_size=self.test_size)
            x_train, x_val = train_test_split(input_arr, snps, test_size=self.test_size, shuffle=False)
            # scale input but ignore layer and type features
            if fit_scaler:
                self.seq_scaler.partial_fit(x_train[:, :, 2:].reshape(-1, input_arr.shape[-1] - 2))

            # store dataset
            self.train_dataset[name] = TraceDataset(x_train, y_train, self.max_ports)
            self.val_dataset[name] = TraceDataset(x_val, y_val, self.max_ports)

        log_info(f'-----------Scaler-----------\n\
samples seen: {self.seq_scaler.n_samples_seen_}\n\
minimum: {self.seq_scaler.min_.round(decimals=2).tolist()}\n\
maximum: {self.seq_scaler.max_.round(decimals=2).tolist()}')
        for key in list(self.train_dataset.keys()):
            self.train_dataset[name] = self.train_dataset[name].transform(self.seq_scaler, add_noise=True)
            self.val_dataset[name] = self.val_dataset[name].transform(self.seq_scaler)

        torch.save(self.seq_scaler, Path(self.trainer.logger.log_dir).joinpath("scaler.pth"))

    def train_dataloader(self):
        return CombinedLoader(
            {
                name: get_loader_from_dataset(
                    dataset=dataset,
                    batch_size=self.batch_size // len(self.train_dataset),
                    shuffle=True,
                )
                for name, dataset in self.train_dataset.items()
            },
            'min_size'
        )

    def val_dataloader(self):
        return CombinedLoader(
            {
                name: get_loader_from_dataset(
                    dataset=dataset,
                    batch_size=int(self.batch_size * 1.6 // len(self.val_dataset)),
                )
                for name, dataset in self.val_dataset.items()
            },
            'min_size'
        )

class InferenceTraceSeqDataLoader(TraceSeqDataLoader):
    def __init__(
        self,
        data_dir: Dict[str, str],
        batch_size: int,
        max_ports: int,
        scaler_path: str,
    ):
        super().__init__(data_dir, batch_size, max_ports=max_ports, scaler_path=scaler_path)
        self.predict_dataset = {}

    def setup(self, stage=None):
        self.seq_scaler = torch.load(self.scaler_path)
        log_info(f'Loaded scaler object from {self.scaler_path}')

        for name, csv_dir in self.input_csv_dir.items():
            input_arr, df = self.parse_input_csv(csv_dir)
            # seed = sum(ord(char) for char in str(csv_dir))
            # np.random.seed(seed)
            # indices = np.random.choice(input_arr.shape[0], 100, replace=False)
            input_arr = input_arr[:100]
            log_info(f'{name}| input_arr: {input_arr.shape}')

            # store dataset
            self.predict_dataset[name] = InferenceTraceDataset(input_arr, self.max_ports)
        log_info(f'-----------Scaler-----------\n\
samples seen: {self.seq_scaler.n_samples_seen_}\n\
minimum: {self.seq_scaler.min_.round(decimals=2).tolist()}\n\
maximum: {self.seq_scaler.max_.round(decimals=2).tolist()}')
        for key in list(self.predict_dataset.keys()):
            self.predict_dataset[name] = self.predict_dataset[name].transform(self.seq_scaler)

    def predict_dataloader(self):
        return CombinedLoader(
            {
                name: get_loader_from_dataset(
                    dataset=dataset,
                    batch_size=self.batch_size * 2 // len(self.predict_dataset),
                    shuffle=False,
                )
                for name, dataset in self.predict_dataset.items()
            },
            'sequential'
        )