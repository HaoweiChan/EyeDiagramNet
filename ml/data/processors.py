import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Tuple

from common.utils import log_info

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
        width_mask = idx.str.contains("W_")
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
        feat_dim = layer_idx[1] - layer_idx[0]
        data_col = case.values.reshape(-1, feat_dim)
        
        return np.hstack([data_col, x_dim[:, None], z_dim[:, None]]) 