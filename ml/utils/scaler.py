import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseScaler(ABC):
    """
    Base class for all scalers.
    
    All scalers should implement these methods to provide consistent
    scaling/unscaling functionality for PyTorch tensors.
    """
    def __init__(self):
        self.n_samples_seen_ = 0

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def fit(self, values):
        pass

    @abstractmethod
    def partial_fit(self, values):
        pass

    @abstractmethod
    def transform(self, values):
        pass

    @abstractmethod
    def fit_transform(self, values):
        pass

    @abstractmethod
    def inverse_transform(self, values):
        pass

class StandardScaler(BaseScaler):
    def __init__(self, mean=None, std=None, epsilon=1e-7, nan=0.0):
        """
        Standard Scaler that normalizes tensors using mean and standard deviation.
        Args:
            mean: The mean of the features (set after calling fit)
            std: The standard deviation of the features (set after calling fit)
            epsilon: Small constant to avoid divide-by-zero errors
            nan: Value to use for replacing NaN values after transformation.
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.nan = nan

    def _reset(self):
        self.n_samples_seen_ = 0
        self.mean = None
        self.std = None

    def fit(self, values):
        self._reset()
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)
        
        # Manually calculate mean and std ignoring NaNs
        non_nan_values = values[~torch.isnan(values)]
        dims = list(range(values.dim() - 1))
        
        self.mean = torch.mean(non_nan_values, dim=dims)
        self.std = torch.std(non_nan_values, dim=dims)
        
        self.n_samples_seen_ = values.shape[0]
        return self

    def transform(self, values):
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)
        values_transformed = (values - self.mean) / (self.std + self.epsilon)
        return torch.nan_to_num(values_transformed, nan=self.nan)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        # Ensure scaler parameters match the input tensor's dtype and device
        std_tensor = self.std.to(device=values.device, dtype=values.dtype)
        mean_tensor = self.mean.to(device=values.device, dtype=values.dtype)
        return values * std_tensor + mean_tensor

class MinMaxScaler(BaseScaler):
    def __init__(self, min_=None, max_=None, nan=0.0):
        """
        MinMax Scaler that scales tensors to a fixed range.
        Args:
            min_: Minimum value for scaling (set after calling fit)
            max_: Maximum value for scaling (set after calling fit)
            nan: Value to use for replacing NaN values after transformation.
        """
        super().__init__()
        self.min_ = min_
        self.max_ = max_
        self.nan = nan

    def _reset(self):
        self.n_samples_seen_ = 0
        if hasattr(self, "scale_"):
            del self.scale_
        if hasattr(self, "min_"):
            del self.min_
        if hasattr(self, "max_"):
            del self.max_

    def fit(self, values):
        self._reset()
        return self.partial_fit(values)

    def partial_fit(self, values):
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)

        # Create a mask for non-NaN values
        not_nan_mask = ~torch.isnan(values)

        # Initialize with extreme values
        data_min = torch.full_like(values[0], float('inf'))
        data_max = torch.full_like(values[0], float('-inf'))

        # Iterate over each feature to find the min and max
        for i in range(values.shape[1]):
            valid_values = values[:, i][not_nan_mask[:, i]]
            if valid_values.numel() > 0:
                data_min[i] = torch.min(valid_values)
                data_max[i] = torch.max(valid_values)

        if self.n_samples_seen_ == 0:
            self.min_ = data_min
            self.max_ = data_max
        else:
            self.min_ = torch.minimum(self.min_, data_min)
            self.max_ = torch.maximum(self.max_, data_max)

        self.n_samples_seen_ += len(values)
        self.scale_ = self.max_ - self.min_
        return self

    def transform(self, values):
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)

        valid_scale = torch.where(self.scale_ == 0.0,
                                torch.ones_like(self.scale_),
                                self.scale_)

        values_transformed = (values - self.min_) / valid_scale
        return torch.nan_to_num(values_transformed, nan=self.nan)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        # Ensure scaler parameters match the input tensor's dtype and device
        max_tensor = self.max_.to(device=values.device, dtype=values.dtype)
        min_tensor = self.min_.to(device=values.device, dtype=values.dtype)
        return values * (max_tensor - min_tensor) + min_tensor

class MaxAbsScaler(BaseScaler):
    def __init__(self, max_=None, nan=0.0):
        """
        MaxAbs Scaler that scales tensors by their maximum absolute value.
        Args:
            max_: Maximum absolute value (set after calling fit)
            nan: Value to use for replacing NaN values after transformation.
        """
        super().__init__()
        self.max_ = max_
        self.nan = nan

    def _reset(self):
        self.n_samples_seen_ = 0
        self.max_ = None

    def fit(self, values):
        self._reset()
        return self.partial_fit(values)

    def partial_fit(self, values):
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)

        # Manually calculate max of absolute values ignoring NaNs
        non_nan_abs_values = torch.abs(values[~torch.isnan(values)])
        current_max = torch.max(non_nan_abs_values) if non_nan_abs_values.numel() > 0 else torch.tensor(float('-inf'))

        if self.n_samples_seen_ == 0:
            self.max_ = current_max
        else:
            self.max_ = torch.maximum(self.max_, current_max)

        self.n_samples_seen_ += len(values)
        return self

    def transform(self, values):
        if isinstance(values, (np.ndarray, np.generic)):
            values = torch.from_numpy(values)
        
        values_transformed = values / (self.max_ + 1e-7)
        return torch.nan_to_num(values_transformed, nan=self.nan)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return values * (self.max_ + 1e-7)