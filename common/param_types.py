import random
import numpy as np

def _get_decimal_places(value):
    value_str = str(value).rstrip('0')  # Remove trailing zeros
    if '.' in value_str:
        return len(value_str.split('.')[-1])  # Count decimal places after the dot
    return 0  # No decimal places for integers

class Parameter:
    def __init__(self, low=None, high=None, step=None, numbers=None):
        self.low = low
        self.high = high
        self.step = step
        self.numbers = numbers
        
    def sample(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
        
    def is_within_range(self, value):
        return self.low <= value <= self.high

class DiscreteParameter(Parameter):
    def __init__(self, values):
        super().__init__()
        assert isinstance(values, list)
        self.values = values
        
    def sample(self):
        return random.choice(self.values)
        
    def is_within_range(self, value):
        return value in self.values

class LinearParameter(Parameter):
    def __init__(self, low, high, step=None, numbers=None, additional_values=None, scaler=1):
        super().__init__(low, high, step, numbers)
        self.additional_values = additional_values if additional_values is not None else []
        self.scaler = scaler

    def sample(self):
        if self.numbers is not None:
            values = np.linspace(self.low, self.high, num=self.numbers, dtype=np.float64)
        elif self.step is not None:
            values = np.arange(self.low, self.high + self.step, step=self.step, dtype=np.float64)
        else:
            raise ValueError("Either 'step' or 'numbers' must be specified for LinearParameter.")
            
        if self.additional_values:
            additional_array = np.array(self.additional_values, dtype=values.dtype)
            values = np.concatenate([values, additional_array])
        
        if self.scaler != 1:
            values = values * self.scaler
        
        # Round to the appropriate number of decimal places
        decimal_places = max(_get_decimal_places(self.low), _get_decimal_places(self.high))
        if self.step is not None:
            decimal_places = max(decimal_places, _get_decimal_places(self.step))
        if decimal_places > 0:
            values = np.round(values, decimal_places)
        
        return np.random.choice(values)

class LogParameter(Parameter):
    def __init__(self, low, high, step=None, numbers=None, additional_values=None, scaler=1):
        super().__init__(low, high, step, numbers)
        self.additional_values = additional_values if additional_values is not None else []
        self.scaler = scaler

    def sample(self):
        if self.numbers is not None:
            values = np.logspace(np.log10(self.low), np.log10(self.high), num=self.numbers, dtype=np.float64)
        elif self.step is not None:
            log_low = np.log10(self.low)
            log_high = np.log10(self.high)
            log_step = np.log10(self.step)
            values = 10 ** np.arange(log_low, log_high + log_step, step=log_step, dtype=np.float64)
        else:
            raise ValueError("Either 'step' or 'numbers' must be specified for LogParameter.")
            
        if self.additional_values:
            additional_array = np.array(self.additional_values, dtype=values.dtype)
            values = np.concatenate([values, additional_array])
        
        if self.scaler != 1:
            values = values * self.scaler
        
        return np.random.choice(values)

class SampleResult:
    def __init__(self, bound_type=None, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._data = kwargs
        if bound_type is not None:
            self.is_within_range(bound_type)
            
    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)
        
    def __add__(self, other):
        if not isinstance(other, SampleResult):
            raise TypeError("Can only add SampleResult instances")

        new_data = self._data.copy()
        for key, value in other._data.items():
            if key in new_data:
                raise ValueError(f"Conflicting key '{key}' found in both SampleResult instances")
            new_data[key] = value

        return SampleResult(**new_data)
        
    def to_dict(self):
        return self._data.copy()

    def to_list(self, return_keys=False):
        if return_keys:
            keys = list(self._data.keys())
            values = list(self._data.values())
            return keys, values
        return list(self._data.values())

    def is_within_range(self, bound_type=None):
        if bound_type is None:
            return
        
        bound_ranges = CONSTRAINT_RANGES.get(bound_type, {})
        
        for key, value in self._data.items():
            if key in bound_ranges:
                if not bound_ranges[key].is_within_range(value):
                    raise ValueError(f"Value {value} for key {key} is out of range")
        
    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

class ParameterSet:
    def __init__(self, **params):
        self.params = params

    def sample(self):
        sample_data = {}
        
        for param_name, param in self.params.items():
            sample_data[param_name] = param.sample()
        
        return SampleResult(**sample_data)

    def is_within_range(self, sample_result):
        for param_name, param in self.params.items():
            if hasattr(sample_result, param_name):
                value = getattr(sample_result, param_name)
                if not param.is_within_range(value):
                    return False
        return True

class RandomToggledParameterSet(ParameterSet):
    def __init__(self, toggle_probability=0.5, **params):
        super().__init__(**params)
        self.toggle_probability = toggle_probability

    def sample(self):
        sample_data = {}
        for param_name, param in self.params.items():
            if random.random() < self.toggle_probability:
                sample_data[param_name] = param.sample()
        return SampleResult(**sample_data)

class CombinedParameterSet(ParameterSet):
    def __init__(self, *parameter_sets):
        self.parameter_sets = parameter_sets

    def sample(self):
        combined_result = SampleResult()
        for param_set in self.parameter_sets:
            result = param_set.sample()
            combined_result = combined_result + result
        return combined_result

class DiscreteParameterSet(ParameterSet):
    def __init__(self, samples):
        self.samples = samples

    def sample(self):
        return random.choice(self.samples)

def constraint_fp2_ge_fp1(sample):
    """Constraint function to ensure fp2 >= fp1."""
    return getattr(sample, 'fp2', float('inf')) >= getattr(sample, 'fp1', -float('inf'))

def to_new_param_name(d: dict):
    """
    Convert old parameter names to new ones for backward compatibility.
    e.g. R_tx -> R_drv, R_rx -> R_odt
    """
    key_map = {
        'R_tx': 'R_drv', 'C_tx': 'C_drv', 'L_tx': 'L_drv',
        'R_rx': 'R_odt', 'C_rx': 'C_odt', 'L_rx': 'L_odt',
    }
    for old_key, new_key in key_map.items():
        if old_key in d:
            d[new_key] = d.pop(old_key)
    return d

# Constraint ranges placeholder - can be populated if needed
CONSTRAINT_RANGES = {}
