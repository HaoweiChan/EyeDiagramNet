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
            values = np.concatenate((values, additional_array))
        
        # Determine precision
        precision = max(_get_decimal_places(self.low), _get_decimal_places(self.step or self.low))

        # Round using dynamically determined precision
        values = np.round(values, decimals=precision)

        return np.random.choice(values) * self.scaler
        
    def is_within_range(self, value):
        if value in self.additional_values:
            return True
        return self.low * self.scaler <= value <= self.high * self.scaler

class LogParameter(Parameter):
    def __init__(self, low, high, step=None, numbers=None, base=10, scaler=1):
        super().__init__(low, high, step, numbers)
        self.base = base
        self.scaler = scaler
        
    def sample(self):
        if self.numbers is not None:
            exponents = np.linspace(self.low, self.high, num=self.numbers, dtype=np.float64)
        elif self.step is not None:
            exponents = np.arange(self.low, self.high + self.step, step=self.step, dtype=np.float64)
        else:
            raise ValueError("Either 'step' or 'numbers' must be specified for LogParameter.")
        
        values = np.power(self.base, exponents)
        
        # Determine precision
        precision = max(_get_decimal_places(self.low), _get_decimal_places(self.step or self.low))

        # Round using dynamically determined precision
        values = np.round(values, decimals=precision)

        return np.random.choice(values) * self.scaler

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
        sorted_keys = sorted(self._data.keys())
        if return_keys:
            return ([self._data[key] for key in sorted_keys], sorted_keys)
        return [self._data[key] for key in sorted_keys]
    
    @classmethod
    def from_list_with_keys(cls, values, keys):
        """Reconstruct SampleResult from list of values and corresponding keys."""
        if len(values) != len(keys):
            raise ValueError(f"Length mismatch: {len(values)} values, {len(keys)} keys")
        data = dict(zip(keys, values))
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data_dict):
        """Create SampleResult from dictionary."""
        return cls(**data_dict)
        
    def to_array(self):
        return np.array(self.to_list())

    def to_structured_array(self):
        """
        Convert boundary conditions to structured array format for StructuredGatedBoundaryProcessor.
        
        Expected order: [electrical(6), signal(3), ctle(4)]
        - Electrical: R_drv, R_odt, C_drv, C_odt, L_drv, L_odt
        - Signal: pulse_amplitude, bits_per_sec, vmask  
        - CTLE: AC_gain, DC_gain, fp1, fp2
        
        Returns:
            numpy array of shape (13,) with structured boundary conditions
        """
        # Define expected parameter order
        electrical_params = ['R_drv', 'R_odt', 'C_drv', 'C_odt', 'L_drv', 'L_odt']
        signal_params = ['pulse_amplitude', 'bits_per_sec', 'vmask']
        ctle_params = ['AC_gain', 'DC_gain', 'fp1', 'fp2']
        
        # Extract values in the correct order, filling with NaN if missing
        structured_values = []
        
        # Electrical parameters
        for param in electrical_params:
            structured_values.append(self._data.get(param, np.nan))
            
        # Signal parameters  
        for param in signal_params:
            structured_values.append(self._data.get(param, np.nan))
            
        # CTLE parameters (these may be NaN if not present)
        for param in ctle_params:
            structured_values.append(self._data.get(param, np.nan))
            
        return np.array(structured_values, dtype=np.float32)

    def is_within_range(self, parameter_set):
        if isinstance(parameter_set, str):
            parameter_set = globals()[parameter_set]
        for key, value in self._data.items():
            if key not in parameter_set.params:
                raise ValueError(f"Parameter '{key}' not found in ParameterSet")
            param = parameter_set.params[key]
            if not param.is_within_range(value):
                raise ValueError(f"Parameter '{key}' is not within valid range")

class ParameterSet:
    def __init__(self, **params):
        self._params = params
        self._constraints = []

    @property
    def params(self):
        return self._params

    def __add__(self, other):
        if not isinstance(other, ParameterSet):
            raise TypeError("Can only add ParameterSet instances")

        # Combine the parameters from self and other
        new_params = self._params.copy()

        for key, value in other.params.items():
            if key in new_params:
                raise ValueError(f"Conflicting key '{key}' found in both ParameterSet instances")
            new_params[key] = value

        # Return a ParameterSet instance
        if isinstance(other, RandomToggledParameterSet):
            return CombinedParameterSet(self, other)
        return ParameterSet(**new_params)

    def add_constraint(self, constraint_func):
        self._constraints.append(constraint_func)

    def print_keys(self):
        print(sorted(self._params.keys()))

    def sample(self):
        while True:
            samples = {}
            for name, param in self._params.items():
                if isinstance(param, Parameter):
                    samples[name] = param.sample()
                else:
                    samples[name] = param

            if all(constraint(samples) for constraint in self._constraints):
                return SampleResult(**samples)

class RandomToggledParameterSet(ParameterSet):
    def __init__(self, toggle_probability=0.5, **params):
        super().__init__(**params)
        self.toggle_probability = toggle_probability

    def sample(self):
        if random.random() < self.toggle_probability:
            return super().sample()
        
        nan_samples = {name: float('nan') for name in self._params.keys()}
        return SampleResult(**nan_samples)

class CombinedParameterSet(ParameterSet):
    def __init__(self, static_set, toggling_set):
        if not isinstance(static_set, ParameterSet) or not isinstance(toggling_set, RandomToggledParameterSet):
            raise TypeError("static_set must be ParameterSet, and toggling_set must be RandomToggledParameterSet")

        self.static_set = static_set
        self.toggling_set = toggling_set
        
    def sample(self):
        static_sample = self.static_set.sample()
        toggling_sample = self.toggling_set.sample()

        combined = static_sample.to_dict()
        combined.update(toggling_sample.to_dict())

        return SampleResult(**combined)

class DiscreteParameterSet(ParameterSet):
    def __init__(self, cases):
        """
        cases: list of dicts, each dict is a full parameter set
        """
        self.cases = cases

    def sample(self):
        selected = random.choice(self.cases)
        return SampleResult(**selected)

# Signal Protocol Parameters
DDR_PARAMS = ParameterSet(
    R_drv=LinearParameter(low=20, high=50, step=10),
    R_odt=LinearParameter(low=40, high=120, step=10, additional_values=[1e9]),
    C_drv=LinearParameter(low=0.1, high=1., step=0.1, scaler=1e-12),
    C_odt=LinearParameter(low=0.1, high=1., step=0.1, scaler=1e-12),
    L_drv=LinearParameter(low=0., high=2., step=0.2, scaler=1e-9),
    L_odt=LinearParameter(low=0., high=2., step=0.2, scaler=1e-9),
    pulse_amplitude=LinearParameter(low=0.4, high=0.6, step=0.1),
    bits_per_sec=LinearParameter(low=6.4, high=9.6, numbers=5, scaler=1e9),
    vmask=LinearParameter(low=0.04, high=0.05, step=0.01)
)

HBM2_PARAMS = ParameterSet(
    R_drv=LinearParameter(low=8, high=20, step=5),
    R_odt=DiscreteParameter(values=[1e9]),
    C_drv=LinearParameter(low=0.1, high=0.5, step=0.1, scaler=1e-12),
    C_odt=LinearParameter(low=0.1, high=0.5, step=0.1, scaler=1e-12),
    L_drv=LinearParameter(low=0., high=2., step=0.2, scaler=1e-9),
    L_odt=LinearParameter(low=0., high=2., step=0.2, scaler=1e-9),
    pulse_amplitude=LinearParameter(low=0.35, high=0.45, step=0.1),
    bits_per_sec=LinearParameter(low=10, high=12.8, numbers=8, scaler=1e9),
    vmask=DiscreteParameter(values=[0.05])
)

UCIE_PARAMS = ParameterSet(
    R_drv=LinearParameter(low=20, high=40, step=5),
    R_odt=DiscreteParameter(values=[1e9]),
    C_drv=LinearParameter(low=0.1, high=0.5, step=0.1, scaler=1e-12),
    C_odt=LinearParameter(low=0.1, high=0.5, step=0.1, scaler=1e-12),
    L_drv=LinearParameter(low=0., high=2., step=0.2, scaler=1e-9),
    L_odt=LinearParameter(low=0., high=2., step=0.2, scaler=1e-9),
    pulse_amplitude=LinearParameter(low=0.3, high=0.8, step=0.1),
    bits_per_sec=LinearParameter(low=10, high=12.8, numbers=8, scaler=1e9),
    vmask=DiscreteParameter(values=[0.05])
)

MIX_PARAMS = ParameterSet(
    R_drv=LinearParameter(low=5, high=40, step=5),
    R_odt=DiscreteParameter(values=[1e9]),
    C_drv=LinearParameter(low=0.1, high=0.5, step=0.1, scaler=1e-12),
    C_odt=LinearParameter(low=0.1, high=0.5, step=0.1, scaler=1e-12),
    L_drv=LinearParameter(low=0., high=2., step=0.2, scaler=1e-9),
    L_odt=LinearParameter(low=0., high=2., step=0.2, scaler=1e-9),
    pulse_amplitude=LinearParameter(low=0.3, high=0.8, step=0.1),
    bits_per_sec=LinearParameter(low=10, high=32, step=2, scaler=1e9),
    vmask=LinearParameter(low=0.02, high=0.05, step=0.01)
)

DER_PARAMS = DiscreteParameterSet([
    {
        "R_drv": 14.45,
        "R_odt": 1e5,
        "C_drv": 2.2e-13,
        "C_odt": 4e-13,
        "bits_per_sec": 12.8e9,
        "vmask": 0.05,
        "vh": 0.3772,
        "vl": 0.0,
        "tvl": 1.51e-10,
        "tvh": 1.59e-10,
        "tr_rising": 2.5e-11,
        "vp": 0.3772,
        "tvp": 6e-11,
        "tf_rising": 1.5e-11,
        "tf_falling": 1.2e-11
    },
    {
        "R_drv": 14.45,
        "R_odt": 1e5,
        "C_drv": 2.2e-13,
        "C_odt": 4e-13,
        "bits_per_sec": 12.8e9,
        "vmask": 0.05,
        "vh": 0.3243,
        "vl": 0.0,
        "tvl": 1.52e-10,
        "tvh": 1.55e-10,
        "tr_rising": 2.5e-11,
        "vp": 0.3772,
        "tvp": 7e-11,
        "tf_rising": 1.5e-11,
        "tf_falling": 1.3e-11
    },
    {
        "R_drv": 10.47,
        "R_odt": 1e5,
        "C_drv": 2.2e-13,
        "C_odt": 4e-13,
        "bits_per_sec": 12.8e9,
        "vmask": 0.05,
        "vh": 0.3772,
        "vl": 0.0,
        "tvl": 1.52e-10,
        "tvh": 1.58e-10,
        "tr_rising": 2.5e-11,
        "vp": 0.3772,
        "tvp": 7e-11,
        "tf_rising": 1.5e-11,
        "tf_falling": 1.3e-11
    },
    {
        "R_drv": 10.47,
        "R_odt": 1e5,
        "C_drv": 2.2e-13,
        "C_odt": 4e-13,
        "bits_per_sec": 12.8e9,
        "vmask": 0.05,
        "vh": 0.3243,
        "vl": 0.0,
        "tvl": 1.52e-10,
        "tvh": 1.55e-10,
        "tr_rising": 2.5e-11,
        "vp": 0.3772,
        "tvp": 7e-11,
        "tf_rising": 1.5e-11,
        "tf_falling": 1.3e-11
    }
])

# CTLE Parameters
CTLE_PARAMS = RandomToggledParameterSet(
    toggle_probability=0.5,
    AC_gain=LinearParameter(low=0, high=5, step=0.25),
    DC_gain=LinearParameter(low=0.3, high=3, step=0.3),
    fp1=LinearParameter(low=5, high=50, step=5, scaler=1e9),
    fp2=LinearParameter(low=5, high=50, step=5, scaler=1e9)
)

def constraint_fp2_ge_fp1(sample):
    return sample["fp2"] >= sample["fp1"]

CTLE_PARAMS.add_constraint(constraint_fp2_ge_fp1)

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

PARAM_SETS_MAP = {
    'DDR_PARAMS': DDR_PARAMS,
    'HBM2_PARAMS': HBM2_PARAMS, 
    'UCIE_PARAMS': UCIE_PARAMS,
    'MIX_PARAMS': MIX_PARAMS,
    'CTLE_PARAMS': CTLE_PARAMS,
    'DER_PARAMS': DER_PARAMS
}