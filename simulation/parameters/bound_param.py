# Import domain-agnostic parameter types from common
from common.param_types import (
    Parameter, DiscreteParameter, LinearParameter, LogParameter,
    SampleResult, ParameterSet, RandomToggledParameterSet, 
    CombinedParameterSet, DiscreteParameterSet,
    constraint_fp2_ge_fp1, to_new_param_name
)

# Signal Protocol Parameters - Simulation-specific parameter sets
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

# CTLE Parameters with constraints
CTLE_PARAMS = RandomToggledParameterSet(
    toggle_probability=0.5,
    AC_gain=LinearParameter(low=0, high=5, step=0.25),
    DC_gain=LinearParameter(low=0.3, high=3, step=0.3),
    fp1=LinearParameter(low=5, high=50, step=5, scaler=1e9),
    fp2=LinearParameter(low=5, high=50, step=5, scaler=1e9)
)

# Custom constraint function for CTLE parameters
def constraint_fp2_ge_fp1_ctle(sample):
    """Custom constraint for CTLE parameters: fp2 >= fp1."""
    return sample["fp2"] >= sample["fp1"]

# Apply constraint to CTLE_PARAMS if it has an add_constraint method
if hasattr(CTLE_PARAMS, 'add_constraint'):
    CTLE_PARAMS.add_constraint(constraint_fp2_ge_fp1_ctle)

# Parameter sets mapping for simulation configuration
PARAM_SETS_MAP = {
    'DDR_PARAMS': DDR_PARAMS,
    'HBM2_PARAMS': HBM2_PARAMS, 
    'UCIE_PARAMS': UCIE_PARAMS,
    'MIX_PARAMS': MIX_PARAMS,
    'CTLE_PARAMS': CTLE_PARAMS,
    'DER_PARAMS': DER_PARAMS
}