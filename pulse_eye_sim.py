import torch
import pathlib
import argparse
import skrf as rf
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field, fields

###############################################################################
# ---  Source pulse models  ---
###############################################################################
class Pulse:
    """Generic helper that stores a time-axis."""

    def __init__(self, start: float, stop: float, num: int, endpoint: bool = True):
        self.t, self.dt = np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True)


class TrapezoidalPulse(Pulse):
    """Symmetric trapezoid with linear flanks.

    Args
    ----
    t_rise : seconds            - linear edge duration
    t_start: seconds            - time offset of pulse start (0 → leading rising edge begins at 0)
    amplitude : volts / amps    - plateau level
    width    : seconds          - plateau width **excluding** the two rise/fall edges
    """

    def __init__(self, t_rise: float, t_start: float, amplitude: float, width: float):
        super().__init__(start=0.0, stop=t_start + 2 * t_rise + width, num=8193, endpoint=True)
        self.t_rise = t_rise
        self.t_start = t_start
        self.amplitude = amplitude
        self.width = width

    # ---------------------------------------------------------------------
    #   Time domain
    # ---------------------------------------------------------------------
    def time_domain(self, t: np.ndarray) -> np.ndarray:
        v = np.zeros_like(t)
        # rise
        m_rise = (self.t_start <= t) & (t < self.t_start + self.t_rise)
        v[m_rise] = self.amplitude * (t[m_rise] - self.t_start) / self.t_rise
        # plateau
        m_high = (
            self.t_start + self.t_rise
            <= t
        ) & (t < self.t_start + self.t_rise + self.width)
        v[m_high] = self.amplitude
        # fall
        m_fall = (
            self.t_start + self.t_rise + self.width
            <= t
        ) & (
            t < self.t_start + self.t_rise + self.width + self.t_rise
        )
        v[m_fall] = self.amplitude * (
            1 - (t[m_fall] - (self.t_start + self.t_rise + self.width)) / self.t_rise
        )
        return v

    # ---------------------------------------------------------------------
    #   Frequency domain (analytical FT)
    # ---------------------------------------------------------------------
    def freq_domain(self, f: np.ndarray) -> np.ndarray:
        # closed-form for symmetric trapezoid via super-position of two linear ramps & a flat top
        tri = np.sinc(f * self.t_rise) ** 2  # triangle FT
        plate = np.exp(-1j * 2 * np.pi * f * (self.t_start + self.t_rise)) * self.width * np.sinc(f * self.width)
        return self.amplitude * (tri + plate)

###############################################################################
# ---  Simulation parameters  ---
###############################################################################
@dataclass
class TrainerParams:
    # impedance / caps ---------------------------------------------------
    R_tx: float
    R_rx: float
    C_tx: float
    C_rx: float

    # pulse -------------------------
    pulse_amplitude: float  # volts
    bits_per_sec: float     # line-rate
    vmask: float            # mask margin (V)

    # Optional CTLE ------------------------------------------------------
    DC_gain: Optional[float] = None  # linear (!)  - will be converted to dB inside
    AC_gain: Optional[float] = None  # reserved
    f_z: Optional[float] = None      # zero  (Hz)
    f_p1: Optional[float] = None     # pole1 (Hz)
    f_p2: Optional[float] = None     # pole2 (Hz)

    # Numerical controls -------------------------------------------------
    perT_intrp: int = 256           # samples per UI for eye diagram
    t_stop_divided_by_UI_dur: int = 10
    t_rise_p_divided_by_UI_dur: float = 0.10
    t_start_p_divided_by_UI_dur: float = 0.00

    # Optional Touchstone location --------------------------------------
    sNp_path: Optional[str] = None

    # Public - filled in __post_init__ ----------------------------------
    UI_dur: float = field(init=False)
    t_step_intrp: float = field(init=False)
    t_axis_sbr: np.ndarray = field(init=False)
    pulse: TrapezoidalPulse = field(init=False)
    vref_dflt: float = field(init=False)

    # --------------------------------------------------------------------
    def __post_init__(self):
        self.UI_dur = 1.0 / self.bits_per_sec
        self.t_step_intrp = self.UI_dur / self.perT_intrp

        # single-bit-response window length (≥10 UIs works fine)
        t_stop = self.UI_dur * self.t_stop_divided_by_UI_dur
        n_pts = int(round(t_stop / self.t_step_intrp)) + 1
        self.t_axis_sbr = np.linspace(0.0, t_stop, n_pts, endpoint=True)

        # build pulse ----------------------------------------------------
        t_rise = self.t_rise_p_divided_by_UI_dur * self.UI_dur
        width = self.UI_dur - 2 * t_rise
        t_start = self.t_start_p_divided_by_UI_dur * self.UI_dur
        self.pulse = TrapezoidalPulse(t_rise, t_start, self.pulse_amplitude, width)

        # default vref @ half steady-state open-circuit divider -----------
        v_half = self.pulse_amplitude * (self.R_rx / (self.R_tx + self.R_rx)) / 2.0
        self.vref_dflt = 0.005 * round(v_half / 0.005)  # snap to 5 mV grid

    # convenience --------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict):
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in allowed})

###############################################################################
# ---  Network helpers  ---
###############################################################################

def add_cap_to_ports(ntwk: rf.Network, C_tx: float, C_rx: float) -> rf.Network:
    """Add ideal shunt capacitors *per port* (pF → F) to a 2-n network."""
    if C_tx == 0 and C_rx == 0:
        return ntwk.copy()

    n_half = ntwk.nports // 2
    if ntwk.nports != 2 * n_half:
        raise ValueError("Network must be even-port (tx|rx pairs).")

    w = 2 * np.pi * ntwk.f  # rad/s
    Yc_tx = 1j * w * C_tx * 1e-12  # S
    Yc_rx = 1j * w * C_rx * 1e-12

    Y_ntwk = rf.s2y(ntwk.s, ntwk.z0)
    Y_total = Y_ntwk.copy()
    for p in range(ntwk.nports):
        Y_total[:, p, p] += Yc_tx if p < n_half else Yc_rx

    s_new = rf.y2s(Y_total, ntwk.z0)
    return rf.Network(f=ntwk.frequency, s=s_new, z0=ntwk.z0)

def add_ctle(ntwk: rf.Network, dc_gain_lin: float, fz: float, fp1: float, fp2: float) -> rf.Network:
    if any(x is None for x in (dc_gain_lin, fz, fp1, fp2)):
        return ntwk.copy()

    w = 2 * np.pi * ntwk.f
    H = dc_gain_lin * (1 + 1j * w / (2 * np.pi * fz)) / (
        (1 + 1j * w / (2 * np.pi * fp1)) * (1 + 1j * w / (2 * np.pi * fp2))
    )

    s = ntwk.s.copy()
    n_half = ntwk.nports // 2
    for i in range(n_half):
        s[:, i + n_half, i] *= H  # forward path S21 etc.
    return rf.Network(f=ntwk.frequency, s=s, z0=ntwk.z0)

def renormalise(ntwk: rf.Network, R_tx: float, R_rx: float) -> rf.Network:
    n_half = ntwk.nports // 2
    Z_new = np.array([R_tx] * n_half + [R_rx] * n_half)
    return ntwk.renormalize(z_new=Z_new)

###############################################################################
# ---  Math helpers  ---
###############################################################################

def fft_frequency_axis(t: np.ndarray) -> np.ndarray:
    dt = t[1] - t[0]
    return np.fft.fftfreq(len(t), dt)


def icft(data_f: np.ndarray, f_axis: np.ndarray) -> np.ndarray:
    df = f_axis[1] - f_axis[0]
    return np.fft.ifft(data_f, axis=0) * len(f_axis) * df

###############################################################################
# ---  Core pipeline  ---
###############################################################################

def get_line_sbr(params: TrainerParams, ntwk: rf.Network) -> np.ndarray:
    """Return SBR array [t, n_rx, n_tx]."""
    f_axis = fft_frequency_axis(params.t_axis_sbr)
    # interpolate S to FFT bins (cheap linear on real/imag)
    s_interp = np.zeros((len(f_axis), ntwk.nports, ntwk.nports), dtype=complex)
    for r in range(ntwk.nports):
        for c in range(ntwk.nports):
            re = np.interp(f_axis, ntwk.f, np.real(ntwk.s[:, r, c]))
            im = np.interp(f_axis, ntwk.f, np.imag(ntwk.s[:, r, c]))
            s_interp[:, r, c] = re + 1j * im

    # pulse spectrum (aligned to fft order)
    pulse_f = np.fft.ifftshift(params.pulse.freq_domain(np.fft.fftshift(f_axis)))

    n_half = ntwk.nports // 2
    sbr_f = np.zeros((len(f_axis), n_half, n_half), complex)
    for ri in range(n_half):
        for ti in range(n_half):
            sbr_f[:, ri, ti] = s_interp[:, ri + n_half, ti] * pulse_f

    sbr_t = icft(sbr_f, f_axis).real  # → [t, rx, tx]
    return sbr_t

def pda_eye(sbr_aligned: np.ndarray, samples_per_ui: int) -> np.ndarray:
    h = sbr_aligned[:, 0]  # assume through path, [t]
    L = len(h)
    eye_min = np.zeros(samples_per_ui)
    eye_max = np.zeros(samples_per_ui)
    n_side = (L // samples_per_ui) // 2

    for t in range(samples_per_ui):
        isi = 0.0
        for k in range(-n_side, n_side + 1):
            if k == 0:
                continue
            idx = t - k * samples_per_ui
            if 0 <= idx < L:
                isi += abs(h[idx])
        mc = h[t] if t < L else 0.0
        eye_max[t] = mc + isi
        eye_min[t] = mc - isi
    return np.column_stack((eye_min, eye_max))

def optimise_vref(params: TrainerParams, eye: np.ndarray) -> Tuple[float, float]:
    opening = eye[:, 1] - eye[:, 0]
    if (opening <= 0).all():
        return params.vref_dflt, 0.0
    idx = opening.argmax()
    vref = eye[idx].mean()

    mask = (eye[:, 1] > vref + params.vmask) & (eye[:, 0] < vref - params.vmask)
    # longest run of True → eye-width
    run = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    width_samples = (run[1::2] - run[::2]).max() if run.size else 0
    return vref, width_samples * params.t_step_intrp

###############################################################################
# ---  Optional pattern → waveform  ---
###############################################################################

def prbs(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n) * 2 - 1  # ±1

def waveforms_from_pattern(pattern: np.ndarray, sbr: np.ndarray, samples_per_ui: int) -> np.ndarray:
    """Depthwise conv1d super-position (no crosstalk)."""
    n_lines = sbr.shape[1]
    # upsample pattern (insert zeros between UIs)
    ups = torch.zeros(pattern.shape[1] * samples_per_ui, dtype=torch.float32)
    ups[::samples_per_ui] = torch.from_numpy(pattern[0]).float()  # single-line for demo
    x = ups.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
    h = torch.from_numpy(sbr[:, 0, 0]).float().flip(0).view(1, 1, -1)  # conv1d uses flipped kernel
    y = torch.nn.functional.conv1d(x, h, padding=h.size(-1) - 1)
    return y.squeeze().numpy()

###############################################################################
# ---  Top-level convenience  ---
###############################################################################

def simulate_eye(params: TrainerParams, snp_file: pathlib.Path) -> Tuple[float, float, float]:
    ntwk = rf.Network(str(snp_file))
    ntwk = add_cap_to_ports(ntwk, params.C_tx, params.C_rx)
    if params.DC_gain is not None:
        ntwk = add_ctle(ntwk, params.DC_gain, params.f_z, params.f_p1, params.f_p2)
    ntwk = renormalise(ntwk, params.R_tx, params.R_rx)

    sbr = get_line_sbr(params, ntwk)

    # align SBR so that max(|h|) is at t=0 ------------------------------
    idx_peak = np.abs(sbr[:, 0, 0]).argmax()
    shift = -idx_peak
    pad = params.perT_intrp * 8  # ±4 UIs window
    sbr_aligned = np.zeros((pad, 1, 1))
    valid = slice(max(0, shift), max(0, shift) + len(sbr))
    sbr_aligned[valid, 0, 0] = sbr[:, 0, 0]

    eye = pda_eye(sbr_aligned[:, :, 0], params.perT_intrp)
    vref, ew = optimise_vref(params, eye)
    eh = eye[:, 1] - eye[:, 0]
    return vref, ew, eh.max()

###############################################################################
# ---  CLI / Demo  ---
###############################################################################

def _demo():
    parser = argparse.ArgumentParser(description="PDA eye-width demo")
    parser.add_argument("touchstone", type=pathlib.Path, help=".sNp file (even-port)")
    args = parser.parse_args()

    cfg = dict(
        R_tx=50.0,
        R_rx=50.0,
        C_tx=0.0,
        C_rx=0.0,
        pulse_amplitude=1.0,
        bits_per_sec=25e9,
        vmask=0.05,
        perT_intrp=256,
    )
    params = TrainerParams.from_config(cfg)
    vref, ew, eh = simulate_eye(params, args.touchstone)
    print(f"vref = {vref*1e3:.1f} mV,  eye-width = {ew*1e12:.1f} ps  ({ew/params.UI_dur*100:.1f}% UI),  eye-height = {eh:.3f}")


if __name__ == "__main__":
    _demo()
