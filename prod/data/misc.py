import torch
import scipy
import numpy as np
import skrf as rf
from dataclasses import dataclass, field, fields

class Axis:
    """numpy.linspace object with its datas."""

    def __init__(self, start, stop, num, endpoint):
        self.start = start
        self.stop = stop
        self.num = num
        self.endpoint = endpoint
        self.array, self.step = np.linspace(start=start, stop=stop, num=num, endpoint=endpoint, retstep=True)


class TrapezoidalPulse:
    """A symmetrical trapezoidal pulse."""

    def __init__(self, t_rise_p: float, t_start_p: float, pulse_amplitude: float, width_p: float):
        self.t_rise_p = t_rise_p
        self.t_start_p = t_start_p
        self.pulse_amplitude = pulse_amplitude
        self.width_p = width_p

    def freq_dom(self, f_array: np.ndarray) -> np.ndarray:
        """Compute the Fourier transform of the trapezoidal pulse for a given frequency array."""
        b = 0
        bp = self.t_start_p
        L = self.t_rise_p
        Lp = self.width_p
        A = 1
        Ap = self.pulse_amplitude / self.t_rise_p

        factor1 = np.exp(-1j * 2 * np.pi * (b + L / 2) * f_array) * A * L * np.sinc(L * f_array)
        factor2 = np.exp(-1j * 2 * np.pi * (bp + Lp / 2) * f_array) * Ap * Lp * np.sinc(Lp * f_array)
        F_tra = factor1 * factor2

        return F_tra

    def time_dom(self, t_array: np.ndarray) -> np.ndarray:
        """Compute the trapezoidal pulse for a given time array."""
        tra = np.zeros_like(t_array)

        # Rising edge
        rise_mask = (self.t_start_p <= t_array) & (t_array <= self.t_start_p + self.t_rise_p)
        tra[rise_mask] = self.pulse_amplitude * (t_array[rise_mask] - self.t_start_p) / self.t_rise_p

        # High level
        high_mask = (self.t_start_p + self.t_rise_p < t_array) & (t_array < self.t_start_p + self.width_p)
        tra[high_mask] = self.pulse_amplitude

        # Falling edge
        fall_mask = (self.t_start_p + self.width_p <= t_array) & (t_array <= self.t_start_p + self.width_p + self.t_rise_p)
        tra[fall_mask] = self.pulse_amplitude - self.pulse_amplitude * (t_array[fall_mask] - (self.t_start_p + self.width_p)) / self.t_rise_p

        return tra


@dataclass
class TransientParams:
    """PDAW constants.

    Instance variables used in instantiation
    -----------------------------------------
    :param R_tx: port impedance of input ports (first half of the ports) after renormalization (assume that it also equals the output impedance (excluding shunt capacitance) of the channel).
    :param R_rx: port impedance of output ports (second half of the ports) after renormalization (assume that it also equals the input impedance (excluding shunt capacitance) of the channel).
    :param C_tx: value of capacitance to be added to each input port (each of the first half of the ports) of network (ntwk).
    :param C_rx: value of capacitance to be added to each output port (each of the second half of the ports) of network (ntwk).
    :param pulse_amplitude: max value of the pulse (values are voltages of voltage source, such as VDD0).
    :param bits_per_sec: number of bits (UI/s) per second.
    :param vmask: locations of upper, lower boundary of eye mask are (vref + vmask), (vref - vmask) respectively.
    :param DC_gain: DC gain of the CTLE.
    :param AC_gain: AC gain of the CTLE.
    :param fp1: first pole frequency of the CTLE.
    :param fp2: second pole frequency of the CTLE.
    :param n_perUI_intrp: number of points per UI in interpolated SBR data.
    :param f_trunc: data of frequencies > f_trunc will be removed.
    :param snp_path_z0: port impedance of path of s1p file (smp_path).
    :param t_stop_divided_by_UI_dur: t stop / duration of an UI (should be integer); time points of single bit response are { 0, t_step, ..., (t_num - 1)*t_step = t_stop }.
    :param UI_dur_divided_by_t_step: duration of an_UI / t_step (should be integer) (At least one of t_stop_divided_by_UI_dur and UI_dur_divided_by_t_step should be even.).
    :param t_rise_p_divided_by_UI_dur: (rise time of the pulse in seconds) / duration of an_UI.
    :param t_start_p_divided_by_UI_dur: (time of starting rising of the pulse in seconds) / duration_of_an_UI.
    """
    R_tx: float
    R_rx: float
    C_tx: float
    C_rx: float
    pulse_amplitude: float
    bits_per_sec: float
    vmask: float
    DC_gain: float = None
    AC_gain: float = None
    fp1: float = None
    fp2: float = None
    n_perUI_intrp: int = 1000
    f_trunc: float = 1e12
    snp_path_z0: float = 50
    t_stop_divided_by_UI_dur: int = 100
    UI_dur_divided_by_t_step: int = 100
    t_rise_p_divided_by_UI_dur: float = 0.1
    t_start_p_divided_by_UI_dur: float = 1
    UI_dur: float = field(init=False)
    t_step_intrp: float = field(init=False)
    vref_dfl: float = field(init=False)
    time_axis_sbr: Axis = field(init=False)
    pulse: TrapezoidalPulse = field(init=False)

    def __post_init__(self):
        self.UI_dur = 1 / self.bits_per_sec

        # time step of interpolated SBR data
        self.t_step_intrp = self.UI_dur / self.n_perUI_intrp

        # default vref (used when the normal procedure can't determine vref and det_vref == 'auto')
        self.vref_dfl = 0.005 * round((self.pulse_amplitude * (self.R_rx / (self.R_tx + self.R_rx)) / 2) / 0.005)
        self.t_num = self.t_stop_divided_by_UI_dur * self.UI_dur_divided_by_t_step + 1
        self.t_stop = self.t_stop_divided_by_UI_dur * self.UI_dur

        # time axis of single bit response
        self.time_axis_sbr = Axis(start=0, stop=self.t_stop, num=self.t_num, endpoint=True)

        # rise time of the pulse (in seconds)
        self.t_rise_p = self.t_rise_p_divided_by_UI_dur * self.UI_dur

        # time of starting rising of the pulse
        self.t_start_p = self.t_start_p_divided_by_UI_dur * self.UI_dur

        # width of the pulse at half of the height (in seconds) (this is equal to the duration of UI)
        self.width_p = self.UI_dur

        # create trapezoidal pulse
        self.pulse = TrapezoidalPulse(t_rise_p=self.t_rise_p, t_start_p=self.t_start_p, pulse_amplitude=self.pulse_amplitude, width_p=self.width_p)

    @classmethod
    def from_config(cls, config):
        valid_fields = {f.name for f in fields(cls)}
        filtered_config = {key: value for key, value in config.items() if key in valid_fields}
        return cls(**filtered_config)


def rsolve_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solves x @ A = B using PyTorch.
    Same as B @ torch.linalg.inv(A) but avoids calculating the inverse and
    should be numerically slightly more accurate.
    """
    return torch.linalg.solve(A.transpose(-2, -1).conj(), B.transpose(-2, -1).conj()).transpose(-2, -1).conj()


def nudge_eig_torch(mat: torch.Tensor, cond: float = 1e-9, min_eig: float = 1e-12) -> torch.Tensor:
    """Nudge eigenvalues with absolute value smaller than
    max(cond * max(eigenvalue), min_eig) to that value.
    Can be used to avoid singularities in solving matrix equations.
    """
    eigw, eigv = [], []
    for mat_freq in mat.cpu().numpy():
        eigw_freq, eigv_freq = scipy.linalg.eig(mat_freq)
        eigw.append(eigw_freq)
        eigv.append(eigv_freq)
    eigw = np.array(eigw)
    eigv = np.array(eigv)
    eigw = torch.from_numpy(eigw).to(mat.device)
    eigv = torch.from_numpy(eigv).to(mat.device)

    max_eig = torch.amax(torch.abs(eigw), dim=1)
    mask = (torch.abs(eigw) < cond * max_eig.unsqueeze(-1)) | (torch.abs(eigw) < min_eig)
    if not mask.any():
        return mat

    mask_cond = cond * max_eig.unsqueeze(-1).expand_as(eigw)[mask]
    mask_min = min_eig * torch.ones_like(mask_cond)
    eigw[mask] = torch.maximum(mask_cond, mask_min).to(eigw.dtype)

    e = torch.zeros_like(mat)
    e.diagonal(dim1=-2, dim2=-1).copy_(eigw)
    return rsolve_torch(eigv, eigv @ e)

def nudge_svd(mat: torch.Tensor, cond: float = 1e-9, min_svd: float = 1e-12) -> torch.Tensor:
    U, s, Vh = torch.linalg.svd(mat)
    max_svd = torch.amax(s, dim=1)
    mask = (s < cond * max_svd[:, None]) | (s < min_svd)
    if not mask.any():
        return mat

    mask_cond = cond * max_svd[:, None].repeat(1, mat.shape[-1])[mask]
    mask_min = min_svd * torch.ones_like(mask_cond)
    s[mask] = torch.maximum(mask_cond, mask_min)

    S = torch.zeros_like(mat)
    S.diagonal(dim1=-2, dim2=-1).copy_(s)
    return torch.einsum('...ij,...jk,...lk->...il', U, S, Vh)


def s2z_torch(s: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Convert scattering parameters to impedance parameters using PyTorch."""
    # Power-waves. Eq (19) from [Kurokawa et al.]
    nfreqs, nports, _ = s.shape

    Id = torch.eye(nports, dtype=s.dtype, device=s.device).expand(nfreqs, -1, -1)
    F = torch.zeros_like(s)
    G = torch.zeros_like(s)
    F.diagonal(dim1=-2, dim2=-1).copy_(1.0 / (2 * torch.sqrt(z0.real)))
    G.diagonal(dim1=-2, dim2=-1).copy_(z0)
    z = torch.linalg.solve(nudge_eig_torch((Id - s) @ F), (s @ G + G.conj()) @ F)
    return z

def z2s_torch(z: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Convert impedance parameters to scattering parameters using PyTorch."""
    # Power-waves. Eq (18) from [Kurokawa et al.]
    F = torch.zeros_like(z)
    G = torch.zeros_like(z)
    F.diagonal(dim1=-2, dim2=-1).copy_(1.0 / (2 * torch.sqrt(z0.real)))
    G.diagonal(dim1=-2, dim2=-1).copy_(z0)
    s = rsolve_torch(F @ (z + G), F @ (z - G.conj()))
    return s

def s2y_torch(s: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Convert scattering parameters to admittance parameters using PyTorch."""
    nfreqs, nports, _ = s.shape

    Id = torch.eye(nports, dtype=s.dtype, device=s.device).expand(nfreqs, -1, -1)

    F = torch.zeros_like(s)
    G = torch.zeros_like(s)
    F.diagonal(dim1=-2, dim2=-1).copy_(1.0 / (2 * torch.sqrt(z0.real)))
    G.diagonal(dim1=-2, dim2=-1).copy_(z0)
    y = rsolve_torch((s @ G + G.conj()) @ F, (Id - s) @ F)
    return y

def y2s_torch(y: torch.Tensor, z0: torch.Tensor, epsilon=1e-4) -> torch.Tensor:
    """Convert admittance parameters to scattering parameters using PyTorch."""
    nfreqs, nports, _ = y.shape

    # Add a small real part in case of pure imaginary char impedance
    z0[z0.real == 0] += epsilon

    Id = torch.eye(nports, dtype=y.dtype, device=y.device).expand(nfreqs, -1, -1)

    F = torch.zeros_like(y)
    G = torch.zeros_like(y)
    F.diagonal(dim1=-2, dim2=-1).copy_(1.0 / (2 * torch.sqrt(z0.real)))
    G.diagonal(dim1=-2, dim2=-1).copy_(z0)
    s = rsolve_torch(F @ (Id + G @ y), F @ (Id - G.conj() @ y))
    return s


def add_cap_to_port(ntwk, C_tx, C_rx, device='cuda'):
    """
    Given the network ntwk and the values C_tx, C_rx of the shunt capacitors,
    return a network ntwk and C representing the result of adding a shunt capacitor of value C_tx (resp. C_rx)
    to each of the first (resp. second) half of the ports of the network
    """
    freq = ntwk.frequency
    n_port = ntwk.number_of_ports
    if n_port % 2 != 0:
        raise RuntimeError('ntwk should be 2n-port.')

    # Calculate the admittance of the capacitors
    omega = 2 * np.pi * freq.f
    y_tx = 1j * omega * C_tx
    y_rx = 1j * omega * C_rx

    y_tx = torch.from_numpy(y_tx).view(-1, 1, 1)
    y_rx = torch.from_numpy(y_rx).view(-1, 1, 1)
    eye_tensor = torch.eye(n_port, dtype=torch.complex128).unsqueeze(0) #.expand(len(freq.f), -1, -1)
    top_left_mask = torch.flip(torch.triu(torch.ones(n_port, n_port), diagonal=0), dims=[1]).unsqueeze(0)
    bottom_right_mask = torch.flip(torch.tril(torch.ones(n_port, n_port), diagonal=0), dims=[1]).unsqueeze(0)
    cap_list = eye_tensor * (top_left_mask * y_tx + bottom_right_mask * y_rx)

    s_torch = torch.from_numpy(ntwk.s).to(device)
    z0_torch = torch.from_numpy(ntwk.z0).to(device)
    cap_list_torch = cap_list.to(device)

    y_torch = s2y_torch(s_torch, z0_torch)
    total_admittance = cap_list_torch + y_torch
    s_new = y2s_torch(total_admittance, z0_torch)

    if s_new.is_cuda:
        s_new = s_new.cpu().numpy()
    else:
        s_new = s_new.resolve_conj().numpy()

    # Clear GPU cache
    torch.cuda.empty_cache()

    ntwk_and_C = rf.Network(frequency=freq, s=s_new, z0=ntwk.z0)
    return ntwk_and_C

def add_inductance_torch(ntwk, L_tx, L_rx, device='cuda'):
    freq = ntwk.frequency
    num_ports = ntwk.nports

    # Convert S-parameters to torch tensors
    s_tensor = torch.from_numpy(ntwk.s).to(device)

    # Calculate inductive impedance for each frequency
    omega = 2 * np.pi * freq.f
    zL_in = 1j * omega * L_tx
    zL_out = 1j * omega * L_rx

    # Convert inductive impedance to torch tensors
    zL_in_tensor = torch.from_numpy(zL_in[:, None]).to(device)
    zL_out_tensor = torch.from_numpy(zL_out[:, None]).to(device)

    # Convert S-parameters to impedance parameters using PyTorch
    z_tensor = s2z_torch(s_tensor, torch.from_numpy(ntwk.z0).to(device))

    # Add inductive impedance to the diagonal elements of the Z matrix
    in_idx = torch.arange(num_ports // 2)
    out_idx = torch.arange(num_ports // 2, num_ports)
    z_tensor[:, in_idx, in_idx] += zL_in_tensor
    z_tensor[:, out_idx, out_idx] += zL_out_tensor

    # Convert back to S-parameters using PyTorch
    s_tensor = z2s_torch(z_tensor, torch.from_numpy(ntwk.z0).to(device))

    # Move the result back to CPU and convert to numpy array
    if s_tensor.is_cuda:
        s_new = s_tensor.cpu().numpy()
    else:
        s_new = s_tensor.resolve_conj().numpy()

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Create a new Network object with updated S-parameters
    ntwk_and_L = rf.Network(frequency=freq, s=s_new, z0=ntwk.z0)
    return ntwk_and_L

def add_ctle(ntwk, DC_gain, AC_gain, fp1, fp2, eps=0):
    # Define the polynomial for the CTLE
    num = [AC_gain * fp2, DC_gain * fp1 * fp2]
    den = [1, fp1 + fp2, fp1 * fp2]

    # Define the transfer function based on polynomial of the CTLE
    system = scipy.signal.TransferFunction(num, den)

    # Calculate the frequency response
    _, TF = scipy.signal.freqs(system.num, system.den, worN=ntwk.f)
    TF = TF[:, None]

    # Create 2D arrays of all input/output port indices
    n = ntwk.nports
    inp, out = np.indices((n, n))

    # === IL ===
    # if output_port == input_port +/- n//2 => multiply by TF
    # if output_port == input_port ./ n//2 => set to 0
    il_pos = (out == inp + n//2)
    il_neg = (out == inp - n//2)

    # === RL ===
    # if output_port == input_port and input_port > n//2 => set to 1
    rl_one = (out == inp) & (inp >= n//2)

    # === FEXT ===
    # input_port == n//2 and output_port > n//2 => multiply by TF
    # input_port == n//2 and output_port < n//2 => 0
    fext_mul = (out != inp + n//2) & (inp < n//2) & (out >= n//2)
    fext_zero = (out != inp + n//2) & (inp >= n//2) & (out < n//2)

    # === NEXT ===
    # if out == inp:
    #   * input_port += n//2 and output_port > n//2 => set to 0
    next_zero = (out != inp) & (inp >= n//2) & (out >= n//2)

    # Now apply each mask to the s-parameter array.
    # Recall that ntwk.s has shape (nfreq, nport, nport).
    # So we select along axis=1 for output port, axis=2 for input port.
    ntwk.s[:, out[il_pos], inp[il_pos]] *= TF
    ntwk.s[:, out[il_neg], inp[il_neg]] = eps

    ntwk.s[:, out[rl_one], inp[rl_one]] = -1

    ntwk.s[:, out[fext_mul], inp[fext_mul]] *= TF
    ntwk.s[:, out[fext_zero], inp[fext_zero]] = eps

    ntwk.s[:, out[next_zero], inp[next_zero]] = eps

    return ntwk