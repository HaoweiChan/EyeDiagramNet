import os
import time
import math
import scipy
import traceback
import skrf as rf
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, fields

from .network_utils import s2y, y2s, s2z, z2s, generate_test_pattern

# ===============================================
# CORE DATA STRUCTURES (moved from misc_refactored.py)
# ===============================================

class Axis:
    """A numpy.linspace object with its data and metadata.
    
    Attributes:
        start (float): Start value of the linear space
        stop (float): End value of the linear space
        num (int): Number of points in the linear space
        endpoint (bool): Whether to include the stop value
        array (np.ndarray): The actual linear space array
        step (float): The step size between points
    """

    def __init__(self, start: float, stop: float, num: int, endpoint: bool):
        """Initialize an Axis object.
        
        Args:
            start: Start value of the linear space
            stop: End value of the linear space
            num: Number of points in the linear space
            endpoint: Whether to include the stop value
        """
        self.start = start
        self.stop = stop
        self.num = num
        self.endpoint = endpoint
        self.array, self.step = np.linspace(start=start, stop=stop, num=num, endpoint=endpoint, retstep=True)


class TrapezoidalPulse:
    """A symmetrical trapezoidal pulse with time and frequency domain representations.
    
    Attributes:
        t_rise_p (float): Rise time of the pulse
        t_start_p (float): Start time of the pulse
        pulse_amplitude (float): Maximum amplitude of the pulse
        width_p (float): Width of the pulse at half height
    """

    def __init__(self, t_rise_p: float, t_start_p: float, pulse_amplitude: float, width_p: float):
        """Initialize a TrapezoidalPulse object.
        
        Args:
            t_rise_p: Rise time of the pulse
            t_start_p: Start time of the pulse
            pulse_amplitude: Maximum amplitude of the pulse
            width_p: Width of the pulse at half height
        """
        self.t_rise_p = t_rise_p
        self.t_start_p = t_start_p
        self.pulse_amplitude = pulse_amplitude
        self.width_p = width_p

    def freq_dom(self, f_array: np.ndarray) -> np.ndarray:
        """Compute the Fourier transform of the trapezoidal pulse.
        
        Args:
            f_array: Array of frequency points
            
        Returns:
            Complex array containing the frequency domain representation
        """
        b = 0
        bp = self.t_start_p
        L = self.t_rise_p
        Lp = self.width_p
        A = 1
        Ap = self.pulse_amplitude / self.t_rise_p

        # Pre-compute common terms for better performance
        exp_term1 = np.exp(-1j * 2 * np.pi * (b + L / 2) * f_array)
        exp_term2 = np.exp(-1j * 2 * np.pi * (bp + Lp / 2) * f_array)
        sinc_term1 = np.sinc(L * f_array)
        sinc_term2 = np.sinc(Lp * f_array)

        factor1 = exp_term1 * A * L * sinc_term1
        factor2 = exp_term2 * Ap * Lp * sinc_term2
        return factor1 * factor2


@dataclass
class TransientParams:
    """Configuration parameters for transient analysis.
    
    Attributes:
        R_tx (float): Port impedance of input ports after renormalization
        R_rx (float): Port impedance of output ports after renormalization
        C_tx (float): Capacitance to add to each input port
        C_rx (float): Capacitance to add to each output port
        pulse_amplitude (float): Maximum pulse voltage
        bits_per_sec (float): Bit rate in bits per second
        vmask (float): Eye mask voltage margin
        DC_gain (Optional[float]): DC gain of the CTLE
        AC_gain (Optional[float]): AC gain of the CTLE
        fp1 (Optional[float]): First pole frequency of the CTLE
        fp2 (Optional[float]): Second pole frequency of the CTLE
        n_perUI_intrp (int): Points per UI in interpolated SBR data
        f_trunc (float): Upper frequency limit for truncation
        snp_path_z0 (float): Port impedance for S-parameter files
        t_stop_divided_by_UI_dur (int): Stop time in UI durations
        UI_dur_divided_by_t_step (int): UI duration in time steps
        t_rise_p_divided_by_UI_dur (float): Rise time in UI durations
        t_start_p_divided_by_UI_dur (float): Start time in UI durations
    """
    R_tx: float
    R_rx: float
    C_tx: float
    C_rx: float
    pulse_amplitude: float
    bits_per_sec: float
    vmask: float
    snp_horiz: str
    L_tx: float = 0
    L_rx: float = 0
    snp_tx: Optional[str] = None
    snp_rx: Optional[str] = None
    directions: Optional[list] = None
    DC_gain: Optional[float] = None
    AC_gain: Optional[float] = None
    fp1: Optional[float] = None
    fp2: Optional[float] = None
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
        """Initialize derived parameters after instance creation."""
        # Calculate UI duration
        self.UI_dur = 1 / self.bits_per_sec

        # Calculate interpolated time step
        self.t_step_intrp = self.UI_dur / self.n_perUI_intrp

        # Calculate default reference voltage
        self.vref_dfl = 0.005 * round((self.pulse_amplitude * (self.R_rx / (self.R_tx + self.R_rx)) / 2) / 0.005)
        
        # Calculate time axis parameters
        self.t_num = self.t_stop_divided_by_UI_dur * self.UI_dur_divided_by_t_step + 1
        self.t_stop = self.t_stop_divided_by_UI_dur * self.UI_dur

        # Create time axis for single bit response
        self.time_axis_sbr = Axis(start=0, stop=self.t_stop, num=self.t_num, endpoint=True)

        # Calculate pulse parameters
        self.t_rise_p = self.t_rise_p_divided_by_UI_dur * self.UI_dur
        self.t_start_p = self.t_start_p_divided_by_UI_dur * self.UI_dur
        self.width_p = self.UI_dur

        # Create trapezoidal pulse
        self.pulse = TrapezoidalPulse(
            t_rise_p=self.t_rise_p,
            t_start_p=self.t_start_p,
            pulse_amplitude=self.pulse_amplitude,
            width_p=self.width_p
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TransientParams':
        """Create a TransientParams instance from a configuration dictionary.
        
        Args:
            config: Dictionary containing configuration parameters
            
        Returns:
            TransientParams instance with validated parameters
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_config = {key: value for key, value in config.items() if key in valid_fields}
        return cls(**filtered_config)

# ===============================================
# EYEWIDTH SIMULATOR CLASS
# ===============================================

class EyeWidthSimulator:
    """
    Main class for eye width simulation that aggregates all the processing functions.
    
    This class encapsulates the entire simulation pipeline from S-parameter processing
    to eye width calculation, providing a clean interface and better code organization.
    """
    
    def __init__(self, config, snp_files=None, directions=None):
        """Initialize the EyeWidthSimulator."""
        # Convert config to dictionary format
        if isinstance(config, dict):
            config_dict = config.copy()
        else:
            if hasattr(config, 'to_dict'):
                config_dict = config.to_dict()
            else:
                config_dict = {attr: getattr(config, attr) for attr in dir(config) if not attr.startswith('_')}
        
        # Handle snp_files if passed separately (for backward compatibility)
        if snp_files is not None:
            if isinstance(snp_files, (tuple, list)) and len(snp_files) >= 3:
                config_dict['snp_horiz'] = snp_files[0]
                config_dict['snp_tx'] = snp_files[1]
                config_dict['snp_rx'] = snp_files[2]
            elif isinstance(snp_files, str):
                config_dict['snp_horiz'] = snp_files
        
        # Handle directions if passed separately
        if directions is not None:
            config_dict['directions'] = directions
        
        # Create TransientParams from the updated config
        self.params = TransientParams.from_config(config_dict)
        
        self.ntwk = self._load_and_cascade_networks()
        
        # Assign Tx/Rx directions using the port flip utility
        nlines = self.ntwk.s.shape[1] // 2
        raw_directions = self.params.directions
        
        if raw_directions is None:
            # Old implementation defaulted to all ports being non-flipped (directions=[1]*nlines).
            # The port flip logic uses 0 for flip, so we use 1 (no flip) as the default.
            directions_for_flip = [1] * nlines
            self.directions = [1] * nlines
        else:
            # The provided directions use 0 for flip, which matches our new function.
            directions_for_flip = raw_directions
            self.directions = raw_directions
        
        self.ntwk = self._ntwk_port_flip(self.ntwk, directions_for_flip)
        
        # Store commonly used parameters as instance attributes
        self.num_lines = self.ntwk.s.shape[1] // 2

    def _ntwk_port_flip(self, ntwk, directions):
        """
        Flips network ports based on a directions array.

        A direction of 0 indicates that the corresponding port pair should be flipped.
        For an 8-port network (4 lines), if directions[1] is 0, then ports 1 and 1 + 4 = 5
        will be swapped.

        Args:
            ntwk (skrf.Network): The network whose ports are to be flipped.
            directions (list or np.ndarray): A list where a 0 indicates a port flip.

        Returns:
            skrf.Network: A new network with the specified ports flipped.
        """
        port_flip_list = [i for i, direction in enumerate(directions) if direction == 0]
        
        port_num = ntwk.number_of_ports
        port_index_list = list(range(port_num))

        for port_to_flip in port_flip_list:
            p1 = port_to_flip
            p2 = port_to_flip + port_num // 2
            port_index_list[p1], port_index_list[p2] = port_index_list[p2], port_index_list[p1]

        return ntwk.subnetwork(port_index_list)

    # ===============================================
    # NETWORK COMPONENT METHODS (moved from standalone functions)
    # ===============================================

    def add_capacitance(self, ntwk: rf.Network, C_tx: float, C_rx: float) -> rf.Network:
        """Add shunt capacitors to network ports.
        
        Args:
            ntwk: Network to modify
            C_tx: Capacitance for input ports
            C_rx: Capacitance for output ports
            
        Returns:
            Network with added capacitance
            
        Raises:
            RuntimeError: If network is not 2n-port
        """
        freq = ntwk.frequency
        n_port = ntwk.number_of_ports
        if n_port % 2 != 0:
            raise RuntimeError('Network should be 2n-port.')

        # Calculate capacitor admittance
        omega = 2 * np.pi * freq.f
        y_tx = 1j * omega * C_tx
        y_rx = 1j * omega * C_rx

        # Create arrays for computation
        y_tx = y_tx[:, np.newaxis, np.newaxis]
        y_rx = y_rx[:, np.newaxis, np.newaxis]
        eye_array = np.tile(np.eye(n_port, dtype=np.complex128), (len(freq.f), 1, 1))
        
        # Create masks for input and output ports
        top_left_mask = np.flip(np.triu(np.ones((n_port, n_port)), k=0), axis=1)[np.newaxis, :, :]
        bottom_right_mask = np.flip(np.tril(np.ones((n_port, n_port)), k=0), axis=1)[np.newaxis, :, :]
        
        # Combine masks with admittances
        cap_list = eye_array * (top_left_mask * y_tx + bottom_right_mask * y_rx)

        # Convert and add capacitance
        y_array = s2y(ntwk.s, ntwk.z0)
        total_admittance = cap_list + y_array
        s_new = y2s(total_admittance, ntwk.z0)

        # Create new network
        return rf.Network(frequency=freq, s=s_new, z0=ntwk.z0)

    def add_inductance(self, ntwk: rf.Network, L_tx: float, L_rx: float) -> rf.Network:
        """Add series inductors to network ports based on transmission direction.
        
        This method uses a vectorized approach to add inductance to the correct
        ports based on the `directions` parameter, avoiding loops for efficiency
        and clarity.

        Args:
            ntwk (rf.Network): Network to modify.
            L_tx (float): Inductance for the transmit-side port of a line.
            L_rx (float): Inductance for the receive-side port of a line.
            
        Returns:
            rf.Network: Network with added inductance, respecting port directions.
        """
        freq = ntwk.frequency
        num_ports = ntwk.nports
        num_lines = num_ports // 2

        # Ensure directions is a numpy array, defaulting to all '1's (forward).
        directions = np.array(self.params.directions if self.params.directions is not None else [1] * num_lines)

        # Calculate inductive impedance, shaped for broadcasting.
        omega = 2 * np.pi * freq.f
        zL_tx = (1j * omega * L_tx)[:, np.newaxis]
        zL_rx = (1j * omega * L_rx)[:, np.newaxis]

        # Convert network to impedance parameters for modification.
        z_array = s2z(ntwk.s, ntwk.z0)

        # --- Vectorized application of inductance ---
        # 1. Create boolean masks to identify forward and flipped lines.
        dir_is_forward = (directions == 1)
        dir_is_flipped = (directions == 0)

        # 2. Get the indices for the first half (p1) and second half (p2) of ports.
        p1_indices = np.arange(num_lines)
        p2_indices = np.arange(num_lines, num_ports)

        # 3. Apply inductance to diagonal impedance elements using the masks.
        # For forward lines, p1 is TX and p2 is RX.
        z_array[:, p1_indices[dir_is_forward], p1_indices[dir_is_forward]] += zL_tx
        z_array[:, p2_indices[dir_is_forward], p2_indices[dir_is_forward]] += zL_rx

        # For flipped lines, p1 is RX and p2 is TX.
        z_array[:, p1_indices[dir_is_flipped], p1_indices[dir_is_flipped]] += zL_rx
        z_array[:, p2_indices[dir_is_flipped], p2_indices[dir_is_flipped]] += zL_tx
        
        # Convert back to S-parameters.
        s_new = z2s(z_array, ntwk.z0)

        # Create a new network with the updated S-parameters.
        return rf.Network(frequency=freq, s=s_new, z0=ntwk.z0)

    def add_ctle(self, ntwk: rf.Network, DC_gain: float, AC_gain: float, fp1: float, fp2: float, eps: float = 0) -> rf.Network:
        """Add Continuous-Time Linear Equalization (CTLE) to network.
        
        Args:
            ntwk: Network to modify
            DC_gain: DC gain of the CTLE
            AC_gain: AC gain of the CTLE
            fp1: First pole frequency
            fp2: Second pole frequency
            eps: Small value for numerical stability
            
        Returns:
            Network with CTLE applied
        """
        # Define CTLE transfer function
        num = [AC_gain * fp2, DC_gain * fp1 * fp2]
        den = [1, fp1 + fp2, fp1 * fp2]
        system = scipy.signal.TransferFunction(num, den)

        # Calculate frequency response
        _, TF = scipy.signal.freqs(system.num, system.den, worN=ntwk.f)
        TF = TF[:, None]

        # Create port index arrays
        n = ntwk.nports
        inp, out = np.indices((n, n))

        # Define masks for different network components
        il_pos = (out == inp + n//2)  # Insertion loss positive
        il_neg = (out == inp - n//2)  # Insertion loss negative
        rl_one = (out == inp) & (inp >= n//2)  # Return loss
        fext_mul = (out != inp + n//2) & (inp < n//2) & (out >= n//2)  # Far-end crosstalk multiply
        fext_zero = (out != inp + n//2) & (inp >= n//2) & (out < n//2)  # Far-end crosstalk zero
        next_zero = (out != inp) & (inp > n//2) & (out >= n//2)  # Near-end crosstalk zero

        # Apply masks to S-parameters
        ntwk.s[:, out[il_pos], inp[il_pos]] *= TF
        ntwk.s[:, out[il_neg], inp[il_neg]] = eps
        ntwk.s[:, out[rl_one], inp[rl_one]] = -1
        ntwk.s[:, out[fext_mul], inp[fext_mul]] *= TF
        ntwk.s[:, out[fext_zero], inp[fext_zero]] = eps
        ntwk.s[:, out[next_zero], inp[next_zero]] = eps

        return ntwk

    # ===============================================
    # NETWORK PROCESSING METHODS
    # ===============================================
    
    def get_network_with_dc(self, ntwk):
        """Return the input 2n-port network with DC data, or create one if it doesn't exist."""
        n_port = ntwk.number_of_ports
        nLine = n_port // 2

        if ntwk.f[0] == 0:
            return ntwk

        f_new = np.concatenate(([0], ntwk.f))
        z0_new = np.vstack((ntwk.z0[0, :], ntwk.z0))
        s_new = np.vstack((np.eye(n_port)[np.newaxis, :, :], ntwk.s))

        return rf.Network(f=f_new, f_unit='hz', s=s_new, z0=z0_new)

    def trunc_network_frequency(self, ntwk, f_upper):
        """Truncate a network frequency response to a specified upper frequency."""
        if ntwk.f[-1] <= f_upper:
            return ntwk
        if ntwk.f[0] > f_upper:
            raise RuntimeError('f_upper too low. No frequency value is at most f_upper.')

        ind_f_upper = np.searchsorted(ntwk.f, f_upper)
        ntwk_lf = rf.Network(f=ntwk.f[:ind_f_upper], f_unit='hz', s=ntwk.s[:ind_f_upper, :, :], z0=ntwk.z0[:ind_f_upper, :])
        return ntwk_lf

    def renormalize_network(self, network, z_new):
        """Renormalize network to new port impedances."""
        # Expand z_new to match the shape of z0
        z_new_expanded = np.tile(z_new, (len(network.f), 1))

        # Convert S-parameters to Z-parameters
        z_array = s2z(network.s, network.z0)

        # Convert Z-parameters back to S-parameters with new impedance
        s_new = z2s(z_array, z_new_expanded)

        # Create a new network with the renormalized S-parameters
        new_network = rf.Network(frequency=network.frequency, s=s_new, z0=z_new)
        return new_network

    def renorm(self, ntwk, R_tx, R_rx):
        """Renormalize network for TX and RX impedances."""
        n_port = ntwk.number_of_ports
        if n_port % 2 != 0:
            raise RuntimeError('Network should be 2n-port.')
        z = np.array([R_tx] * (n_port//2) + [R_rx] * (n_port//2))
        ntwk = self.renormalize_network(ntwk, z)
        return ntwk

    # ===============================================
    # SINGLE BIT RESPONSE METHODS
    # ===============================================

    def get_line_sbr(self, ntwk):
        """Calculate single bit response using self attributes."""
        n_port = ntwk.number_of_ports
        n_line = n_port // 2

        R_tx = ntwk.z0[0, 0].real
        R_rx = ntwk.z0[0, n_line].real

        # Create frequency axis directly from time axis
        time_axis = self.params.time_axis_sbr
        if not (time_axis.start == 0 and time_axis.endpoint == True):
            raise RuntimeError('time_axis.start == 0 and time_axis.endpoint == True should be true.')
        if time_axis.num % 2 != 1:
            raise RuntimeError('time_axis.num should be an odd number.')
        
        f_num = (time_axis.num + 1) // 2
        f_step = 1 / (time_axis.num * time_axis.step)
        f_stop = (f_num - 1) * f_step
        f_ax_sp_new = Axis(start=0, stop=f_stop, num=f_num, endpoint=True)

        # single bit pulse in frequency domain
        pulse_f_volt_at_source = self.params.pulse.freq_dom(f_ax_sp_new.array)
        pulse_f_wave_at_port = pulse_f_volt_at_source / (2 * np.sqrt(R_tx))

        # --- Fully vectorized S-parameter interpolation ---
        new_freq_array = f_ax_sp_new.array
        s_data_all = ntwk.s[:, n_line:, :n_line]  # Shape: [nfreq, n_line, n_line]
        
        magnitude_all = np.abs(s_data_all)
        phase_all = np.unwrap(np.angle(s_data_all), axis=0)

        mag_reshaped = magnitude_all.reshape(ntwk.f.size, -1)
        phase_reshaped = phase_all.reshape(ntwk.f.size, -1)

        magnitude_interp_flat = np.array([np.interp(new_freq_array, ntwk.f, mag_reshaped[:, i], left=0, right=0) for i in range(mag_reshaped.shape[1])]).T
        phase_interp_flat = np.array([np.interp(new_freq_array, ntwk.f, phase_reshaped[:, i], left=0, right=0) for i in range(phase_reshaped.shape[1])]).T
        
        s_param_interp = (magnitude_interp_flat * np.exp(1j * phase_interp_flat)).reshape(f_ax_sp_new.num, n_line, n_line)
        
        # Vectorized operation to obtain SBR in frequency domain
        pulse_f_wave_at_port_bcast = pulse_f_wave_at_port[:, np.newaxis, np.newaxis]
        sbr_f_dom = pulse_f_wave_at_port_bcast * s_param_interp * np.sqrt(R_rx)

        # --- Vectorized inverse continuous FT ---
        M = sbr_f_dom.shape[0]
        sbr_f_flat = sbr_f_dom.reshape(M, -1)
        
        f_data_ext = np.zeros((2 * M - 1, sbr_f_flat.shape[1]), dtype=np.cdouble)
        f_data_ext[:M, :] = sbr_f_flat
        f_data_ext[M:, :] = np.conj(np.flip(sbr_f_flat, axis=0))[:M - 1, :]
        
        sbr_t_flat = np.real((2 * f_ax_sp_new.num - 1) * f_ax_sp_new.step * np.fft.ifft(f_data_ext, axis=0))
        sbr_t_dom = sbr_t_flat.reshape((2 * M - 1, n_line, n_line))

        line_sbrs = []
        for idx_line in range(sbr_t_dom.shape[1]):
            sbr_one_line = sbr_t_dom[:, idx_line, :]
            time_vec_reshape = np.reshape(self.params.time_axis_sbr.array, (-1, 1))
            time_sbr = np.hstack((time_vec_reshape, sbr_one_line))
            line_sbrs.append(time_sbr)
        return line_sbrs

    def sparam_to_pulse(self, ntwk):
        """Convert S-parameters to pulse responses using self attributes."""
        ntwk = self.get_network_with_dc(ntwk)
        ntwk = self.trunc_network_frequency(ntwk, self.params.f_trunc)
        ntwk.z0 = self.params.snp_path_z0

        # add shunt C
        ntwk = self.add_capacitance(ntwk, self.params.C_tx, self.params.C_rx)

        # add ctle
        if self.params.DC_gain is not None and not np.isnan(self.params.DC_gain):
            ntwk = self.add_ctle(ntwk, self.params.DC_gain, self.params.AC_gain, self.params.fp1, self.params.fp2)

        # renormalize
        ntwk = self.renorm(ntwk, self.params.R_tx, self.params.R_rx)

        # get single bit response of each line
        line_sbrs = self.get_line_sbr(ntwk)

        return line_sbrs

    # ===============================================
    # PDA PROCESSING METHODS
    # ===============================================

    def prepare_subtracted_raw_data(self, line_sbrs, line_idx):
        """Prepare raw data from single bit responses with improved naming."""
        raw_data = line_sbrs[line_idx]
        time_step_raw = raw_data[1, 0] - raw_data[0, 0]

        if raw_data[0, 0] != 0:
            raise RuntimeError('start time of SBR raw data should be zero.')
        if time_step_raw <= 0:
            raise RuntimeError('time step of SBR raw data should be positive.')

        # Extract all columns except the first (time column)
        raw_matrix = raw_data[:, 1:].copy()

        return raw_matrix, time_step_raw

    def interpolate(self, t_step_raw, t_step_interpolated, num_xtalk, raw_matrix):
        """Interpolate raw data to the desired time step using fully vectorized operations."""
        num_rows_raw = raw_matrix.shape[0]
        t_stop_raw = t_step_raw * (num_rows_raw - 1)
        num_rows_interpolated = math.floor(t_stop_raw / t_step_interpolated) + 1
        t_stop_interpolated = t_step_interpolated * (num_rows_interpolated - 1)

        # Create time vectors
        t_samples_raw = np.linspace(start=0, stop=t_stop_raw, num=num_rows_raw, endpoint=True)
        t_samples_interpolated = np.linspace(start=0, stop=t_stop_interpolated, num=num_rows_interpolated, endpoint=True)
        
        # Vectorized interpolation using broadcasting - process all columns at once
        # Use numpy's interp which is optimized for this exact use case
        interpolated_matrix = np.column_stack([
            np.interp(t_samples_interpolated, t_samples_raw, raw_matrix[:, i])
            for i in range(num_xtalk + 1)
        ])

        return num_rows_interpolated, interpolated_matrix

    def find_main_cursor(self, interpolated_matrix, num_rows_interpolated, target_line_idx):
        """Find the main cursor in interpolated data using vectorized operations."""
        # Extract the signal of interest
        signal = interpolated_matrix[:, target_line_idx].reshape(num_rows_interpolated, 1)
        
        # Find the maximum value position
        max_signal_idx = np.argmax(signal)
        cursor_idx = max_signal_idx
        
        # Use vectorized comparison to check condition while ensuring we don't go out of bounds
        while cursor_idx < num_rows_interpolated - self.params.n_perUI_intrp and cursor_idx >= self.params.n_perUI_intrp and signal[cursor_idx, 0] > signal[cursor_idx - self.params.n_perUI_intrp, 0]:
            cursor_idx += 1

        return cursor_idx

    def rshift_zeropad(self, cursor_idx, num_rows_interpolated, num_xtalk, interpolated_matrix):
        """Shift and zero-pad the data using optimized operations."""
        # Calculate shift parameters
        right_shift_len = self.params.n_perUI_intrp - (cursor_idx % self.params.n_perUI_intrp)
        pad_left = right_shift_len
        pad_right = self.params.n_perUI_intrp - ((pad_left + num_rows_interpolated) % self.params.n_perUI_intrp)
        num_rows_shifted = pad_left + num_rows_interpolated + pad_right

        # Create and fill shifted matrix in one operation
        shifted_matrix = np.zeros((num_rows_shifted, num_xtalk + 1))
        shifted_matrix[pad_left:pad_left + num_rows_interpolated, :] = interpolated_matrix

        return shifted_matrix

    def process_pulse_responses(self, line_sbrs):
        """Convert pulse responses to response matrices for waveform calculation."""
        num_lines = len(line_sbrs)
        num_xtalk = num_lines - 1
        samples_per_ui = self.params.n_perUI_intrp
        
        # First pass: calculate dimensions for all lines
        ui_counts = np.zeros(num_lines, dtype=int)
        cursor_indices = np.zeros(num_lines, dtype=int)
        interpolated_data = []
        
        for line_idx in range(num_lines):
            # Extract and interpolate raw data
            raw_matrix, time_step_raw = self.prepare_subtracted_raw_data(line_sbrs, line_idx)
            num_rows_interpolated, interpolated_matrix = self.interpolate(
                time_step_raw, self.params.t_step_intrp, num_xtalk, raw_matrix)
            
            # Find cursor and calculate dimensions
            cursor_indices[line_idx] = self.find_main_cursor(
                interpolated_matrix, num_rows_interpolated, line_idx)
            
            # Calculate UI count
            right_shift_len = samples_per_ui - (cursor_indices[line_idx] % samples_per_ui)
            pad_left = right_shift_len
            pad_right = samples_per_ui - ((pad_left + num_rows_interpolated) % samples_per_ui)
            num_rows_shifted = pad_left + num_rows_interpolated + pad_right
            ui_counts[line_idx] = num_rows_shifted // samples_per_ui
            
            # Store for second pass
            interpolated_data.append((num_rows_interpolated, interpolated_matrix))
        
        # Allocate final response matrix with maximum size
        max_ui_count = np.max(ui_counts)
        response_matrices = np.zeros((num_lines, num_xtalk + 1, max_ui_count, samples_per_ui))
        
        # Second pass: fill the response matrices
        half_steady = None
        for line_idx in range(num_lines):
            num_rows_interpolated, interpolated_matrix = interpolated_data[line_idx]
            
            # Shift and zero-pad
            shifted_matrix = self.rshift_zeropad(
                cursor_indices[line_idx], num_rows_interpolated, num_xtalk, interpolated_matrix)
            
            # Reshape and store directly into final array
            response_matrix = np.transpose(shifted_matrix).reshape(
                num_xtalk + 1, ui_counts[line_idx], samples_per_ui)
            response_matrices[line_idx, :, :ui_counts[line_idx], :] = response_matrix
            
            # Calculate half_steady from the first line
            if line_idx == 0:
                main_signal = response_matrix[0, :, :]
                half_steady = np.mean(np.sum(main_signal, axis=0)) / 2
        
        return half_steady, response_matrices

    # ===============================================
    # EYE WIDTH CALCULATION METHODS
    # ===============================================

    def calculate_eye_width(self, eye_index):
        """Vectorized eye width calculation using numpy operations."""
        if not np.any(eye_index):
            return -1
        
        # Find run lengths of True values using numpy operations
        # Add False at beginning and end to handle edge cases
        padded = np.concatenate(([False], eye_index, [False]))
        
        # Find transitions: 1 for False->True, -1 for True->False
        diff = np.diff(padded.astype(int))
        run_starts = np.where(diff == 1)[0]
        run_ends = np.where(diff == -1)[0]
        
        if len(run_starts) == 0:
            return -1
        
        # Calculate run lengths and return max - 1
        run_lengths = run_ends - run_starts
        return np.max(run_lengths) - 1

    def calculate_eyewidth_percentage(self, half_steady, waveform, vref_num=None):
        """
        Calculate eye width percentage using an adaptive reference voltage search.
        
        This implementation matches the legacy behavior exactly while using more
        readable constants and variable names.
        
        Args:
            half_steady: Half of the steady-state voltage level
            waveform: 3D waveform array [samples_per_ui, conv_length, num_lines]
            vref_num: Number of reference voltages to test (default: 11)
            
        Returns:
            Array of eye width percentages for each line
        """
        # Constants for voltage reference search
        VREF_NUM_DEFAULT = 11
        VREF_STEP = 0.005  # 5mV step size
        VREF_SEARCH_RANGE_STEPS = 5  # ±5 steps around center
        EDGE_BOUNDARY_INDEX = 10  # Hard-coded boundary check (legacy behavior)
        MAX_SEARCH_ITERATIONS = 100  # Safety limit for extended search
        
        if vref_num is None:
            vref_num = VREF_NUM_DEFAULT

        def calculate_eye_width_for_vref(vref):
            """Calculate eye width for all lines at a given reference voltage."""
            vih = vref + self.params.vmask  # Upper voltage threshold
            vil = vref - self.params.vmask  # Lower voltage threshold
            eyewidths = np.zeros(self.num_lines, dtype=int)
            
            # Check which waveform points are outside the eye mask
            valid_eye_points = (waveform > vih) | (waveform < vil)
            eye_indices_per_line = np.all(valid_eye_points, axis=1)
            
            for line_idx in range(self.num_lines):
                eyewidths[line_idx] = self.calculate_eye_width(eye_indices_per_line[:, line_idx])
            return eyewidths

        # Calculate initial reference voltage search range
        vref_center = VREF_STEP * round(half_steady / VREF_STEP)
        vref_range = VREF_SEARCH_RANGE_STEPS * VREF_STEP
        vref_candidates = np.linspace(
            start=vref_center - vref_range,
            stop=vref_center + vref_range,
            num=vref_num,
            endpoint=True,
        )

        # Test all candidate reference voltages
        eye_width_results = np.array([calculate_eye_width_for_vref(vref) for vref in vref_candidates])
        min_eye_widths = np.min(eye_width_results, axis=1)
        best_vref_idx = np.argmax(min_eye_widths)
        
        # Determine final eye width based on where the optimum was found
        if min_eye_widths[best_vref_idx] == -1:
            # No valid eye found in search range - use default reference voltage
            final_eyewidth = calculate_eye_width_for_vref(self.params.vref_dfl)
        elif 0 < best_vref_idx < EDGE_BOUNDARY_INDEX:
            # Optimum found in the middle of search range - use it directly
            final_eyewidth = eye_width_results[best_vref_idx]
        else:
            # Optimum at edge of search range - extend search in that direction
            search_direction = 1 if best_vref_idx == EDGE_BOUNDARY_INDEX else -1
            current_vref = vref_candidates[best_vref_idx]
            found_optimum = False
            iteration_count = 0
            
            while not found_optimum:
                if iteration_count > MAX_SEARCH_ITERATIONS:
                    raise RuntimeError("Cannot find optimal reference voltage")
                
                iteration_count += 1
                next_vref = current_vref + search_direction * VREF_STEP
                next_eye_width = calculate_eye_width_for_vref(next_vref)
                
                if np.min(next_eye_width) > np.min(eye_width_results[best_vref_idx]):
                    # Found better result - stop here and use previous best
                    found_optimum = True
                    final_eyewidth = eye_width_results[best_vref_idx]
                else:
                    # Continue searching - update current position
                    current_vref = next_vref
                    eye_width_results[best_vref_idx] = next_eye_width
            
            final_eyewidth = eye_width_results[best_vref_idx]
        
        # Convert to percentage
        return 100 * final_eyewidth / self.params.n_perUI_intrp

    def calculate_waveform(self, test_patterns, response_matrices):
        """
        Waveform calculation implemented to be bit-exact with the legacy version.
        This version uses np.convolve and matches the legacy summation order to
        ensure identical output, replacing the faster but less precise FFT-based method.
        """
        pattern_length = test_patterns.shape[2]
        response_length = response_matrices.shape[2]
        conv_length = pattern_length + response_length - 1
        
        # Pre-allocate output waveform array
        waveform = np.zeros((self.params.n_perUI_intrp, conv_length, self.num_lines))
        
        # Process each output line
        for output_line_idx in range(self.num_lines):
            # Get patterns and responses for the current output line
            patterns_for_output = test_patterns[output_line_idx]
            
            # Transpose responses to [samples_per_ui, num_lines, response_length]
            # to match the legacy loop structure for convolution.
            responses_for_output = np.transpose(response_matrices[output_line_idx], (2, 0, 1))

            # This array stores all convolution results for the current output line before summing,
            # exactly matching the legacy implementation's approach to ensure bit-wise equality.
            # Shape: [samples_per_ui, conv_length, num_input_lines]
            waves_to_add = np.zeros((self.params.n_perUI_intrp, conv_length, self.num_lines))

            # Perform convolution for each sample in UI and each input line
            for pt in range(self.params.n_perUI_intrp):
                for input_line_idx in range(self.num_lines):
                    pattern = patterns_for_output[input_line_idx]
                    response = responses_for_output[pt, input_line_idx]
                    waves_to_add[pt, :, input_line_idx] = np.convolve(pattern, response)
            
            # Sum over the input lines axis after all convolutions are done, matching legacy logic.
            waveform[:, :, output_line_idx] = np.sum(waves_to_add, axis=2)
        
        return waveform

    def calculate_eyewidth(self):
        """Eye width calculation with adaptive strategies based on problem size."""
        try:
            line_sbrs = self.sparam_to_pulse(self.ntwk)
            half_steady, response_matrices = self.process_pulse_responses(line_sbrs)
            test_patterns = generate_test_pattern(self.num_lines)
            
            # Calculate waveform and eye width
            wave = self.calculate_waveform(test_patterns, response_matrices)
            eyewidth_pct = self.calculate_eyewidth_percentage(half_steady, wave)
            
            return eyewidth_pct
                
        except Exception as e:
            print(f"Error in calculate_eyewidth: {e}")
            print(traceback.format_exc())
            raise e

    def _load_and_cascade_networks(self):
        """
        Load S-parameter files and cascade them.
        Handles both file paths and pre-loaded skrf.Network objects.
        """
        # Load networks, checking if they are already network objects
        ntwk_horiz = self.params.snp_horiz if isinstance(self.params.snp_horiz, rf.Network) else rf.Network(self.params.snp_horiz)
        
        if self.params.snp_tx and self.params.snp_rx:
            ntwk_tx = self.params.snp_tx if isinstance(self.params.snp_tx, rf.Network) else rf.Network(self.params.snp_tx)
            ntwk_rx = self.params.snp_rx if isinstance(self.params.snp_rx, rf.Network) else rf.Network(self.params.snp_rx)
            
            # Apply inductance before cascading
            ntwk_horiz = self.add_inductance(ntwk_horiz, self.params.L_tx, self.params.L_rx)
            
            # Flip RX network for cascading
            ntwk_rx.flip()
            
            # Cascade the networks
            ntwk = ntwk_tx ** ntwk_horiz ** ntwk_rx
        else:
            # Apply inductance directly if no vertical networks
            ntwk = self.add_inductance(ntwk_horiz, self.params.L_tx, self.params.L_rx)
            
        return ntwk

# ===============================================
# MODULE-LEVEL FUNCTIONS (PUBLIC INTERFACE)
# ===============================================

def snp_eyewidth_simulation(config, snp_files=None, directions=None):
    """
    Main function for eye width simulation that provides backward compatibility.
    
    Args:
        config: Configuration object or dictionary
        snp_files: Optional S-parameter files (for backward compatibility)
        directions: Optional directions array (for backward compatibility)
    
    Returns:
        Tuple of (eye_widths, directions)
    """
    simulator = EyeWidthSimulator(config, snp_files, directions)
    return simulator.calculate_eyewidth()


def main():
    try:
        from bound_param import SampleResult
    except ImportError:
        print("Error: Could not import SampleResult from bound_param")
        raise
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    possible_data_dirs = [
        os.path.join(project_root, "test_data"), "test_data", "../../test_data", "../test_data"
    ]
    data_dir = None
    for test_dir in possible_data_dirs:
        if os.path.exists(test_dir) and os.path.isdir(test_dir):
            data_dir = test_dir
            break
    if data_dir is None:
        data_dir = "../../test_data"
    
    snp_horiz = os.path.join(data_dir, "tlines4_seed0.s8p")
    snp_tx = os.path.join(data_dir, "tlines4_seed1.s8p")
    snp_rx = os.path.join(data_dir, "tlines4_seed2.s8p")
    
    config_dict = {
        "R_tx": 10, "R_rx": 1.0e9, "C_tx": 1e-13, "C_rx": 1e-13, "L_tx": 1e-10, "L_rx": 1e-10,
        "pulse_amplitude": 0.4, "bits_per_sec": 6.4e9, "vmask": 0.04,
        "snp_horiz": snp_horiz, "snp_tx": snp_tx, "snp_rx": snp_rx,
        "directions": [1] * 1 + [0] * 1 + [1] * 1 + [0] * 1
    }
    
    config = SampleResult(**config_dict)
    simulator = EyeWidthSimulator(config)
    
    # Run optimized simulation
    result = simulator.calculate_eyewidth()
    print(f"Eye width results: {result}")
    return result


if __name__ == "__main__":
    main() 