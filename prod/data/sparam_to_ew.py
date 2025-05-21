import os
import time
import math
import json
import scipy
import torch
import argparse
import traceback
import skrf as rf
import numpy as np
from functools import partial
from dataclasses import dataclass, field, fields

from misc import *

def get_network_with_dc(ntwk):
    """Return the input 2n-port network with DC data, or create one if it doesn't exist."""
    n_port = ntwk.number_of_ports
    nLine = n_port // 2

    if ntwk.f[0] == 0:
        return ntwk

    f_new = np.concatenate(([0], ntwk.f))
    z0_new = np.vstack((ntwk.z0[0, :], ntwk.z0))
    s_new = np.vstack((np.eye(n_port)[np.newaxis, :, :], ntwk.s))

    return rf.Network(f=f_new, f_unit='hz', s=s_new, z0=z0_new)

def trunc_network_frequency(ntwk, f_upper):
    """Truncate a network frequency response to a specified upper frequency."""
    if ntwk.f[-1] <= f_upper:
        return ntwk
    if ntwk.f[0] > f_upper:
        raise RuntimeError('f_upper too low. No frequency value is at most f_upper.')

    ind_f_upper = np.searchsorted(ntwk.f, f_upper)
    ntwk_lf = rf.Network(f=ntwk.f[:ind_f_upper], f_unit='hz', s=ntwk.s[:ind_f_upper, :, :], z0=ntwk.z0[:ind_f_upper, :])
    return ntwk_lf

def renormalize_network_cuda(network, z_new, device='cuda'):
    # Convert S-parameters and impedances to PyTorch tensors and move to GPU
    s_torch = torch.from_numpy(network.s).to(device)
    z0_torch = torch.from_numpy(network.z0).to(device)
    z_new_torch = torch.from_numpy(z_new).to(s_torch.dtype).to(device)

    # Expand Z new to match the shape of Z0
    z_new_torch = z_new_torch.unsqueeze(0).expand_as(z0_torch)

    # Convert S-parameters to Z-parameters
    z_torch = s2z_torch(s_torch, z0_torch)

    # Convert Z-parameters back to S-parameters with new impedance
    s_new_torch = z2s_torch(z_torch, z_new_torch)

    # Move the result back to CPU and convert to numpy array
    if s_new_torch.is_cuda:
        s_new = s_new_torch.cpu().numpy()
    else:
        s_new = s_new_torch.resolve_conj().numpy()

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Create a new network with the renormalized S-parameters
    new_network = rf.Network(frequency=network.frequency, s=s_new, z0=z_new)
    return new_network

def renorm(ntwk, R_tx, R_rx, device='cuda'):
    """
    Renormalize network such that each port of the first half of the ports of the network has port impedance R_tx
    and each port of the second half of the ports has port impedance R_rx.
    """
    n_port = ntwk.number_of_ports
    if n_port % 2 != 0:
        raise RuntimeError('Network should be 2n-port.')
    z = np.array([R_tx] * (n_port//2) + [R_rx] * (n_port//2))
    ntwk = renormalize_network_cuda(ntwk, z, device)
    return ntwk

def time_axis_to_frequency_axis(time_axis):
    """Inverse function of f2t."""
    if not (time_axis.start == 0 and time_axis.endpoint == True):
        raise RuntimeError('time_axis.start == 0 and time_axis.endpoint == True should be true.')
    if (time_axis.num - 1) % 2 != 0:
        raise RuntimeError('time_axis.num should be an odd number.')

    f_num = (time_axis.num + 1) // 2
    f_step = 1 / (time_axis.num * time_axis.step)
    f_stop = (f_num - 1) * f_step
    f_ax = Axis(start=0, stop=f_stop, num=f_num, endpoint=True)

    return f_ax

def inverse_continuous_ft(f_data, f_ax):
    """Calculate inverse continuous Fourier transform."""
    M = f_data.shape[0]
    f_data_ext = np.zeros((2 * M - 1, ), dtype=np.cdouble)
    f_data_ext[:M] = f_data
    f_data_ext[M:] = np.conj(np.flip(f_data)[:-1])
    return np.real((2 * f_ax.num - 1) * f_ax.step * np.fft.ifft(f_data_ext))

def continuous_ft(time_axis, t_data):
    """Calculate continuous Fourier transform."""
    f_ax = time_axis_to_frequency_axis(time_axis)
    ft = time_axis.step * np.fft.fft(t_data)[:f_ax.num].copy()
    return f_ax, ft

def conv(time_axis_1, t_data_1, time_axis_2, t_data_2):
    """Calculate continuous convolution."""
    if not (time_axis_1.start == 0 and time_axis_1.endpoint == True):
        raise RuntimeError('time_axis_1.start == 0 and time_axis_1.endpoint == True.')
    if not (time_axis_2.start == 0 and time_axis_2.endpoint == True):
        raise RuntimeError('time_axis_2.start == 0 and time_axis_2.endpoint == True.')
    if not (time_axis_1.step == time_axis_2.step):
        raise RuntimeError('time_axis_1.step == time_axis_2.step should be true.')

    t_data_conv = time_axis_1.step * np.convolve(t_data_1, t_data_2, mode='full')
    assert t_data_conv.shape[0] == t_data_1.shape[0] + t_data_2.shape[0] - 1
    time_axis_conv = Axis(start=0, stop=time_axis_1.step * (t_data_conv.shape[0] - 1), num=t_data_conv.shape[0], endpoint=True)

    return time_axis_conv, t_data_conv

def interp_s_param(s_data, new_freq_ax, old_freq):
    """Interpolate S-parameters with causal enforcement."""
    new_freq_array = new_freq_ax.array
    magnitude, phase = np.abs(s_data), np.unwrap(np.angle(s_data))
    magnitude_interp = np.interp(new_freq_array, old_freq, magnitude, left=0, right=0)
    phase_interp = np.interp(new_freq_array, old_freq, phase, left=0, right=0)
    return magnitude_interp * np.exp(1j * phase_interp)

def interp_s_mat(ntwk, n_line, f_ax_sp_new):
    """Interpolate 1/4 of the S-matrix with causal enforcement."""
    s_param_interp = np.zeros((f_ax_sp_new.num, n_line, n_line), dtype=np.cdouble)
    for idx_output_line, idx_input_line in np.ndindex(n_line, n_line):
        s_data = ntwk.s[:, idx_output_line, idx_input_line]
        s_param_interp[:, idx_output_line, idx_input_line] = interp_s_param(s_data, f_ax_sp_new, ntwk.f)
    return s_param_interp

def get_line_sbr(ntwk, time_axis_sbr, pulse):
    """Calculate single bit response."""
    n_port = ntwk.number_of_ports
    n_line = n_port // 2

    R_tx = ntwk.z0[0, 0].real
    R_rx = ntwk.z0[0, n_line].real

    # frequency axis of S-parameters after interpolation
    f_ax_sp_new = time_axis_to_frequency_axis(time_axis_sbr)

    # single bit pulse in frequency domain: first find of FT of pulse (voltage at source)
    pulse_f_volt_at_source = pulse.freq_dom(f_ax_sp_new.array)
    # single bit pulse in frequency domain (incident wave at port)
    pulse_f_wave_at_port = pulse_f_volt_at_source / (2 * np.sqrt(R_tx))

    # SBR in frequency and time domain
    sbr_f_dom = np.zeros((f_ax_sp_new.num, n_line, n_line), dtype=np.cdouble)
    sbr_t_dom = np.zeros((2 * f_ax_sp_new.num - 1, n_line, n_line))

    s_param_interp = interp_s_mat(ntwk, n_line, f_ax_sp_new)

    # Vectorized operation to obtain SBR in frequency domain -
    pulse_f_wave_at_port = pulse_f_wave_at_port[:, np.newaxis, np.newaxis]
    sbr_f_dom = pulse_f_wave_at_port * s_param_interp * np.sqrt(R_rx)

    sbr_t_dom = np.apply_along_axis(inverse_continuous_ft, 0, sbr_f_dom, f_ax_sp_new)

    line_sbrs = []
    for idx_line in range(sbr_t_dom.shape[1]):
        sbr_one_line = sbr_t_dom[:, idx_line, :]
        time_vec_reshape = np.reshape(time_axis_sbr.array, (-1, 1))
        time_sbr = np.hstack((time_vec_reshape, sbr_one_line))
        line_sbrs.append(time_sbr)
    return line_sbrs

def sparam_to_pulse(params, ntwk, device='cuda'):
    ntwk = get_network_with_dc(ntwk)
    ntwk = trunc_network_frequency(ntwk, params.f_trunc)
    ntwk.z0 = params.snp_path_z0

    # add shunt C
    ntwk = add_cap_to_port(ntwk, params.C_tx, params.C_rx, device)

    # add ctle
    if params.DC_gain is not None and not np.isnan(params.DC_gain):
        ntwk = add_ctle(ntwk, params.DC_gain, params.AC_gain, params.fp1, params.fp2)

    # renormalize
    ntwk = renorm(ntwk, params.R_tx, params.R_rx, device)

    # get single bit response of each line
    line_sbrs = get_line_sbr(ntwk, params.time_axis_sbr, params.pulse)

    return line_sbrs

def prepare_subtracted_raw_data(line_sbrs, idx_line):
    raw = line_sbrs[idx_line]
    t_step_raw = raw[1, 0] - raw[0, 0]

    if not raw[0, 0] == 0:
        raise RuntimeError('start time of SBR raw data should be zero.')
    if not t_step_raw > 0:
        raise RuntimeError('time step of SBR raw data should be positive.')

    Pmatrix_raw = raw[:, 1:].copy()

    return Pmatrix_raw, t_step_raw

def interpolate(t_step_raw, t_step_intrp, nXtk, Pmatrix_raw):
    nrow_raw = Pmatrix_raw.shape[0]
    t_stop_raw = t_step_raw * (nrow_raw - 1)
    nrow_intrp = math.floor(t_stop_raw / t_step_intrp) + 1
    t_stop_intrp = t_step_intrp * (nrow_intrp - 1)

    t_sample_raw = np.linspace(start=0, stop=t_stop_raw, num=nrow_raw, endpoint=True)
    t_sample_intrp = np.linspace(start=0, stop=t_stop_intrp, num=nrow_intrp, endpoint=True)
    Pmatrix_intrp = np.zeros((nrow_intrp, nXtk + 1))
    for ind_line in range(nXtk + 1):
        Pmatrix_intrp[:, ind_line] = np.interp(t_sample_intrp, t_sample_raw, Pmatrix_raw[:, ind_line])

    return nrow_intrp, Pmatrix_intrp

def find_main_cursor(Pmatrix_intrp, nrow_intrp, n_perUI_intrp, idx_line):
    ISI_intrp = Pmatrix_intrp[:, idx_line].copy().reshape((nrow_intrp, 1))
    ind_ISI_max_intrp = np.argmax(ISI_intrp)
    ind_MCr_intrp = ind_ISI_max_intrp

    ind_MCr_intrp = ind_ISI_max_intrp + n_perUI_intrp
    while ISI_intrp[ind_MCr_intrp - n_perUI_intrp, 0] >= ISI_intrp[ind_MCr_intrp, 0]:
        ind_MCr_intrp += 1

    return ind_MCr_intrp

def rshift_zeropad(n_perUI_intrp, ind_MCr_intrp, nrow_intrp, nXtk, Pmatrix_intrp):
    rshift_len = n_perUI_intrp - (ind_MCr_intrp % n_perUI_intrp)
    npad_l = rshift_len
    npad_r = n_perUI_intrp - ((npad_l + nrow_intrp) % n_perUI_intrp)
    nrow_shift = npad_l + nrow_intrp + npad_r
    nUI_shift = nrow_shift // n_perUI_intrp

    Pmatrix_shift = np.zeros((nrow_shift, nXtk + 1))
    Pmatrix_shift[npad_l:(npad_l + nrow_intrp), :] = Pmatrix_intrp[:, :]

    ind_MCr_shift = ind_MCr_intrp + rshift_len
    MCloc_shift = ind_MCr_shift // n_perUI_intrp

    return nUI_shift, Pmatrix_shift, MCloc_shift

def PDA_eye_calculation(MCloc_shift_array, Pmatrix_tprs_lines):
    nline = Pmatrix_tprs_lines.shape[0]
    pda_eye = np.zeros((2, Pmatrix_tprs_lines.shape[3], nline))

    ind_MC_tprs = MCloc_shift_array - 1
    lessthan0_tprs = (Pmatrix_tprs_lines < 0)
    lessthan0_tprs[np.arange(nline), np.arange(nline), ind_MC_tprs, :] = 0
    geq0_tprs = (Pmatrix_tprs_lines >= 0)
    geq0_tprs[np.arange(nline), np.arange(nline), ind_MC_tprs, :] = 0

    worst_val = np.diagonal(Pmatrix_tprs_lines).transpose(2, 0, 1)
    worst1_val = worst_val[np.arange(nline), ind_MC_tprs] + np.sum(lessthan0_tprs * Pmatrix_tprs_lines, axis=(1, 2))
    worst0_val = np.sum(geq0_tprs * Pmatrix_tprs_lines, axis=(1, 2))
    
    pda_eye[0] = worst1_val.T
    pda_eye[1] = worst0_val.T

    return pda_eye

def pulse_to_pda(params, line_sbrs):
    nline = len(line_sbrs)
    nXtk = nline - 1

    t_step_intrp, n_perUI_intrp = params.t_step_intrp, params.n_perUI_intrp
    MCloc_shift_array = np.zeros(nline, dtype=int)
    nUI_shift_array = np.zeros(nline, dtype=int)
    ind_MCr_intrp_array = np.zeros(nline, dtype=int)

    for idx_line in range(nline):
        Pmatrix_raw, t_step_raw = prepare_subtracted_raw_data(line_sbrs, idx_line)
        nrow_intrp, Pmatrix_intrp = interpolate(t_step_raw, t_step_intrp, nXtk, Pmatrix_raw)
        ind_MCr_intrp_array[idx_line] = find_main_cursor(Pmatrix_intrp, nrow_intrp, n_perUI_intrp, idx_line)

        nUI_shift_array[idx_line], Pmatrix_shift, MCloc_shift_array[idx_line] = rshift_zeropad(n_perUI_intrp, ind_MCr_intrp_array[idx_line], nrow_intrp, nXtk, Pmatrix_intrp)

        # response (transpose and reshaped)
        Pmatrix_tprs = np.reshape(np.transpose(Pmatrix_shift), (1 + nXtk, nUI_shift_array[idx_line], n_perUI_intrp))

        if idx_line == 0:
            nUI_shift_max = nUI_shift_array[idx_line]
            Pmatrix_tprs_lines = np.zeros((nline, 1 + nXtk, nUI_shift_array[idx_line], n_perUI_intrp))

        # if max of UI of idx lines is larger than previous ones
        if Pmatrix_tprs_lines.shape[2] < nUI_shift_array[idx_line] and idx_line > 0:
            # update back and shift
            nUI_shift_max = nUI_shift_array[idx_line]
            # create a larger array
            Pmatrix_tprs_lines_new = np.zeros((nline, 1 + nXtk, nUI_shift_array[idx_line], n_perUI_intrp))
            Pmatrix_tprs_lines_new[:, :, :Pmatrix_tprs_lines.shape[2], :] = Pmatrix_tprs_lines
            Pmatrix_tprs_lines = Pmatrix_tprs_lines_new

        assert Pmatrix_tprs_lines.shape[2] >= nUI_shift_array[idx_line] == Pmatrix_tprs.shape[1]
        Pmatrix_tprs_lines[idx_line, :, :nUI_shift_array[idx_line], :] = Pmatrix_tprs

        # half steady is for calculating vref later
        if idx_line == 0:
            line2line0_tprs = Pmatrix_tprs[0, :, :]
            half_steady = (np.mean(np.sum(Pmatrix_tprs[0, :, :], axis=0))) / 2

    assert Pmatrix_tprs_lines.shape[2] == nUI_shift_max == np.max(nUI_shift_array)

    pda_eye = PDA_eye_calculation(MCloc_shift_array, Pmatrix_tprs_lines)

    return half_steady, pda_eye, Pmatrix_tprs_lines

def eyeindex_to_eyewidth(eye_index, n_perUI_intrp):
    """Returns width, left & right boundary of eye mask."""
    max_width = 0
    max_width_l_endpt = 0
    max_width_r_endpt = 0

    l_endpt = 0
    width = 0

    for probe in range(n_perUI_intrp):
        if eye_index[probe]:
            width += 1
            if width > max_width:
                max_width = width
                max_width_l_endpt = l_endpt
                max_width_r_endpt = probe + 1
        else:
            width = 0
            l_endpt = probe + 1

    return max_width - 1

def get_vref_eyewidth(params, half_steady, pda_eye, vref_num=11):
    """Calculate vref and eyewidth."""
    def calculate_eye_width(vref):
        """
        Calculates eye width, left & right boundary of eye mask for a given output line.
        """
        vih = vref + vmask  # upper boundary of mask
        vil = vref - vmask  # lower boundary of mask

        eyewidth = np.zeros((nline, ), dtype=int)
        for i in range(nline):
            eye_index = np.all(((pda_eye[:, :, i] > vih) | (pda_eye[:, :, i] < vil)), axis=1)
            eyewidth[i] = eyeindex_to_eyewidth(eye_index, n_perUI_intrp)

        return eyewidth

    vmask, n_perUI_intrp, vref_dfl = params.vmask, params.n_perUI_intrp, params.vref_dfl
    nline = pda_eye.shape[2]

    vref_test_center = 0.005 * round(half_steady / 0.005)
    vref_test = np.linspace(start=vref_test_center - 5 * 0.005, stop=vref_test_center + 5 * 0.005, num=vref_num, endpoint=True)

    eye_width_test = np.zeros((vref_num, nline), dtype=int)
    for ind_v in range(vref_num):
        eye_width_test[ind_v] = calculate_eye_width(vref_test[ind_v])

    argmax_idx_v = np.argmax(np.min(eye_width_test, axis=1))
    if np.min(eye_width_test, axis=1)[argmax_idx_v] == -1:
        vref = vref_dfl
        eyewidth = calculate_eye_width(vref)
    elif 0 <= argmax_idx_v < vref_num:
        vref, eyewidth = vref_test[argmax_idx_v], eye_width_test[argmax_idx_v]
    else:
        p_or_n = 1 if argmax_idx_v <= 10 else -1
        vref_old = vref_test[argmax_idx_v]
        found_vref = False
        count_while_loop = 0
        while not found_vref:
            if count_while_loop > 10000:
                raise RuntimeError('Cannot find vref')
            count_while_loop += 1
            vref_new = vref_old + p_or_n * 0.005
            eye_width_new = calculate_eye_width(vref_new)
            if np.min(eye_width_new) >= np.min(eye_width_test[argmax_idx_v]):
                found_vref = True
                vref, eyewidth = vref_old, eye_width_test[argmax_idx_v]
            else:
                vref_old = vref_new
                eye_width_test[argmax_idx_v] = eye_width_new

    return 100 * eyewidth / n_perUI_intrp

def generate_pattern(nline):
    inv = lambda x: -x + 1

    one1_9b = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
    one11_10b = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    three1_6b = np.array([1, 0, 1, 0, 1, 0])
    three11_12b = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    one11_18b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    five1_9b = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])
    five11_18b = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

    WSI_V_ODD = np.hstack((one1_9b, one11_10b, three1_6b, three11_12b, inv(one11_18b), inv(one11_10b)))
    WSI_A_ODD = inv(WSI_V_ODD)
    WSI_V_ISI = WSI_V_ODD
    WSI_A_ISI = np.zeros_like(WSI_V_ISI)
    WSI_V_XTALK = np.hstack((one1_9b, one11_18b, three1_6b, three11_12b, inv(one1_9b), inv(one11_18b)))
    WSI_A_XTALK = np.hstack((five1_9b, five11_18b, inv(three1_6b), inv(three11_12b), inv(five1_9b), inv(five11_18b)))

    WSI_V = np.hstack((WSI_V_ODD, WSI_V_ISI, WSI_V_XTALK))
    WSI_A = np.hstack((WSI_A_ODD, WSI_A_ISI, WSI_A_XTALK))
    SSO = np.hstack((inv(three1_6b), inv(three1_6b), inv(three11_12b), inv(three11_12b)))

    DOQ_pattern = np.hstack((SSO, WSI_V, np.tile(WSI_A, 8), SSO))
    test_pattern = np.zeros((nline, DOQ_pattern.size))

    for i in range(nline):
        remainder = i % 9
        test_pattern[i, :] = np.hstack((SSO, np.tile(WSI_A, remainder), WSI_V, np.tile(WSI_A, 8 - remainder), SSO))

    return np.broadcast_to(test_pattern, (nline, nline, DOQ_pattern.size))

def get_waveform(test_patt_lines, n_perUI_intrp, Pmatrix_tprs_lines, nLine):
    for ind_output_line in range(nLine):
        # waves to be added together
        waves_to_add = np.zeros((n_perUI_intrp,
                                 np.convolve(test_patt_lines[ind_output_line, 0, :], Pmatrix_tprs_lines[ind_output_line, 0, :, 0]).shape[0],
                                 nLine))

        for pt in range(n_perUI_intrp):
            for ind_input_line in range(nLine):
                waves_to_add[pt, :, ind_input_line] = np.convolve(test_patt_lines[ind_output_line, ind_input_line, :], Pmatrix_tprs_lines[ind_output_line, ind_input_line, :, pt])

        # waveform of eye diagram (0th axis: index of position in UI, 1st axis: index of UI, 2nd axis: index of line)
        if ind_output_line == 0:
            wave = np.zeros((n_perUI_intrp,
                             np.convolve(test_patt_lines[0, 0, :], Pmatrix_tprs_lines[ind_output_line, 0, :, 0]).shape[0],
                             nLine))
        wave[:, :, ind_output_line] = np.sum(waves_to_add, axis=2)

    return wave

def get_waveform2(test_patt_lines, Pmatrix_tprs_lines):
    nline, _, UIs, n_perUI_intrp = Pmatrix_tprs_lines.shape

    # test_patt_lines = torch.from_numpy(test_patt_lines)
    # Pmatrix_tprs_lines = torch.from_numpy(Pmatrix_tprs_lines)

    # # wave = []
    # # for (test_patt, Pmatrix_tprs) in zip(test_patt_lines, Pmatrix_tprs_lines):
    # #     test_patt = test_patt.unsqueeze(0).unsqueeze(1)
    #     Pmatrix_tprs = Pmatrix_tprs.permute(2, 0, 1).unsqueeze(1)
    #     convolved = torch.nn.functional.conv2d(test_patt, Pmatrix_tprs, padding=(0, UIs - 1)) 
    #     wave.append(convolved.squeeze(0))
    # wave = torch.stack(wave).permute(1, 2, 0)

    test_patt_lines = torch.from_numpy(test_patt_lines).reshape(nline ** 2, -1)
    Pmatrix_tprs_lines = torch.from_numpy(Pmatrix_tprs_lines).reshape(nline ** 2, UIs, n_perUI_intrp)

    wave = []
    for (test_patt, Pmatrix_tprs) in zip(test_patt_lines, Pmatrix_tprs_lines):
        test_patt = test_patt.unsqueeze(0).unsqueeze(1)
        Pmatrix_tprs = Pmatrix_tprs.transpose(2, 0, 1).unsqueeze(1)
        convolved = torch.nn.functional.conv1d(test_patt, Pmatrix_tprs, padding=(UIs - 1,))
        wave.append(convolved.squeeze())

    wave = torch.stack(wave).permute(1, 2, 0)
    wave = wave.view(n_perUI_intrp, -1, nline).sum(-1)

    return wave.numpy()

def sparam_to_eyediwtch(config, ntwk, device='cuda'):
    try:
        params = TransientParams.from_config(config)
        line_sbrs = sparam_to_pulse(params, ntwk, device)
        half_steady, pda_eye, Pmatrix_tprs_lines = pulse_to_pda(params, line_sbrs)

        test_patt_lines_user = generate_pattern(nline=len(line_sbrs))
        wave = get_waveform(test_patt_lines_user, params.n_perUI_intrp, Pmatrix_tprs_lines, len(line_sbrs))
        eyewidth_pct = get_vref_eyewidth(params, half_steady, wave)
    except Exception as e:
        print(traceback.format_exc())

    return eyewidth_pct

def read_and_cascade_snps(snp_horiz, snp_tx, snp_rx, config, device):
    ntwk_horiz = rf.Network(snp_horiz)
    ntwk_tx = rf.Network(snp_tx)
    ntwk_rx = rf.Network(snp_rx)

    # Add inductance to trace network
    ntwk_horiz = add_inductance_torch(ntwk_horiz, config.L_tx, config.L_rx, device)

    # Flipping the port order in rx end
    ntwk_rx.flip()

    ntwk = ntwk_tx ** ntwk_horiz ** ntwk_rx
    return ntwk

def assign_tx_rx_directions(ntwk, directions=None):
    def divide_until_odd(num):
        result = []
        while num % 2 == 0:
            num = num // 2
        result.append(num)
        return result

    nLines = ntwk.s.shape[1] // 2

    if not directions:
        block_size = np.random.choice(divide_until_odd(nLines), 1)[0]
        nBlocks = nLines // block_size
        blocks = [0] * (nBlocks // 2) + [1] * (nBlocks // 2)
        np.random.shuffle(blocks)
        directions = np.repeat(blocks, block_size)
    else:
        directions = np.array(directions)
    mask = directions.astype(bool)

    front_ports = np.arange(0, nLines)
    back_ports = front_ports + nLines
    curr_ports = np.dstack([front_ports, back_ports]).reshape(-1)

    front_ports[mask], back_ports[mask] = back_ports[mask], front_ports[mask]
    next_ports = np.dstack([front_ports, back_ports]).reshape(-1)

    ntwk.renumber(curr_ports, next_ports)
    return ntwk, directions

def snp_eyewidth_simulation(config, snp_file, directions=None, device='cuda'):
    snp_horiz, snp_tx, snp_rx = snp_file
    ntwk = read_and_cascade_snps(snp_horiz, snp_tx, snp_rx, config, device)
    ntwk, directions = assign_tx_rx_directions(ntwk, directions)
    line_ew = sparam_to_eyediwtch(config.to_dict(), ntwk, device)
    return line_ew, directions

if __name__ == '__main__':
    from line_profiler import LineProfiler

    from bound_param import LinearParameter, LogParameter, ParameterSet, SampleResult

    def main(device=None):            
        snp_horiz = "/Users/willychan/Documents/work/EyeDiagramNet/test_data/transmission_lines_48lines_seed0.s96p"
        snp_tx = "/Users/willychan/Documents/work/EyeDiagramNet/test_data/vertical/tx_snp.s96p"
        snp_rx = "/Users/willychan/Documents/work/EyeDiagramNet/test_data/vertical/rx_snp.s96p"
        snp_file = (snp_horiz, snp_tx, snp_rx)

        config_dict = {
            "R_tx": 32,
            "R_rx": 1.0e9,
            "C_tx": 4e-13,
            "C_rx": 2e-13,
            "L_tx": 2e-10,
            "L_rx": 1.6e-9,
            "pulse_amplitude": 0.55,
            "bits_per_sec": 1.3e10,
            "vmask": 0.03
        }
        config = SampleResult(**config_dict)

        line_ew, directions = snp_eyewidth_simulation(config, snp_file, directions=None, device=device)
        print(line_ew)

    lp = LineProfiler()
    lp.add_function(snp_eyewidth_simulation)
    lp.add_function(read_and_cascade_snps)

    # Run without specifying device to use auto-detection
    lp.run('main()')
    # lp.run('main("cuda")')
    # lp.run('main("mps")')
    lp.print_stats()