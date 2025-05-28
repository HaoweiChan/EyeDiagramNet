import json
import torch
import loguru
import torch
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from einops import rearrange, repeat
from pathlib import Path, PosixPath
from itertools import combinations
from lightning.pytorch.utilities import rank_zero_only

@rank_zero_only
def log_info(msg):
    loguru.logger.info(msg)

def read_snp(snp_file: PosixPath):
    if snp_file.suffix != '.npz':
        return rf.Network(snp_file).s

    data = np.load(snp_file)
    compress_arr = data['compress_arr']
    diag = data['diag']

    decompress_arr = np.repeat(compress_arr, 2, axis=0)
    if diag.shape[0] != compress_arr.shape[0]:
        decompress_arr = np.delete(decompress_arr, -1, axis=0)
    decompress_arr[:,:2] = np.triu(decompress_arr[:,:2]) + np.transpose(np.triu(decompress_arr[:,:2], 1), (0, 2, 1))
    decompress_arr[1::2] = np.tril(decompress_arr[1::2], -1) + np.transpose(np.tril(decompress_arr[1::2], -1), (0, 2, 1))
    decompress_arr[1::2, np.arange(diag.shape[1]), np.arange(diag.shape[1])] = diag

    return decompress_arr

def parse_snps(snp_dir, indices):
    suffix = '*.s*p'
    if len(list(snp_dir.glob("*.npz"))):
        # suffix = '*.npz'
        pass
    # filter using file indices
    all_snp_files = [None] * len(indices)
    for f in snp_dir.glob(suffix):
        idx = int(f.stem.split('_')[-1])
        idx = np.where(indices == idx)[0]
        if len(idx):
            all_snp_files[idx[0]] = f

    missing_idx = [i for i, x in enumerate(all_snp_files) if x is None]
    missing_idx = sorted(missing_idx, reverse=True)
    for i in missing_idx:
        all_snp_files.pop(i)

    return all_snp_files, missing_idx

def flip_snp(s_matrix: torch.Tensor) -> torch.Tensor:
    """
    Invert the ports of a network's scattering parameter matrix (s-matrix), 'flipping' it over left and right.
    In case the network is a 2n-port and n > 1, the 'second' numbering scheme is
    assumed to be consistent with the ** cascade operator.

    Parameters
    ----------
    s_matrix : torch.Tensor
        Scattering parameter matrix. Shape should be `2n x 2n`, or `f x 2n x 2n`.

    Returns
    -------
    flipped_s_matrix : torch.Tensor
        Flipped scattering parameter matrix.

    See Also
    --------
    renumber
    """
    # Clone the input tensor to create a new tensor for the flipped s-matrix
    flipped_s_matrix = s_matrix.clone()

    # Get the number of columns and rows (should be equal and even)
    num_cols = s_matrix.shape[-1]
    num_rows = s_matrix.shape[-2]
    n = num_cols // 2  # Number of ports n (since total ports = 2n)

    # Check if the matrix is square and has an even dimension
    if (num_cols == num_rows) and (num_cols % 2 == 0):
        # Create index tensors for old and new port ordering
        old_indices = torch.arange(0, 2 * n)
        new_indices = torch.cat((torch.arange(n, 2 * n), torch.arange(0, n)))

        if s_matrix.dim() == 2:
            # For 2D tensors (single s-matrix)
            # Renumber rows and columns according to the new indices
            flipped_s_matrix[new_indices, :] = flipped_s_matrix[old_indices, :]  # Renumber rows
            flipped_s_matrix[:, new_indices] = flipped_s_matrix[:, old_indices]  # Renumber columns
        else:
            # For higher-dimensional tensors (e.g., frequency x ports x ports)
            # Use ellipsis to handle any leading dimensions
            flipped_s_matrix[..., new_indices, :] = flipped_s_matrix[..., old_indices, :]  # Renumber rows
            flipped_s_matrix[..., new_indices] = flipped_s_matrix[..., :, old_indices]  # Renumber columns
    else:
        raise IndexError('Matrices should be 2n x 2n, or f x 2n x 2n')

    return flipped_s_matrix

def renumber_snp(s_matrix: torch.Tensor) -> torch.Tensor:
    """
    Renumber the ports of a network's scattering parameter matrix (s-matrix),
    transforming the port order from (1, 2, ..., n) to (1, n/2+1, 2, n/2+2, ..., n/2, n).

    Parameters
    ----------
    s_matrix : torch.Tensor
        Scattering parameter matrix. Shape should be `n x n`, or `f x n x n`.

    Returns
    -------
    renumbered_s_matrix : torch.Tensor
        Renumbered scattering parameter matrix.

    Raises
    ------
    ValueError
        If the number of ports `n` is not even.

    See Also
    --------
    flip
    """
    # Clone the input tensor to create a new tensor for the renumbered s-matrix
    renumbered_s_matrix = s_matrix.clone()

    # Get the number of ports (should be even)
    num_ports = s_matrix.shape[-1]
    num_rows = s_matrix.shape[-2]

    # Check if the matrix is square and has an even number of ports
    if (num_ports == num_rows) and (num_ports % 2 == 0):
        n = num_ports  # Total number of ports
        half_n = n // 2  # Half the number of ports

        # Generate the original and new port indices
        # Original indices: [0, 1, 2, ..., n-1]
        original_indices = torch.arange(n)

        # New indices: [0, half_n, 1, half_n+1, ..., half_n-1, n-1]
        new_indices = torch.empty(n, dtype=torch.long)
        new_indices[0:2] = torch.arange(0, half_n)
        new_indices[1::2] = torch.arange(half_n, n)

        if s_matrix.dim() == 2:
            # For 2D tensors (single s-matrix)
            # Renumber rows and columns according to the new indices
            renumbered_s_matrix[new_indices, :] = renumbered_s_matrix[original_indices, :]  # Renumber rows
            renumbered_s_matrix[:, new_indices] = renumbered_s_matrix[:, original_indices]  # Renumber columns
        else:
            # For higher-dimensional tensors (e.g., frequency x ports x ports)
            # Use ellipsis to handle any leading dimensions
            renumbered_s_matrix[..., new_indices, :] = renumbered_s_matrix[..., original_indices, :]  # Renumber rows
            renumbered_s_matrix[..., new_indices] = renumbered_s_matrix[..., :, original_indices]  # Renumber columns
    else:
        raise ValueError('Number of ports must be even and s-matrix must be square.')

    return renumbered_s_matrix

def complex_to_db(x):
    if not torch.is_complex(x):
        return 10 * torch.log10(x)

    min_val = torch.finfo(x.dtype).tiny
    magnitude = torch.clamp(x.abs(), min=min_val)
    return 20 * torch.log10(magnitude)

def image_to_buffer(figure):
    buf = BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    return image

def plot_sparam_curve(outputs, metrics):
    pair = outputs['pair']
    pred_freq = complex_to_db(outputs['pred_freq'])
    true_freq = complex_to_db(outputs['true_freq'])
    # pred_time = complex_to_db(outputs['pred_time'])
    # true_time = complex_to_db(outputs['true_time'])
    # pred_real = log_compression_transform(outputs['pred_freq'].real.detach().cpu())
    # true_real = log_compression_transform(outputs['true_freq'].real.cpu())
    # pred_imag = log_compression_transform(outputs['pred_freq'].imag.detach().cpu())
    # true_imag = log_compression_transform(outputs['true_freq'].imag.cpu())
    pred_real = outputs['pred_freq'].real.detach().cpu()
    pred_imag = outputs['pred_freq'].imag.detach().cpu()
    true_real = outputs['true_freq'].real.cpu()
    true_imag = outputs['true_freq'].imag.cpu()

    metrics = {k: v for k, v in metrics.items() if v != 0}
    pretty_string = '\n'.join(f'{k}: {str(v.round(decimals=3))[7:12]}' for k, v in metrics.items())

    plt.close()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].set_title(f'S-parameter of pair {pair}')
    axs[0].plot(pred_freq.detach().cpu(), alpha=0.8, label='Pred')
    axs[0].plot(true_freq.cpu(), alpha=0.8, label='True')
    axs[0].text(0.05, 0.95, pretty_string,
                verticalalignment='bottom', horizontalalignment='left', transform=axs[0].transAxes)
    axs[0].legend(loc='lower right')

    axs[1].set_title(f'Real/Imag of S-parameter of pair {pair}')
    axs[1].plot(pred_real, 'b-', alpha=0.7, label='Real Pred')
    axs[1].plot(true_real, 'b--', alpha=0.7, label='Real True')
    axs[1].plot(pred_imag, 'r-', alpha=0.7, label='Imag Pred')
    axs[1].plot(true_imag, 'r--', alpha=0.7, label='Imag True')
    axs[1].legend(loc='lower right')

    # pred_real = log_compression_transform(pred_real)
    # true_real = log_compression_transform(true_real)
    # pred_imag = log_compression_transform(pred_imag)
    # true_imag = log_compression_transform(true_imag)
    # time_signal = generate_trapezoidal_signal()
    time_signal = generate_trapezoidal_signal()
    pred_tdr = tdr_analysis(outputs['pred_freq'].unsqueeze(0), time_signal).squeeze(0).detach().cpu()
    true_tdr = tdr_analysis(outputs['true_freq'].unsqueeze(0), time_signal).squeeze(0).cpu()

    axs[2].set_title(f'TDT of pair {pair}')
    axs[2].plot(pred_tdr, 'b-', alpha=0.5, label='TDT Pred')
    axs[2].plot(true_tdr, 'b-', alpha=0.8, label='TDT True')
    axs[2].legend(loc='lower right')

    # axs[1].set_title('Time domain of pair {pair}')
    # axs[1].plot(pred_time.cpu())
    # axs[1].plot(true_time.cpu())
    return fig

def plot_ew_curve(outputs, metrics, ew_threshold, sigma=2):
    pred_ew = outputs['pred_ew'].float().detach().cpu()
    true_ew = outputs['true_ew'].float().cpu()
    pred_prob = outputs['pred_prob'].float().detach().cpu()
    true_prob = outputs['true_prob'].float().cpu()
    pred_sigma = outputs['pred_sigma'].float().detach().cpu()

    # Conversions
    pred_mask = pred_ew > ew_threshold
    upper_mask = np.ma.masked_where(~pred_mask, pred_ew + sigma * pred_sigma)
    lower_masked = np.ma.masked_where(~pred_mask, pred_ew - sigma * pred_sigma)

    stage_key = next(iter(metrics)).split('_')[0].replace('/', '').capitalize()
    metrics = {k.split('/')[1]: v for k, v in metrics.items() if v != 0}
    # pretty_string = '\n'.join(f'{k.split("/")[-1]}: {str(v.round(decimals=3))[7:12]}' for k, v in metrics.items())
    metric_lines = [f'{k:<9}: {v}' for k, v in metrics.items()]
    pretty_string = f"{stage_key}\\n" + '\n'.join(metric_lines)

    plt.close()
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[2, 1])

    # First subplot for eye width
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Eye width')
    ax1.plot(pred_ew * pred_mask, color='#1777b4', alpha=0.8, label='Pred')
    ax1.fill_between(np.arange(len(pred_ew)), upper_mask, lower_masked, color='#aec7e8', alpha=0.3, label='\u00B1\u03C3')
    ax1.plot(true_ew * true_prob, color='#111111', alpha=0.8, label='True')
    ax1.legend(loc='lower right')

    # Second subplot for open-eye Probability
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.set_title('Prediction Probability')
    ax2.plot(pred_prob, color='#1777b4', alpha=0.8)
    ax2.axhline(ew_threshold, color='black', linestyle='--', linewidth=1)
    ax2.set_ylim(-0.1, 1.1)  # Set y-axis limits to [0, 1]
    ax2.yaxis.set_ticks([0, ew_threshold, 1])
    ax2.yaxis.set_ticklabels(['0', f'ew_threshold: {ew_threshold:.1f}', '1'])

    # Right-hand text box
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis('off')
    ax_text.text(0, 1, pretty_string, fontsize=11, family='monospace', verticalalignment='top', horizontalalignment='left', fontweight='bold')

    fig.tight_layout()
    return fig

def tdr_analysis(s_parameters, time_signal, interp_scale=5):
    s_parameters = rearrange(s_parameters, 'b f -> b 1 f')
    s_parameters_interp_real = F.interpolate(s_parameters.real, scale_factor=5, mode='linear')
    s_parameters_interp_image = F.interpolate(s_parameters.imag, scale_factor=5, mode='linear')
    s_parameters = torch.complex(s_parameters_interp_real, s_parameters_interp_image)
    batch_num, _, freq_num = s_parameters.size()

    impulse_response = torch.fft.ifft(s_parameters, dim=-1).real

    time = len(time_signal)
    time_signal = torch.tensor(time_signal, device=s_parameters.device)
    time_signal = repeat(time_signal, 't -> b t', b=batch_num)

    tdr_result = F.conv1d(signal, impulse_response, groups=batch_num, padding=freq_num - 1)
    tdr_result = tdr_result.squeeze(0)

    return tdr_result[:, :time]

def generate_trapezoidal_signal(
    amplitude=1.0,
    rise_time=1e-12,
    fall_time=1e-12,
    high_time=9e-12,
    zero_time=1e-12,
    total_time=100e-12,
    sample_rate=1e13
):
    t = torch.arange(0, total_time, 1/sample_rate)
    signal = torch.zeros_like(t)

    zero_samples = int(zero_time * sample_rate)
    rise_samples = int(rise_time * sample_rate)
    high_samples = int(high_time * sample_rate)
    fall_samples = int(fall_time * sample_rate)

    start_rise = zero_samples
    end_rise = start_rise + rise_samples
    start_high = end_rise
    end_high = start_high + high_samples
    start_fall = end_high
    end_fall = start_fall + fall_samples

    signal[start_rise:end_rise] = torch.linspace(0, amplitude, rise_samples)
    signal[start_high:end_high] = amplitude
    signal[start_fall:end_fall] = torch.linspace(amplitude, 0, fall_samples)

    return signal

def log_compression_transform(x, epsilon=1e-20):
    max_val = torch.tensor(torch.finfo(x.dtype).max, device=x.device)
    clamped_x = torch.clamp(x.abs(), 0, max_val * epsilon)
    return torch.sign(x) * torch.log10(clamped_x / epsilon)

def greedy_covering_design(total_elements, group_size, window_size=None):
    if window_size is None:
        window_size = total_elements
    cache_folder = Path("common/greedy_covering_design")
    cache_folder.mkdir(parents=True, exist_ok=True)
    cache_file = cache_folder / f"{total_elements}_{group_size}_{window_size}.json"

    if cache_file.exists():
        with cache_file.open('r') as f:
            return np.array(json.load(f))

    def generate_pairs_within_window(total_elements, window_size):
        all_pairs = []
        for i in range(total_elements - window_size + 1):
            lines = list(range(i, i + window_size))
            pairs = list(combinations(lines, 2))
            all_pairs.extend(pairs)
        return set(all_pairs)

    def generate_valid_groups(total_elements, group_size, window_size):
        valid_groups = []
        for i in range(total_elements - window_size + 1):
            for group in combinations(range(i, i + window_size), group_size):
                valid_groups.append(group)
        return valid_groups

    window_size = max(window_size, group_size)
    all_pairs = generate_pairs_within_window(total_elements, window_size)
    valid_groups = generate_valid_groups(total_elements, group_size, window_size)

    selected_groups = []
    covered_pairs = set()

    while all_pairs - covered_pairs:
        best_group = None
        best_covered = set()
        for group in valid_groups:
            covered = {pair for pair in combinations(group, 2) if pair in all_pairs}
            new_covered = covered - covered_pairs
            if len(new_covered) > len(best_covered):
                best_group = group
                best_covered = new_covered

        if not best_group:
            break

        selected_groups.append(best_group)
        covered_pairs.update(best_covered)

    selected_groups = np.array(selected_groups)
    array1 = 2 * selected_groups
    array2 = 2 * selected_groups + 1

    interleaved_groups = np.empty(array1.size + array2.size, dtype=array1.dtype)
    interleaved_groups[0::2] = array1.flatten()
    interleaved_groups[1::2] = array2.flatten()
    interleaved_groups = interleaved_groups.reshape((len(selected_groups), group_size * 2))

    with open(cache_file, 'w') as f:
        json.dump(interleaved_groups.tolist(), f)

    return interleaved_groups