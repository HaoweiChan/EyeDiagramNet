import io
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from io import BytesIO
from PIL import Image


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

def _format_dict_for_plot(data_dict, title=""):
    """Formats a dictionary into a string for plot display."""
    if not data_dict:
        return ""
    
    lines = [f"\n{title.upper()}:"]
    for key, value in data_dict.items():
        if isinstance(value, (int, float)):
            lines.append(f"  {key:<12}: {value:.4f}")
        else:
            lines.append(f"  {key:<12}: {value}")
    return "\n".join(lines)

def _format_meta_for_subplot(meta_dict):
    """Format metadata dictionary for clean display in subplot, converting tensors to readable format."""
    if not meta_dict:
        return ""
    
    formatted_lines = []
    for key, value in meta_dict.items():
        # Convert tensors and arrays to readable format
        if torch.is_tensor(value):
            if value.numel() == 1:
                # Single value tensor
                formatted_value = f"{value.item():.4f}" if value.dtype.is_floating_point else str(value.item())
            else:
                # Multi-value tensor - show as list
                formatted_value = value.tolist()
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                formatted_value = f"{value.item():.4f}" if np.issubdtype(value.dtype, np.floating) else str(value.item())
            else:
                formatted_value = value.tolist()
        elif isinstance(value, (list, tuple)):
            # Already a list/tuple, but ensure nested tensors are converted
            formatted_value = [item.tolist() if torch.is_tensor(item) else item for item in value]
        elif isinstance(value, (int, float)):
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
        else:
            formatted_value = str(value)
        
        # Format the line
        if isinstance(formatted_value, list):
            if len(formatted_value) <= 5:
                formatted_lines.append(f"{key}: {formatted_value}")
            else:
                # For long lists, show first few items and count
                preview = formatted_value[:3]
                formatted_lines.append(f"{key}: {preview}... ({len(formatted_value)} items)")
        else:
            formatted_lines.append(f"{key}: {formatted_value}")
    
    return "\n".join(formatted_lines)

def plot_ew_curve(outputs, metrics, ew_threshold, sigma=2):
    pred_ew = outputs['pred_ew'].float().detach().cpu()
    true_ew = outputs['true_ew'].float().cpu()
    pred_prob = outputs['pred_prob'].float().detach().cpu()
    true_prob = outputs['true_prob'].float().cpu()
    pred_sigma = outputs['pred_sigma'].float().detach().cpu()
    meta = outputs['meta']

    # Random sample one batch index to get (N,) arrays for plotting
    batch_size = pred_ew.shape[0]
    sample_idx = np.random.randint(0, batch_size)
    
    pred_ew = pred_ew[sample_idx]
    true_ew = true_ew[sample_idx]
    pred_prob = pred_prob[sample_idx]
    true_prob = true_prob[sample_idx]
    pred_sigma = pred_sigma[sample_idx]
    config_keys = meta['config_keys']
    param_types = [m[0] for m in meta['param_types']]
    meta = {k: v[sample_idx] for k, v in meta.items() if k not in ['config_keys', 'param_types']}
    meta['boundary'] = {k: v.item() for k, v in zip(config_keys, meta['boundary'])}
    meta['param_types'] = param_types
    
    # Conversions
    pred_mask = pred_ew > ew_threshold

    # Calculate and clip uncertainty bounds to be within [0, 100]
    upper_bound = np.clip(pred_ew + sigma * pred_sigma, a_min=None, a_max=100.0)
    lower_bound = np.clip(pred_ew - sigma * pred_sigma, a_min=0.0, a_max=None)
    
    upper_mask = np.ma.masked_where(~pred_mask, upper_bound)
    lower_masked = np.ma.masked_where(~pred_mask, lower_bound)

    stage_key = next(iter(metrics)).split('_')[0].replace('/', '').capitalize()
    metrics = {k.split('/')[1]: v for k, v in metrics.items() if v != 0 and not math.isnan(v)}
    
    # Format values like the desired clean format
    def format_value(v):
        if hasattr(v, 'item'):  # Handle tensor values
            v = v.item()
        if isinstance(v, (int, float)):
            return f"{v:.3f}"
        return str(v)
    
    metric_lines = [f'{k:<8} : {format_value(v)}' for k, v in metrics.items()]
    metrics_display_string = f"{stage_key}\n\n" + '\n'.join(metric_lines)

    plt.close()
    fig = plt.figure(figsize=(12, 8))  # Reduced figure size for more compact layout
    
    # Create a more compact grid layout with tighter spacing
    gs = fig.add_gridspec(3, 3, width_ratios=[2.5, 0.3, 1], height_ratios=[2, 1, 1], 
                          hspace=0.15, wspace=0.05)  # Much tighter spacing

    # First subplot for eye width (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Eye width', fontsize=11, fontweight='bold', pad=5)
    ax1.plot(pred_ew * pred_mask, color='#1777b4', alpha=0.8, label='Pred', linewidth=1.5)
    if pred_sigma.abs().sum() > 1e-6:
        ax1.fill_between(np.arange(len(pred_ew)), upper_mask, lower_masked, color='#aec7e8', alpha=0.3, label='±σ')
    ax1.plot(true_ew * true_prob, color='#111111', alpha=0.8, label='True', linewidth=1.5)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Second subplot for prediction probability (middle left)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.set_title('Prediction Probability', fontsize=11, fontweight='bold', pad=5)
    ax2.plot(pred_prob, color='#1777b4', alpha=0.8, linewidth=1.5)
    ax2.axhline(ew_threshold, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylim(-0.05, 1.05)  # Tighter y-axis limits
    ax2.yaxis.set_ticks([0, ew_threshold, 1])
    ax2.yaxis.set_ticklabels(['0', f'{ew_threshold:.1f}', '1'])
    ax2.grid(True, alpha=0.3)
    
    # Remove x-axis labels from top plot to prevent overlap
    ax1.set_xticklabels([])

    # Right-hand text box for metrics (spans first two rows, positioned closer)
    ax_text = fig.add_subplot(gs[0:2, 2])
    ax_text.axis('off')
    
    # Position metrics text much closer to the plots
    ax_text.text(0.02, 0.95, metrics_display_string, 
                fontsize=10, family='monospace', 
                verticalalignment='top', horizontalalignment='left', 
                fontweight='bold', transform=ax_text.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

    # Metadata section (bottom, spans all columns, more compact)
    ax_meta = fig.add_subplot(gs[2, :])
    ax_meta.axis('off')
    
    if meta:
        meta_formatted = _format_meta_for_subplot_improved(meta)
        ax_meta.text(0.02, 0.95, f"METADATA:\n{meta_formatted}", 
                    fontsize=8, family='monospace', 
                    verticalalignment='top', horizontalalignment='left', 
                    fontweight='normal', transform=ax_meta.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

    # Use constrained_layout instead of tight_layout for better compatibility
    fig.set_constrained_layout(True)
    return fig


def _format_meta_for_subplot_improved(meta_dict):
    """Improved metadata formatting with better handling of nested structures."""
    if not meta_dict:
        return ""
    
    formatted_lines = []
    for key, value in meta_dict.items():
        if key == 'boundary' and isinstance(value, dict):
            # Special handling for boundary dictionary
            formatted_lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    formatted_value = f"{sub_value:.6f}" if isinstance(sub_value, float) else str(sub_value)
                else:
                    formatted_value = str(sub_value)
                formatted_lines.append(f"  {sub_key}: {formatted_value}")
        else:
            # Handle other values
            if torch.is_tensor(value):
                if value.numel() == 1:
                    formatted_value = f"{value.item():.6f}" if value.dtype.is_floating_point else str(value.item())
                else:
                    formatted_value = f"[{', '.join([f'{v:.3f}' for v in value.tolist()[:5]])}{'...' if len(value) > 5 else ''}]"
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    formatted_value = f"{value.item():.6f}" if np.issubdtype(value.dtype, np.floating) else str(value.item())
                else:
                    formatted_value = f"[{', '.join([f'{v:.3f}' for v in value.tolist()[:5]])}{'...' if len(value) > 5 else ''}]"
            elif isinstance(value, (list, tuple)):
                if len(value) <= 5:
                    formatted_value = str(value)
                else:
                    formatted_value = f"{value[:3]}... ({len(value)} items)"
            elif isinstance(value, (int, float)):
                formatted_value = f"{value:.6f}" if isinstance(value, float) else str(value)
            else:
                formatted_value = str(value)
            
            formatted_lines.append(f"{key}: {formatted_value}")
    
    return "\n".join(formatted_lines)

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

    tdr_result = F.conv1d(time_signal, impulse_response, groups=batch_num, padding=freq_num - 1)
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

def plot_sparam_reconstruction(
    freqs: np.ndarray,
    true_sparam: np.ndarray,
    recon_sparam: np.ndarray,
    port1: int,
    port2: int,
    title: str = "S-Parameter Reconstruction"
) -> Image.Image:
    """
    Plots the magnitude and phase of true vs. reconstructed S-parameters.

    Args:
        freqs: Array of frequency points.
        true_sparam: The ground truth S-parameter matrix (Freq, Port, Port).
        recon_sparam: The reconstructed S-parameter matrix.
        port1: Index of the first port.
        port2: Index of the second port.
        title: Title for the plot.

    Returns:
        A PIL Image object of the plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"{title}: S({port1+1}, {port2+1})", fontsize=16)

    # Magnitude Plot
    ax1.plot(freqs, 20 * np.log10(np.abs(true_sparam)), 'b-', label='True Magnitude (dB)')
    ax1.plot(freqs, 20 * np.log10(np.abs(recon_sparam)), 'r--', label='Reconstructed Magnitude (dB)')
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True)
    ax1.legend()

    # Phase Plot
    ax2.plot(freqs, np.angle(true_sparam, deg=True), 'b-', label='True Phase (deg)')
    ax2.plot(freqs, np.angle(recon_sparam, deg=True), 'r--', label='Reconstructed Phase (deg)')
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    
    return image

def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image to a PyTorch tensor for TensorBoard."""
    return torch.tensor(np.array(image)).permute(2, 0, 1) 