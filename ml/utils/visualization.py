import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from einops import rearrange, repeat
import io

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
    config = outputs['config']

    # Random sample one batch index to get (N,) arrays for plotting
    batch_size = pred_ew.shape[0]
    sample_idx = np.random.randint(0, batch_size)
    
    pred_ew = pred_ew[sample_idx]
    true_ew = true_ew[sample_idx]
    pred_prob = pred_prob[sample_idx]
    true_prob = true_prob[sample_idx]
    pred_sigma = pred_sigma[sample_idx]
    if isinstance(config, dict):
        config = {k: (v[sample_idx] if isinstance(v, (list, np.ndarray)) else v) for k, v in config.items()}

    # Conversions
    pred_mask = pred_ew > ew_threshold

    # Calculate and clip uncertainty bounds to be within [0, 100]
    upper_bound = np.clip(pred_ew + sigma * pred_sigma, a_min=None, a_max=100.0)
    lower_bound = np.clip(pred_ew - sigma * pred_sigma, a_min=0.0, a_max=None)
    
    upper_mask = np.ma.masked_where(~pred_mask, upper_bound)
    lower_masked = np.ma.masked_where(~pred_mask, lower_bound)

    stage_key = next(iter(metrics)).split('_')[0].replace('/', '').capitalize()
    metrics = {k.split('/')[1]: v for k, v in metrics.items() if v != 0}
    
    # Format values like the desired clean format
    def format_value(v):
        if hasattr(v, 'item'):  # Handle tensor values
            v = v.item()
        if isinstance(v, (int, float)):
            return f"{v:.3f}"
        return str(v)
    
    metric_lines = [f'{k:<8} : {format_value(v)}' for k, v in metrics.items()]
    pretty_string = f"{stage_key}\n\n" + '\n'.join(metric_lines)

    # Format config details
    config_pretty_string = ""
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict):
                config_pretty_string += f"\n{key.upper()}:\n"
                for sub_key, sub_value in value.items():
                    config_pretty_string += f"  {sub_key:<10}: {sub_value}\n"
            elif isinstance(value, (list, np.ndarray)):
                config_pretty_string += f"\n{key.upper()}: {value.tolist()}\n"
            else:
                config_pretty_string += f"\n{key.upper()}: {value}\n"
    pretty_string += config_pretty_string

    plt.close()
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[2, 1])

    # First subplot for eye width
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Eye width')
    ax1.plot(pred_ew * pred_mask, color='#1777b4', alpha=0.8, label='Pred')
    if pred_sigma.abs().sum() > 1e-6:
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
    ax2.yaxis.set_ticklabels(['0', f'{ew_threshold:.1f}', '1'])

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