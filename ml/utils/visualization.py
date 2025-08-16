import io
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


def image_to_buffer(figure):
    buf = BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    return image

def plot_ew_curve(outputs, metrics, ew_threshold, sigma=2):
    pred_ew = outputs['pred_ew'].float().detach().cpu()
    true_ew = outputs['true_ew'].float().cpu()
    pred_prob = outputs['pred_prob'].float().detach().cpu()
    true_prob = outputs['true_prob'].float().cpu()
    meta = outputs['meta']

    # Random sample one batch index to get (N,) arrays for plotting
    batch_size = pred_ew.shape[0]
    sample_idx = np.random.randint(0, batch_size)
    
    pred_ew = pred_ew[sample_idx]
    true_ew = true_ew[sample_idx]
    pred_prob = pred_prob[sample_idx]
    true_prob = true_prob[sample_idx]
    
    # Correctly sample metadata
    sampled_meta = {}
    for k, v in meta.items():
        if isinstance(v, (list, torch.Tensor)) and len(v) == batch_size:
            sampled_meta[k] = v[sample_idx]
        elif k == 'param_types':
            sampled_meta[k] = [t[0] for t in v]
        else:
            sampled_meta[k] = v  # Keep non-batched metadata as is
    meta = sampled_meta
    
    # Conversions
    pred_mask = pred_ew > ew_threshold

    stage_key = next(iter(metrics)).split('_')[0].replace('/', ' ').upper()
    metrics = {k.split('/')[1]: v for k, v in metrics.items() if v != 0 and not math.isnan(v)}
    
    # Format values like the desired clean format
    def format_value(v):
        if hasattr(v, 'item'):  # Handle tensor values
            v = v.item()
        if isinstance(v, (int, float)):
            return f"{v:.3f}"
        return str(v)
    
    metric_lines = [f'{k:<8} : {format_value(v)}' for k, v in metrics.items()]
    metrics_display_string = f"{stage_key.replace('_', ' ')}\n\n" + '\n'.join(metric_lines)

    plt.close()
    fig = plt.figure(figsize=(12, 9))  # Increased figure size for better layout
    
    # Create a 3x3 grid layout with proper spacing
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1], height_ratios=[2, 1.5, 1], 
                          hspace=0.25, wspace=0.15)  # Increased spacing to prevent overlap

    # First subplot for eye width (top, spans all 3 columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Eye width', fontsize=12, fontweight='bold', pad=15)  # Increased pad
    ax1.plot(pred_ew * pred_mask, color='#1777b4', alpha=0.8, label='Pred', linewidth=1.5)
    ax1.plot(true_ew * true_prob, color='#111111', alpha=0.8, label='True', linewidth=1.5)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Second subplot for prediction probability (middle, spans all 3 columns)
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.set_title('Prediction Probability', fontsize=12, fontweight='bold', pad=15)  # Increased pad
    ax2.plot(pred_prob, color='#1777b4', alpha=0.8, linewidth=1.5)
    ax2.axhline(ew_threshold, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylim(-0.05, 1.05)  # Tighter y-axis limits
    ax2.yaxis.set_ticks([0, ew_threshold, 1])
    ax2.yaxis.set_ticklabels(['0', f'{ew_threshold:.1f}', '1'])
    ax2.grid(True, alpha=0.3)
    
    # Remove x-axis labels from top plot to prevent overlap
    ax1.set_xticklabels([])

    # Metrics box (bottom left - uses only first column, 1/3 width)
    ax_metrics = fig.add_subplot(gs[2, 0])
    ax_metrics.axis('off')
    
    # Metadata section (bottom right - uses last two columns, 2/3 width)
    ax_meta = fig.add_subplot(gs[2, 1:])
    ax_meta.axis('off')
    
    # Format metadata
    meta_formatted = _format_meta_for_subplot_improved(meta)
    meta_text = f"METADATA\n{'-' * 20}\n{meta_formatted}"
    
    # Position metrics text in bottom left (1/3 width)
    ax_metrics.text(0.05, 0.95, metrics_display_string, 
                fontsize=9, family='monospace', 
                verticalalignment='top', horizontalalignment='left', 
                fontweight='bold', transform=ax_metrics.transAxes)

    # Position metadata text in bottom right (2/3 width)
    if meta:
        ax_meta.text(0.02, 0.95, meta_text, 
                    fontsize=8, family='monospace', 
                    verticalalignment='top', horizontalalignment='left', 
                    fontweight='normal', transform=ax_meta.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

    # Use subplots_adjust for precise control over layout
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, hspace=0.3, wspace=0.15)
    return fig


def _format_meta_value(key, value):
    """Helper to format values for metadata display, with specific rules for different number types."""
    if not isinstance(value, (int, float)):
        return str(value)

    # Special handling for capacitance and inductance parameters - always show in scientific notation
    if key.startswith(('C_', 'L_')):
        return f"{value:.2e}"
    
    # Special handling for frequency parameters and other large numbers - show in scientific notation
    if key in ['fp1', 'fp2', 'bits_per_sec', 'R_odt'] or abs(value) >= 1e6:
        return f"{value:.2e}"

    # Rule for values that are effectively integers (but not large numbers)
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e6:
        return str(int(round(value)))

    # Rule for scientific notation for very small non-zero numbers
    if 0 < abs(value) and abs(value) < 1e-3:
        return f"{value:.2e}"

    # Default float formatting, removing trailing zeros to be clean
    if isinstance(value, float):
        return f"{value:.6f}".rstrip('0').rstrip('.')
    
    return str(value)

def _format_meta_for_subplot_improved(meta_dict):
    """Improved metadata formatting with better handling of nested structures and long text."""
    if not meta_dict:
        return ""
    
    formatted_lines = []
    for key, value in meta_dict.items():
        if key == 'boundary' and isinstance(value, dict):
            # Special handling for boundary dictionary
            formatted_lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                formatted_lines.append(f"  {sub_key}: {_format_meta_value(sub_key, sub_value)}")
        else:
            formatted_value_str = ""
            # Handle other values
            if torch.is_tensor(value):
                if value.numel() == 1:
                    formatted_value_str = _format_meta_value(key, value.item())
                else:
                    formatted_value_str = f"[{', '.join([f'{v:.3f}' for v in value.tolist()[:5]])}{'...' if len(value) > 5 else ''}]"
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    formatted_value_str = _format_meta_value(key, value.item())
                else:
                    formatted_value_str = f"[{', '.join([f'{v:.3f}' for v in value.tolist()[:5]])}{'...' if len(value) > 5 else ''}]"
            elif isinstance(value, (list, tuple)):
                if len(value) <= 5:
                    formatted_value_str = str(value)
                else:
                    formatted_value_str = f"{value[:3]}... ({len(value)} items)"
            elif isinstance(value, (int, float)):
                formatted_value_str = _format_meta_value(key, value)
            else:
                # Handle long string values (like snp_horiz) with line breaks
                if isinstance(value, str) and len(value) > 60:
                    # Break long strings at reasonable points
                    words = value.split('/')
                    formatted_value = ''
                    current_line = ''
                    for i, word in enumerate(words):
                        if i == 0:
                            current_line = word
                        else:
                            if len(current_line + '/' + word) > 60:
                                formatted_value += current_line + '\n'
                                current_line = '  ' + word
                            else:
                                current_line += '/' + word
                    formatted_value += current_line
                    formatted_value_str = formatted_value
                else:
                    formatted_value_str = str(value)
            
            formatted_lines.append(f"{key}: {formatted_value_str}")
    
    return "\n".join(formatted_lines)

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
