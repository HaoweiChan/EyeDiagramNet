import torch
import numpy as np
import matplotlib.pyplot as plt
from ml.utils.visualization import plot_ew_curve

def generate_sample_plot():
    """Generate a sample plot using plot_ew_curve with random data."""
    
    # Set random seed for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create sample data
    batch_size = 4
    seq_len = 50
    
    # Generate realistic-looking data
    outputs = {
        'pred_ew': torch.randn(batch_size, seq_len) * 15 + 45,  # Eye width predictions around 45±15
        'true_ew': torch.randn(batch_size, seq_len) * 12 + 35,  # True eye width around 35±12
        'pred_prob': torch.sigmoid(torch.randn(batch_size, seq_len) * 0.5 + 0.3),  # Probabilities around 0.3-0.7
        'true_prob': torch.sigmoid(torch.randn(batch_size, seq_len) * 0.3 + 0.4),  # True probabilities
        'pred_sigma': torch.abs(torch.randn(batch_size, seq_len)) * 3 + 1,  # Uncertainty around 1-4
        'meta': {
            'snp_horiz': ['/proj/siaiadm/AI_training_data/D2D/UCIe_Trace/pattern2/cowos_8mi/snp/UCIe_pattern2_cowos_8mi-112.s96p'] * batch_size,
            'n_ports': torch.tensor([96] * batch_size),
            'boundary': [
                {
                    'AC_gain': -5.0,
                    'C_drv': 2.5e-12,
                    'C_odt': 1.8e-12,
                    'DC_gain': -2.4,
                    'L_drv': 3.2e-9,
                    'L_odt': 1.5e-9,
                    'R_drv': 40.0,
                    'R_odt': 1000000000.0,
                    'bits_per_sec': 12000000000.0,
                    'fp1': -40000000000.0,
                    'fp2': -40000000000.0,
                    'pulse_amplitude': 0.3,
                    'vmask': 0.06
                }
            ] * batch_size,
            'param_types': [['MIX_PARAMS', 'CTLE_PARAMS']] * batch_size
        }
    }
    
    # Generate realistic metrics
    metrics = {
        'train/loss': 0.089,
        'train/mae': 18.234,
        'train/mape': 0.412,
        'train/r2': 0.156,
        'train/accuracy': 0.723,
        'train/f1': 0.781,
        'val/loss': 0.092,
        'val/mae': 19.567,
        'val/mape': 0.445,
        'val/r2': 0.123
    }
    
    ew_threshold = 0.2
    
    print("Generating sample plot with random data...")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"Eye width threshold: {ew_threshold}")
    print(f"Prediction uncertainty enabled: {outputs['pred_sigma'].abs().sum() > 1e-6}")
    
    # Generate the plot
    fig = plot_ew_curve(outputs, metrics, ew_threshold)
    
    print("\n✅ Sample plot generated successfully!")
    print("Key features to examine:")
    print("- Compact layout with reduced white space")
    print("- Metrics box positioned closer to plots")
    print("- Uncertainty shading (if pred_sigma > 0)")
    print("- Pretty-printed metadata in bottom section")
    print("- Better spacing and font sizes")
    
    # Save the plot
    output_path = "sample_ew_plot.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📁 Plot saved as: {output_path}")
    
    return fig

if __name__ == "__main__":
    fig = generate_sample_plot()
