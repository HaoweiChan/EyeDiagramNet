import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

def plot_eye_width_comparison(pred_csv_path, true_csv_path, output_dir):
    """
    Plot eye width comparison between predicted and true results.
    
    Args:
        pred_csv_path: Path to predicted results CSV
        true_csv_path: Path to true/ground truth results CSV  
        output_dir: Directory to save the plots
    """
    # Load the CSV files
    pred_df = pd.read_csv(pred_csv_path)
    true_df = pd.read_csv(true_csv_path)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get line columns (excluding 'index' column)
    line_columns = [col for col in pred_df.columns]
    line_numbers = [int(col) for col in line_columns]
    
    # Plot comparison for each row (sample)
    for idx in tqdm(range(len(pred_df)), desc="Creating individual plots"):
        if idx >= len(true_df):
            print(f"Warning: True data has fewer rows than predicted data. Stopping at row {idx}")
            break
            
        # Extract eye width values for current row
        pred_values = [pred_df.iloc[idx][col] for col in line_columns]
        true_values = [true_df.iloc[idx][col] for col in line_columns]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot both lines
        plt.plot(line_numbers, pred_values, 'b-o', label='Predicted', linewidth=2, markersize=4)
        plt.plot(line_numbers, true_values, 'r-s', label='True', linewidth=2, markersize=4)
        
        # Customize the plot
        plt.xlabel('Line Number')
        plt.ylabel('Eye Width')
        plt.title(f'Eye Width Comparison - Sample {pred_df.index[idx]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add MAE statistics
        mae = np.mean(np.abs(np.array(pred_values) - np.array(true_values)))
        plt.text(0.02, 0.98, f'MAE: {mae:.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save the plot
        sample_idx = pred_df.index[idx]
        output_path = output_dir / f'sample_{sample_idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    # Create a summary plot showing all samples
    plt.figure(figsize=(15, 10))
    
    # Calculate overall statistics
    all_pred = []
    all_true = []
    sample_indices = []
    
    for idx in range(min(len(pred_df), len(true_df))):
        pred_values = [pred_df.iloc[idx][col] for col in line_columns]
        true_values = [true_df.iloc[idx][col] for col in line_columns]
        all_pred.extend(pred_values)
        all_true.extend(true_values)
        sample_indices.extend([pred_df.iloc[idx]["index"]] * len(line_columns))
    
    # Scatter plot of predicted vs true
    plt.subplot(2, 2, 1)
    plt.scatter(all_true, all_pred, alpha=0.6)
    plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--', label='Perfect Prediction')
    plt.xlabel('True Eye Width')
    plt.ylabel('Predicted Eye Width')
    plt.title('Predicted vs True Eye Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = np.array(all_pred) - np.array(all_true)
    plt.scatter(all_true, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Eye Width')
    plt.ylabel('Residuals (Pred - True)')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # Distribution of residuals
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (Pred - True)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    # Sample-wise MAE
    plt.subplot(2, 2, 4)
    sample_maes = []
    unique_samples = sorted(pred_df['index'].unique())
    
    for sample_idx in unique_samples:
        if sample_idx in true_df['index'].values:
            pred_row = pred_df[pred_df['index'] == sample_idx].iloc[0]
            true_row = true_df[true_df['index'] == sample_idx].iloc[0]
            
            pred_vals = [pred_row[col] for col in line_columns]
            true_vals = [true_row[col] for col in line_columns]
            
            mae = np.mean(np.abs(np.array(pred_vals) - np.array(true_vals)))
            sample_maes.append(mae)
    
    plt.bar(range(len(sample_maes)), sample_maes)
    plt.xlabel('Sample Index')
    plt.ylabel('MAE')
    plt.title('MAE per Sample')
    plt.xticks(range(len(unique_samples)), unique_samples)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = output_dir / 'comparison_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plot: {summary_path}")
    
    # Print overall statistics
    overall_mae = np.mean(np.abs(residuals))
    print(f"\nOverall Statistics:")
    print(f"MAE: {overall_mae:.4f}")
    print(f"Number of samples compared: {min(len(pred_df), len(true_df))}")

def main():
    parser = argparse.ArgumentParser(description='Plot eye width comparison between predicted and true results')
    parser.add_argument('--pred_csv', type=str, required=True, 
                       help='Path to predicted results CSV file')
    parser.add_argument('--true_csv', type=str, required=True,
                       help='Path to true/ground truth results CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save the comparison plots')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.pred_csv):
        raise FileNotFoundError(f"Predicted CSV file not found: {args.pred_csv}")
    if not os.path.exists(args.true_csv):
        raise FileNotFoundError(f"True CSV file not found: {args.true_csv}")
    
    plot_eye_width_comparison(args.pred_csv, args.true_csv, args.output_dir)

if __name__ == "__main__":
    main() 