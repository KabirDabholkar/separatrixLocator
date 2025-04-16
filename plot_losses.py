import os

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import argparse
import sys
import numpy as np

def load_and_plot_losses(log_dirs, metrics=None, save_path=None, figsize=(12, 6)):
    """
    Load CSV logs from multiple directories and create plots for specified metrics across all models.
    
    Args:
        log_dirs (list of str): List of directories containing the CSV log files
        metrics (list, optional): List of metrics to plot. If None, plots all metrics in directory
        save_path (str, optional): Path to save the figure. If None, displays the plot
        figsize (tuple): Figure size in inches
    """
    # Convert log_dirs to Path objects
    log_dirs = [Path(log_dir) for log_dir in log_dirs]
    
    # Set up the plot grid
    n_metrics = len(metrics) if metrics else 1
    fig, axes = plt.subplots(n_metrics,1, figsize=figsize, sharex=True)
    if n_metrics == 1:
        axes = [axes]
    
    # Set style
    sns.set_style("whitegrid")
    
    # Set up a color palette for directories
    dir_colors = sns.color_palette("tab10", n_colors=len(log_dirs))
    
    # Iterate over each log directory
    for dir_idx, log_dir in enumerate(log_dirs):
        # Load model metadata
        try:
            metadata_path = log_dir / "model_id.csv"
            metadata_df = pd.read_csv(metadata_path)
            print(f'Loaded model_id from {log_dir}')
            metadata_df['model_id'] = metadata_df['value'].astype(int)
        except Exception as e:
            print(f"Error loading metadata from {log_dir}: {e}")
            continue
        
        # If metrics not specified, find all CSV files in Loss directory
        if metrics is None:
            metrics = [p.stem for p in (log_dir / "Loss").glob("*.csv")]

        # Plot each metric
        for ax_id,(ax, metric) in enumerate(zip(axes, metrics)):
            try:
                # Load the CSV file for this metric
                metric_path = metric if "/" in metric else f"Loss/{metric}"
                csv_path = log_dir / f"{metric_path}.csv"
                loss_df = pd.read_csv(csv_path)
                print(f'Loaded {metric} from {log_dir}')
                
                # Get unique model IDs
                unique_model_ids = sorted(metadata_df['model_id'].unique())
                print(f'Found {len(unique_model_ids)} unique models in {log_dir}')
                
                # Define n_models based on the number of unique model IDs
                n_models = len(unique_model_ids)
                
                # Initialize min_val and max_val
                min_val = float('inf')
                max_val = float('-inf')
                
                # Get the number of rows for each model (assuming equal distribution)
                rows_per_model = len(loss_df) // n_models
                
                for idx, model_id in enumerate(unique_model_ids):
                    # Get data for this model
                    start_idx = idx * rows_per_model
                    end_idx = (idx + 1) * rows_per_model if idx < n_models - 1 else len(loss_df)
                    model_data = loss_df.iloc[start_idx:end_idx]

                    # Apply smoothing if requested
                    if CONFIG['smooth'] > 0:
                        smoothed_values = moving_average(model_data['value'], CONFIG['smooth'])
                        # Adjust the step array to match the length of the smoothed data
                        smoothed_steps = model_data['step'].iloc[:len(smoothed_values)].reset_index(drop=True)
                        model_data = pd.DataFrame({'step': smoothed_steps, 'value': smoothed_values})
                    
                    # Plot
                    # Add a legend entry only for the first model in each directory
                    label = f'{log_dir}' if idx == 0 else None
                    ax.plot(
                        model_data['step'], model_data['value'],
                        label=label,
                        color=dir_colors[dir_idx],  # Use the same color for all models in this directory
                        alpha=0.6,
                        lw=1
                    )
                    
                    min_val = min(min_val, model_data['value'].min())
                    max_val = max(max_val, model_data['value'].max())
                
                # ax.set_title(metric.split('/')[-1])  # Remove 'Loss/' prefix from title
                if ax_id == 0:
                    ax.set_xlabel('Step')
                ax.set_ylabel(metric)
                
                # Use log scale if values span multiple orders of magnitude
                if max_val / min_val > 100:
                    ax.set_yscale('log')
                else:
                    # ax.set_xlim(0,)
                    # ax.set_ylim(0, )
                    ax.set_yscale('log')

            except Exception as e:
                print(f"Error plotting {metric} from {log_dir}: {e}")
    
    # Add legend to the plot
    plt.legend(title='Log Directories', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        # Increase right margin to accommodate legend
        plt.subplots_adjust(right=0.85)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def moving_average(data, window_size):
    """
    Compute the moving average of a 1D array.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

if __name__ == "__main__":
    # Configuration variables - modify these directly in the script
    # =========================================================
    CONFIG = {
        'log_dirs':
            [Path('results/bistable200D')/d for d in os.listdir('results/bistable200D') if ('experiment' in d) and (d.endswith('_gmmratio0.0'))] +
            [
            # 'results/hypercube_2D/experiment',
            # 'results/hypercube_5D/experiment',
            # 'results/hypercube_10D/experiment',
            # 'results/hypercube_10D/experiment_width400',
            # 'results/hypercube_10D/experiment_width1000'
            # 'results/finkelstein_fontolan_RNN/experiment_DNN_layers6_Tanh_hidden1000_output7'
            # 'results/finkelstein_fontolan_RNN/experiment_ResidualMLP_layers15_hidden671_output7_lr0.001_std',
            # 'results/finkelstein_fontolan_RNN/experiment_ResidualMLP_layers25_hidden671_output7_lr0.001_std',

            # 'results/bistable20D/experiment_Transformer_dmodel16_layers4_nhead4_out7_poolingattention_lr0.01_std',
            # 'results/bistable20D/experiment_Transformer_dmodel32_layers2_nhead2_out7_poolingattention_lr0.01_std',
            # 'results/bistable20D/experiment_Transformer_dmodel32_layers4_nhead4_out7_poolingattention_lr0.001_std'


            # 'results/finkelstein_fontolan_RNN/experiment_ResidualMLP_layers40_hidden668_output7_lr0.001_std',
            # 'results/finkelstein_fontolan_RNN/experiment_details',
            # 'results/bistable100D/experiment_ResidualMLP_layers10_hidden100_output7',
            # 'results/bistable100D/experiment_ResidualMLP_layers15_hidden100_output7',
            # 'results/bistable100D/experiment_ResidualMLP_layers15_hidden100_output7_lr0.001',
            # 'results/bistable100D/experiment_ResidualMLP_layers20_hidden100_output7',
            # 'results/bistable100D/experiment_ResidualMLP_layers25_hidden100_output7'
            # 'results/bistable600D/experiment_ResidualMLP_layers15_hidden600_output7_lr0.001_std'
        ],  # Replace with your list of log directories
        # 'metrics': ['Loss/Total'],  # List of metrics to plot, e.g. ['loss1', 'loss2'] or None for all
        'metrics': [
            'Loss/Total',
            # 'Loss/NormalisedLoss_Dist_0',
            # 'Loss/NormalisedLoss_Dist_1',
            # 'Loss/NormalisedLoss_Dist_2'
        ],
        # 'save_path': 'test_plots/finkelstein_fontolan_RNN_losses.png',  # Path to save figure, e.g. 'losses.png' or None to display
        # 'save_path': 'test_plots/bistable600D_losses.png',
        'save_path': 'test_plots/bistable200D_losses.png',
        'figsize': (6, 6),  # Figure size in inches (width, height)
        'smooth': 10,  # Default window size for smoothing (0 means no smoothing)
    }
    
    # Command-line argument parsing (overrides CONFIG if provided)
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Plot training losses from CSV logs')
        parser.add_argument('log_dirs', nargs='+', type=str, help='Directories containing the CSV log files')
        parser.add_argument('--metrics', nargs='+', help='Specific metrics to plot (default: all in Loss/)')
        parser.add_argument('--save', type=str, help='Path to save the figure (default: display plot)')
        parser.add_argument('--figsize', nargs=2, type=int, default=[12, 6], help='Figure size (width height)')
        parser.add_argument('--smooth', type=int, default=0, help='Window size for smoothing (default: 0, no smoothing)')
        
        args = parser.parse_args()
        
        # Override CONFIG with command-line arguments
        CONFIG['log_dirs'] = args.log_dirs
        if args.metrics is not None:
            CONFIG['metrics'] = args.metrics
        if args.save is not None:
            CONFIG['save_path'] = args.save
        if args.figsize is not None:
            CONFIG['figsize'] = tuple(args.figsize)
        if args.smooth is not None:
            CONFIG['smooth'] = args.smooth
    
    # Remove the 'smooth' key from CONFIG before calling the function
    config_for_plotting = {k: v for k, v in CONFIG.items() if k != 'smooth'}

    # Run the plotting function with the final configuration
    load_and_plot_losses(**config_for_plotting) 