import os

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import argparse
import sys
import numpy as np
import re

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def load_and_plot_losses(log_dirs, metrics=None, save_path=None, figsize=(12, 6), sort_by_eigenvalue=False):
    """
    Load CSV logs from multiple directories and create plots for specified metrics across all models.
    
    Args:
        log_dirs (list of str): List of directories containing the CSV log files
        metrics (list, optional): List of metrics to plot. If None, plots all metrics in directory
        save_path (str, optional): Path to save the figure. If None, displays the plot
        figsize (tuple): Figure size in inches
        sort_by_eigenvalue (bool): Whether to sort and color by eigenvalue numbers in directory names
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
    
    if sort_by_eigenvalue:
        # Extract dimensions and sort directories
        dir_eigenvalues = []
        for log_dir in log_dirs:
            match = re.search(r'Vanilla(\d+)', str(log_dir))
            if match:
                dir_eigenvalues.append(float(match.group(1)))
            else:
                dir_eigenvalues.append(float('inf'))
        
        sorted_indices = np.argsort(dir_eigenvalues)
        log_dirs = [log_dirs[i] for i in sorted_indices]
        dir_eigenvalues = [dir_eigenvalues[i] for i in sorted_indices]
        
        # Create discrete color map for dimensions
        valid_eigenvalues = [e for e in dir_eigenvalues if e != float('inf')]
        if valid_eigenvalues:
            # Use discrete colors for each dimension
            unique_dims = sorted(set(valid_eigenvalues))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_dims)))
            dim_to_color = dict(zip(unique_dims, colors))
    else:
        # Set up a regular color palette for directories
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
                    if sort_by_eigenvalue:
                        eigenvalue = dir_eigenvalues[dir_idx]
                        label = str(eigenvalue) if idx == 0 else None
                        color = dim_to_color.get(eigenvalue, 'gray') # Use discrete color or gray
                    else:
                        label = str(log_dir) if idx == 0 else None
                        color = dir_colors[dir_idx]
                    
                    ax.plot(
                        model_data['step'], model_data['value'],
                        label=label,
                        color=color,
                        alpha=0.6,
                        lw=1
                    )
                    
                    min_val = min(min_val, model_data['value'].min())
                    max_val = max(max_val, model_data['value'].max())
                
                if ax_id == 0:
                    ax.set_xlabel('Step')
                ax.set_ylabel(metric)
                
                # Use log scale if values span multiple orders of magnitude
                if max_val / min_val > 100:
                    ax.set_yscale('log')
                else:
                    ax.set_yscale('log')

            except Exception as e:
                print(f"Error plotting {metric} from {log_dir}: {e}")
    
    # Add legend and colorbar
    legend_title = 'Dimensions' if sort_by_eigenvalue else 'Log Directories'
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
    # Note: No colorbar needed for discrete dimensions
    
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

def plot_final_error_vs_eigenvalue(log_dirs, metric='Loss/Total', save_path=None, figsize=(8, 6)):
    """
    Plot the final training error as a function of eigenvalue.
    
    Args:
        log_dirs (list of str): List of directories containing the CSV log files
        metric (str): The metric to plot (default: 'Loss/Total')
        save_path (str, optional): Path to save the figure. If None, displays the plot
        figsize (tuple): Figure size in inches
    """
    # Convert log_dirs to Path objects
    log_dirs = [Path(log_dir) for log_dir in log_dirs]
    
    # Extract eigenvalues and final errors
    eigenvalues = []
    final_errors = []
    
    for log_dir in log_dirs:
        try:
            # Extract dimension from directory name (Vanilla32, Vanilla64, etc.)
            match = re.search(r'Vanilla(\d+)', str(log_dir))
            if not match:
                continue
            eigenvalue = float(match.group(1))
            
            # Load the CSV file for this metric
            metric_path = metric if "/" in metric else f"Loss/{metric}"
            csv_path = log_dir / f"{metric_path}.csv"
            loss_df = pd.read_csv(csv_path)
            
            # Get the final error (last value in the dataframe)
            final_error = loss_df['value'].iloc[-1]
            
            eigenvalues.append(eigenvalue)
            final_errors.append(final_error)
            
        except Exception as e:
            print(f"Error processing {log_dir}: {e}")
    
    if not eigenvalues:
        print("No valid data found to plot")
        return
    
    # Sort by eigenvalue
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = [eigenvalues[i] for i in sorted_indices]
    final_errors = [final_errors[i] for i in sorted_indices]
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(eigenvalues, final_errors, 'o-', c='blue', alpha=0.6, markersize=6)
    
    plt.xlabel(r'Dimension')
    plt.ylabel(f'Final {metric} Error')
    plt.title(f'Final Training Error vs Dimension')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Configuration variables - modify these directly in the script
    # =========================================================
    # dynamical_function = 'finkelstein_fontolan_RNN_multiscale_on_separatrix_speed40'
    # dynamical_function = 'bistable1D'
    # common_pattern = 'experiment_AdditiveRBFResNet_hidden400_output1_numKernels10_numlayers20_lr0.001_noscheduler_l20.001_batch5000_epochs1000_learn_koopman_eig.shuffle_normaliser_gmmratio0.0'

    # dynamical_function = '2bitFlipFlop_GRU3'
    # common_pattern = 'experiment_AdditiveRBFResNet_hidden400_output1_numKernels10_numlayers20_lr0.0001_noscheduler_l21e-05_batch1000_epochs1000_learn_koopman_eig.shuffle_normaliser_gmmratio0.0'

    pathlist = [
        'results/1bitFlipFlop_long_isotropic_Vanilla32/experiment_AdditiveRBFResNet_hidden400_output1_numKernels10_numlayers20_lr0.0001_noscheduler_l21e-05_batch1000_epochs1000_learn_koopman_eig.shuffle_normaliser_gmmratio0.0_eigenvalue1.0_AdamW_balanceloss0.05',
        'results/1bitFlipFlop_long_isotropic_Vanilla64/experiment_AdditiveRBFResNet_hidden400_output1_numKernels10_numlayers20_lr0.0001_noscheduler_l21e-05_batch1000_epochs1000_learn_koopman_eig.shuffle_normaliser_gmmratio0.0_eigenvalue1.0_AdamW_balanceloss0.05',
        'results/1bitFlipFlop_long_isotropic_Vanilla128/experiment_AdditiveRBFResNet_hidden400_output1_numKernels10_numlayers20_lr0.0001_noscheduler_l21e-05_batch1000_epochs1000_learn_koopman_eig.shuffle_normaliser_gmmratio0.0_eigenvalue1.0_AdamW_balanceloss0.05',
        'results/1bitFlipFlop_long_isotropic_Vanilla256/experiment_AdditiveRBFResNet_hidden400_output1_numKernels10_numlayers20_lr0.0001_noscheduler_l21e-05_batch1000_epochs1000_learn_koopman_eig.shuffle_normaliser_gmmratio0.0_eigenvalue1.0_AdamW_balanceloss0.05',
        'results/1bitFlipFlop_long_isotropic_Vanilla512/experiment_AdditiveRBFResNet_hidden550_output1_numKernels10_numlayers20_lr0.0001_noscheduler_l21e-05_batch1000_epochs1000_learn_koopman_eig.shuffle_normaliser_gmmratio0.0_eigenvalue1.0_AdamW_balanceloss0.05',
    ]
    dynamical_function = '1bitflipflop'

    CONFIG = {
        'log_dirs':
            pathlist + #and (d.endswith('_gmmratio0.0'))
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
        'save_path': f'test_plots/1bitflipflop_dimensions_losses.png',  # Path to save figure, e.g. 'losses.png' or None to display
        # 'save_path': 'test_plots/bistable600D_losses.png',
        # 'save_path': 'test_plots/bistable200D_losses.png',
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
        parser.add_argument('--sort_by_eigenvalue', action='store_true', help='Sort by eigenvalue (default: False)')
        
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
    
    # Add sort_by_eigenvalue=True for this run
    config_for_plotting['sort_by_eigenvalue'] = True

    # Run the plotting function with the final configuration
    load_and_plot_losses(**config_for_plotting)
    
    # Create the final error vs eigenvalue plot
    final_error_save_path = f'test_plots/{dynamical_function}_final_error_vs_eigenvalue.pdf'
    plot_final_error_vs_eigenvalue(
        log_dirs=CONFIG['log_dirs'],
        metric='Loss/Total',
        save_path=final_error_save_path,
        figsize=(4, 3)
    )