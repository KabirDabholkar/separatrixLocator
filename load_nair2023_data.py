import pynwb
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_and_plot_neural_data(nwb_file_path, subject_name):
    """Load NWB file and create neural activity plots"""
    print(f"\n{'='*60}")
    print(f"Loading dataset: {subject_name}")
    print(f"File: {nwb_file_path}")
    print(f"{'='*60}")
    
    # Load the NWB file
    nwb_file = pynwb.NWBHDF5IO(nwb_file_path, mode='r')
    nwb = nwb_file.read()
    
    print(f"File identifier: {nwb.identifier}")
    print(f"Session description: {nwb.session_description}")
    print(f"Session start time: {nwb.session_start_time}")
    
    # Load ophys data
    ophys_module = nwb.processing['ophys']
    neural_trace = ophys_module['NeuralTrace']
    
    print(f"\n=== NEURAL TRACE DATA ===")
    print(f"Data shape: {neural_trace.data.shape}")
    print(f"Sampling rate: {neural_trace.rate} Hz")
    print(f"Starting time: {neural_trace.starting_time}")
    print(f"Unit: {neural_trace.unit}")
    print(f"Description: {neural_trace.description}")
    
    # Load the actual data
    neural_data = neural_trace.data[:]  # Load all data into memory
    print(f"Loaded data shape: {neural_data.shape}")
    
    # Calculate time axis
    total_time = neural_data.shape[0] / neural_trace.rate
    time_axis = np.linspace(0, total_time, neural_data.shape[0])
    
    print(f"Total recording time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot neural activity as heatmap
    plt.subplot(2, 1, 1)
    im = plt.imshow(neural_data.T, aspect='auto', cmap='viridis', 
                    extent=[0, total_time, 0, neural_data.shape[1]])
    plt.colorbar(im, label='Neural Activity')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Neuron ID')
    plt.title(f'Neural Activity Heatmap - {subject_name}')
    
    # Plot mean activity over time
    plt.subplot(2, 1, 2)
    mean_activity = np.mean(neural_data, axis=1)
    plt.plot(time_axis, mean_activity, 'b-', linewidth=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mean Neural Activity')
    plt.title(f'Population Mean Activity Over Time - {subject_name}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create test_plots directory if it doesn't exist
    os.makedirs('test_plots', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'test_plots/neural_activity_plots_{subject_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\n=== DATA STATISTICS ===")
    print(f"Mean activity: {np.mean(neural_data):.4f}")
    print(f"Std activity: {np.std(neural_data):.4f}")
    print(f"Min activity: {np.min(neural_data):.4f}")
    print(f"Max activity: {np.max(neural_data):.4f}")
    
    # Close the file
    nwb_file.close()
    
    return neural_data, total_time, time_axis

def perform_pca_analysis(neural_data, subject_name, time_axis):
    """Perform PCA analysis on neural data"""
    print(f"\n=== PCA ANALYSIS - {subject_name} ===")
    
    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    neural_data_scaled = scaler.fit_transform(neural_data)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(neural_data_scaled)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"Number of components: {len(explained_variance_ratio)}")
    print(f"Variance explained by first 5 components:")
    for i in range(min(5, len(explained_variance_ratio))):
        print(f"  PC{i+1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:.2f}%)")
    print(f"Cumulative variance (first 5): {cumulative_variance[4]:.4f} ({cumulative_variance[4]*100:.2f}%)")
    
    # Create PCA plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Explained variance
    axes[0, 0].plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-', markersize=3)
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title(f'Explained Variance - {subject_name}')
    axes[0, 0].set_xlim(1, min(20, len(explained_variance_ratio)))
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative explained variance
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', markersize=3)
    axes[0, 1].set_xlabel('Principal Component')
    axes[0, 1].set_ylabel('Cumulative Explained Variance')
    axes[0, 1].set_title(f'Cumulative Explained Variance - {subject_name}')
    axes[0, 1].set_xlim(1, min(20, len(cumulative_variance)))
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: First two principal components over time
    axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], c=time_axis, cmap='viridis', alpha=0.6, s=1)
    axes[1, 0].set_xlabel('PC1')
    axes[1, 0].set_ylabel('PC2')
    axes[1, 0].set_title(f'PC1 vs PC2 (colored by time) - {subject_name}')
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Time (seconds)')
    
    # Plot 4: First three principal components over time
    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
    scatter = ax_3d.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], 
                           c=time_axis, cmap='viridis', alpha=0.6, s=1)
    ax_3d.set_xlabel('PC1')
    ax_3d.set_ylabel('PC2')
    ax_3d.set_zlabel('PC3')
    ax_3d.set_title(f'PC1 vs PC2 vs PC3 - {subject_name}')
    
    plt.tight_layout()
    plt.savefig(f'test_plots/pca_analysis_{subject_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return pca, pca_result, explained_variance_ratio

# Define the datasets
datasets = [
    {
        'path': "/Users/kabir/Documents/datasets/001037/sub-M1/sub-M1_ses-20230323_behavior.nwb",
        'name': 'sub-M1'
    },
    {
        'path': "/Users/kabir/Documents/datasets/001037/sub-M30L/sub-M30L_ses-20230517_behavior.nwb",
        'name': 'sub-M30L'
    }
]

# Load and plot each dataset
results = {}
pca_results = {}

for dataset in datasets:
    try:
        neural_data, total_time, time_axis = load_and_plot_neural_data(dataset['path'], dataset['name'])
        results[dataset['name']] = {
            'neural_data': neural_data,
            'total_time': total_time,
            'num_neurons': neural_data.shape[1],
            'num_timepoints': neural_data.shape[0],
            'time_axis': time_axis
        }
        
        # Perform PCA analysis
        pca, pca_result, explained_variance = perform_pca_analysis(neural_data, dataset['name'], time_axis)
        pca_results[dataset['name']] = {
            'pca': pca,
            'pca_result': pca_result,
            'explained_variance': explained_variance
        }
        
    except Exception as e:
        print(f"Error loading {dataset['name']}: {e}")

# Create comparison plot
if len(results) == 2:
    print(f"\n{'='*60}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*60}")
    
    # Compare explained variance
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for name in results.keys():
        plt.plot(range(1, len(pca_results[name]['explained_variance']) + 1), 
                pca_results[name]['explained_variance'], 'o-', markersize=3, label=name)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Comparison')
    plt.legend()
    plt.xlim(1, 20)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    for name in results.keys():
        cumulative = np.cumsum(pca_results[name]['explained_variance'])
        plt.plot(range(1, len(cumulative) + 1), cumulative, 'o-', markersize=3, label=name)
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance Comparison')
    plt.legend()
    plt.xlim(1, 20)
    plt.grid(True, alpha=0.3)
    
    # Compare PC1 trajectories
    plt.subplot(2, 2, 3)
    for name in results.keys():
        plt.plot(results[name]['time_axis'], pca_results[name]['pca_result'][:, 0], 
                alpha=0.7, label=f'{name} PC1')
    plt.xlabel('Time (seconds)')
    plt.ylabel('PC1')
    plt.title('PC1 Trajectories Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare PC2 trajectories
    plt.subplot(2, 2, 4)
    for name in results.keys():
        plt.plot(results[name]['time_axis'], pca_results[name]['pca_result'][:, 1], 
                alpha=0.7, label=f'{name} PC2')
    plt.xlabel('Time (seconds)')
    plt.ylabel('PC2')
    plt.title('PC2 Trajectories Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_plots/pca_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Print summary comparison
print(f"\n{'='*60}")
print("SUMMARY COMPARISON")
print(f"{'='*60}")
for name, result in results.items():
    print(f"\n{name}:")
    print(f"  Neurons: {result['num_neurons']}")
    print(f"  Timepoints: {result['num_timepoints']}")
    print(f"  Duration: {result['total_time']:.2f} seconds ({result['total_time']/60:.2f} minutes)")
    print(f"  Mean activity: {np.mean(result['neural_data']):.4f}")
    print(f"  Std activity: {np.std(result['neural_data']):.4f}")
    if name in pca_results:
        print(f"  PC1 variance: {pca_results[name]['explained_variance'][0]:.4f} ({pca_results[name]['explained_variance'][0]*100:.2f}%)")
        print(f"  PC2 variance: {pca_results[name]['explained_variance'][1]:.4f} ({pca_results[name]['explained_variance'][1]*100:.2f}%)")

print(f"\nPlots saved to test_plots/ directory")

