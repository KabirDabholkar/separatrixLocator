import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

def inspect_mat_file(file_path):
    """Load and inspect a MATLAB file, printing its structure and basic info."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    try:
        # Load the MATLAB file
        mat_data = scipy.io.loadmat(file_path)
        
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        print(f"Number of variables: {len(mat_data)}")
        
        # Print all variable names (excluding MATLAB metadata)
        variables = [key for key in mat_data.keys() if not key.startswith('__')]
        print(f"Variables: {variables}")
        
        # Inspect each variable
        for var_name in variables:
            var_data = mat_data[var_name]
            print(f"\nVariable: {var_name}")
            print(f"  Type: {type(var_data)}")
            print(f"  Shape: {var_data.shape}")
            print(f"  Data type: {var_data.dtype}")
            
            # If it's a structured array, show field names
            if var_data.dtype.names is not None:
                print(f"  Fields: {var_data.dtype.names}")
                for field in var_data.dtype.names:
                    field_data = var_data[field][0, 0] if var_data.shape == (1, 1) else var_data[field]
                    print(f"    {field}: {type(field_data)}, shape: {field_data.shape}")
            
            # For large arrays, show statistics
            if var_data.size > 20 and hasattr(var_data, 'flatten'):
                flat_data = var_data.flatten()
                if flat_data.dtype.kind in 'fc':  # float or complex
                    print(f"  Min: {np.min(flat_data):.4f}")
                    print(f"  Max: {np.max(flat_data):.4f}")
                    print(f"  Mean: {np.mean(flat_data):.4f}")
                    print(f"  Std: {np.std(flat_data):.4f}")
        
        return mat_data
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_neural_data(lip_data):
    """Extract and analyze the neural data from the LIP session."""
    print(f"\n{'='*60}")
    print("NEURAL DATA ANALYSIS")
    print(f"{'='*60}")
    
    # Extract the main data structure
    d = lip_data['d'][0, 0]  # This is the structured array containing all trial data
    
    # Extract neural data fields
    spCell = d['spCell']  # Spike cell data
    unitIdx = d['unitIdx']  # Unit indices
    spCellPop = d['spCellPop']  # Population spike data
    chGood = d['chGood']  # Good channels
    
    print(f"Neural data fields found:")
    print(f"  spCell: shape {spCell.shape}")
    print(f"  unitIdx: shape {unitIdx.shape}")
    print(f"  spCellPop: shape {spCellPop.shape}")
    print(f"  chGood: shape {chGood.shape}")
    
    # Analyze spCell (individual unit spike data)
    print(f"\nspCell analysis:")
    print(f"  Type: {type(spCell)}")
    print(f"  Shape: {spCell.shape}")
    
    # Look at a few trials to understand the structure
    for i in range(min(3, spCell.shape[0])):
        trial_spikes = spCell[i, 0]
        print(f"  Trial {i}: type {type(trial_spikes)}, shape {trial_spikes.shape}")
        
        # If it's a structured array, explore its fields
        if hasattr(trial_spikes, 'dtype') and trial_spikes.dtype.names is not None:
            print(f"    Fields: {trial_spikes.dtype.names}")
            for field in trial_spikes.dtype.names:
                field_data = trial_spikes[field]
                print(f"      {field}: shape {field_data.shape}")
    
    # Analyze spCellPop (population data)
    print(f"\nspCellPop analysis:")
    print(f"  Type: {type(spCellPop)}")
    print(f"  Shape: {spCellPop.shape}")
    
    # Look at a few trials
    for i in range(min(3, spCellPop.shape[0])):
        trial_pop = spCellPop[i, 0]
        print(f"  Trial {i}: type {type(trial_pop)}, shape {trial_pop.shape}")
        
        if hasattr(trial_pop, 'dtype') and trial_pop.dtype.names is not None:
            print(f"    Fields: {trial_pop.dtype.names}")
            for field in trial_pop.dtype.names:
                field_data = trial_pop[field]
                print(f"      {field}: shape {field_data.shape}")
    
    return d

def main():
    # Path to the specific LIP file
    lip_file = "/Users/kabir/Documents/datasets/7946011/Stine et al_2023_Code/Figure 3/Data/LIP_example_session.mat"
    
    print(f"Loading LIP_example_session.mat...")
    
    # Load and inspect the LIP file
    lip_data = inspect_mat_file(lip_file)
    
    if lip_data is not None:
        # Analyze neural data specifically
        trial_data = analyze_neural_data(lip_data)
    
    return lip_data

if __name__ == "__main__":
    data = main()
