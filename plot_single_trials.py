import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats import norm
from scipy import signal
import os

def get_spk_vec(spks, t, t_start, t_end):
    """Python version of getSpkVec.m"""
    spk_vec = np.zeros(len(t))
    zero_ind = np.where(t == 0)[0]
    
    if len(spks) == 0:
        return spk_vec
    
    good_spikes = (spks > t_start) & (spks < t_end)
    
    if np.sum(good_spikes) > 0:
        spk_idx = np.int32(spks[good_spikes] * 1000) + zero_ind
        spk_vec[spk_idx] = 1
        
        if t_start != t[0]:
            spk_vec[:np.int32(t_start * 1000) + zero_ind] = np.nan
        
        if t_end != t[-1]:
            spk_vec[np.int32(t_end * 1000) + zero_ind:] = np.nan
    
    return spk_vec

def get_single_trial_avg_dots_task(d, alpha, ex_sacc, ex_dots, max_dur):
    """Python version of getSingleTrialAvg_dotsTask.m"""
    n_trials = len(d)
    dt = 0.001
    
    dots_t = np.arange(-0.4, max_dur + 0.5, dt)
    sacc_t = np.round(np.arange(-max_dur, 0.4, dt) / dt) * dt
    
    zero_dots = np.where(dots_t == 0)[0][0]
    zero_sacc = np.where(sacc_t == 0)[0][0]
    
    dots_on = np.array([trial['dotsOn'][0, 0] for trial in d])
    sacc_on = np.array([trial['saccadeDetected'][0, 0] for trial in d])
    duration = sacc_on - dots_on
    
    # Preallocate
    fr_vec_dots = [None] * n_trials
    fr_vec_sacc = [None] * n_trials
    t_end_sacc = sacc_t[-1]
    t_start_dots = dots_t[0]
    
    # Loop through trials
    for i in range(n_trials):
        n_units = len(d[i]['spCellPop'][0, 0])
        trial_dur = np.round(duration[i], 3)
        t_start_sacc = -trial_dur + ex_dots
        t_end_dots = trial_dur - ex_sacc
        
        spk_vec_dots = np.zeros((n_units, len(dots_t)))
        spk_vec_sacc = np.zeros((n_units, len(sacc_t)))
        
        for j in range(n_units):
            spk = d[i]['spCellPop'][0, 0][j].flatten()
            spk_dots = np.round(spk - dots_on[i], 3)
            spk_sacc = np.round(spk - sacc_on[i], 3)
            
            spk_vec_dots[j, :] = get_spk_vec(spk_dots, dots_t, t_start_dots, t_end_dots)
            spk_vec_sacc[j, :] = get_spk_vec(spk_sacc, sacc_t, t_start_sacc, t_end_sacc)
        
        # Convolve with alpha kernel
        fr_vec_dots[i] = signal.convolve2d(spk_vec_dots, alpha.reshape(1, -1), mode='same')
        fr_vec_dots[i] = fr_vec_dots[i][:, :len(dots_t)]
        fr_vec_dots[i][:, int(t_end_dots * 1000) + zero_dots:] = np.nan
        
        fr_vec_sacc[i] = signal.convolve2d(spk_vec_sacc, alpha.reshape(1, -1), mode='same')
        fr_vec_sacc[i] = fr_vec_sacc[i][:, :len(sacc_t)]
        fr_vec_sacc[i][:, :int(t_start_sacc * 1000) + zero_sacc] = np.nan
    
    dots_t = dots_t - dt * len(alpha) / 2
    sacc_t = sacc_t - dt * len(alpha) / 2
    
    return fr_vec_dots, fr_vec_sacc, dots_t, sacc_t

def plot_single_trials_paper(d, unit_idx=None):
    """Python version of plotSingleTrials_paper.m"""
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Get completed dots trials
    complete = np.array([trial['complete'][0, 0] for trial in d])
    coh = np.array([trial['coh'][0, 0] for trial in d])
    duration = np.array([trial['duration'][0, 0] for trial in d])
    trial_type = np.array([trial['trialType'][0, 0] for trial in d])
    
    # Filter trials
    l1 = complete & ~np.isnan(coh) & (duration > 0) & (trial_type == 20)
    d_filtered = [d[i] for i in range(len(d)) if l1[i]]
    
    u_coh = np.unique([trial['coh'][0, 0] for trial in d_filtered])
    n_coh = len(u_coh)
    
    max_dur = max([trial['duration'][0, 0] for trial in d_filtered])
    ex_dots = 0
    ex_sacc = 0.07
    alpha = norm.pdf(np.arange(0.001, 0.101, 0.001), 0.05, 0.025)
    
    fr_vec_dots, fr_vec_sacc, dots_t, sacc_t = get_single_trial_avg_dots_task(
        d_filtered, alpha, ex_sacc, ex_dots, max_dur
    )
    
    # Loop through coherences
    dots = {}
    sacc = {}
    
    for ii in range(n_coh):
        coh_values = np.array([trial['coh'][0, 0] for trial in d_filtered])
        l = coh_values == u_coh[ii]
        
        fv_dots = [fr_vec_dots[i] for i in range(len(fr_vec_dots)) if l[i]]
        fv_sacc = [fr_vec_sacc[i] for i in range(len(fr_vec_sacc)) if l[i]]
        
        dots[ii] = np.zeros((len(fv_dots), fv_dots[0].shape[1]))
        sacc[ii] = np.zeros((len(fv_sacc), fv_sacc[0].shape[1]))
        
        for i in range(len(fv_dots)):
            if unit_idx is None:
                dots[ii][i, :] = np.nanmean(fv_dots[i], axis=0)
                sacc[ii][i, :] = np.nanmean(fv_sacc[i], axis=0)
            else:
                dots[ii][i, :] = np.nanmean(fv_dots[i][unit_idx, :], axis=0)
                sacc[ii][i, :] = np.nanmean(fv_sacc[i][unit_idx, :], axis=0)
        
        # Subtract baseline activity
        l_t = (dots_t > 0.18) & (dots_t < 0.2)
        m = np.nanmean(dots[ii][:, l_t], axis=1)
        dots[ii] = dots[ii] - m.reshape(-1, 1)
        
        # Select relevant trials
        d_g = [d_filtered[i] for i in range(len(d_filtered)) if l[i]]
        idx = np.ones(len(d_g), dtype=bool)
        
        choice = np.array([trial['choice'][0, 0] for trial in d_g])
        tin = choice == 0
        tout = choice == 1
        
        # Plot saccade-aligned activity
        plt.subplot(2, n_coh, ii + n_coh + 1)
        for j in range(len(sacc[ii])):
            if tout[j] and idx[j]:
                plt.plot(sacc_t, sacc[ii][j, :], 'r', linewidth=0.2, alpha=0.5)
            elif tin[j] and idx[j]:
                plt.plot(sacc_t, sacc[ii][j, :], 'b', linewidth=0.2, alpha=0.5)
        
        plt.ylim([0, 80])
        plt.xlim([-1, 0.2])
        plt.axvline(x=0, color='k', linewidth=1)
        plt.title(f'Coh: {u_coh[ii]}')
        
        # Plot dots-aligned activity
        plt.subplot(2, n_coh, ii + 1)
        for j in range(len(dots[ii])):
            if tout[j] and idx[j]:
                plt.plot(dots_t, dots[ii][j, :], 'r', linewidth=0.2, alpha=0.5)
            elif tin[j] and idx[j]:
                plt.plot(dots_t, dots[ii][j, :], 'b', linewidth=0.2, alpha=0.5)
        
        plt.ylim([-20, 50])
        plt.xlim([0.2, 0.8])
        plt.title(f'Coh: {u_coh[ii]}')
    
    # Add labels
    plt.subplot(2, n_coh, 1)
    plt.xlabel('Time from motion onset (s)')
    plt.ylabel('Firing rate (spk/s)')
    
    plt.subplot(2, n_coh, n_coh + 1)
    plt.xlabel('Time from saccade (s)')
    plt.ylabel('Firing rate (spk/s)')
    
    plt.tight_layout()
    
    return {
        'dots_t': dots_t,
        'sacc_t': sacc_t,
        'fr_vec_dots': fr_vec_dots,
        'fr_vec_sacc': fr_vec_sacc,
        'dots_mean': dots,
        'sacc_mean': sacc
    }

def main():
    """Load data and create the plot"""
    # Load the LIP data
    lip_file = "/Users/kabir/Documents/datasets/7946011/Stine et al_2023_Code/Figure 3/Data/LIP_example_session.mat"
    lip_data = scipy.io.loadmat(lip_file)
    
    # Extract the trial data - d is already a structured array
    d = lip_data['d']  # This is the structured array containing all trial data
    
    # Convert to list of dictionaries for easier handling
    d_list = []
    for i in range(d.shape[0]):
        trial_dict = {}
        for field_name in d.dtype.names:
            trial_dict[field_name] = d[field_name][i, 0]
        d_list.append(trial_dict)
    
    print(f"Loaded {len(d_list)} trials")
    print(f"Fields: {list(d_list[0].keys())}")
    
    # Create the plot
    result = plot_single_trials_paper(d_list, unit_idx=None)
    
    plt.show()
    
    return result

if __name__ == "__main__":
    result = main() 