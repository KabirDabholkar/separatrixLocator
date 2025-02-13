import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Load trained RNN data
input_file_path = "RNN_ALM_gating/input_data/"
output_file_path = "RNN_ALM_gating/output_data/all_struct.mat"

file_name = "input_data_wramp"  # Load input data
data = sio.loadmat(f"{input_file_path}{file_name}.mat")

# Load parameters
params_file_name = "params_data_wramp"
params = sio.loadmat(f"{input_file_path}{params_file_name}.mat")

N = int(params['params']['N'][0][0])

params_dict = {
    "t_ramp_start": 500,
    "t_stim_interval": np.arange(1000, 1400),  # Adjusted for Python 0-indexing
    "T_test": 5000,
    "ramp_dur": 3000,
    "sigma_noise_cd": 100. / N,
    "ramp_mean": 1.0,
    "ramp_sigma": 0.05,
    "amp_stim": 1,
    "sigma_stim": 0.1,
    "endpoint": 3500,
    "amp_chirp": 1,
    "dt": params['params']['dt'][0][0][0,0],
    "tau": params['params']['tau'][0][0][0,0],
    "f0": params['params']['f0'][0][0][0,0],
    "beta0": float(params['params']['beta0'][0][0]),
    "theta0": float(params['params']['theta0'][0][0]),
    "M": params['params']['M'][0,0][:N, :N],
    "h": params['params']['h'][0,0][:N].flatten(),
    "sigma_noise": params['params']['tau_noise'][0][0][0,0],
    "ramp_train": params['params']['ramp_train'],
    "fr_smooth": params['params']['fr_smooth'][0][0],
    "ramp_bsln": params['params']['ramp_bsln'][0][0],
    "eff_dt": params['params']['eff_dt'][0][0],
    "des_out_left": params['params']['des_out_left'],
    "des_out_right": params['params']['des_out_right'],
    "des_r_left_norm": params['params']['des_r_left_norm'],
    "des_r_right_norm": params['params']['des_r_right_norm'],
}

cd_span = 20 #0
N_trials_cd = 2 #00

# print("fr_smooth",params_dict["fr_smooth"])

# Function to compute coding direction
# def f_cd(N_trials_cd, params, cd_span):
#     print("Running trials to compute the CD mode...")
#
#     simtime_test = np.arange(0, params["T_test"], params["dt"])
#     simtime_test_len = len(simtime_test)
#
#     rp_vec_nd = []
#     for i in range(N_trials_cd):
#         print(f"Trial # {i + 1}")
#         r = np.zeros((N, simtime_test_len))  # Placeholder firing rates
#         print(r.shape)
#         rp_vec_nd.append(r)
#
#     RNN_fr_cd = np.stack(rp_vec_nd, axis=-1)
#     return {"RNN_fr_cd": RNN_fr_cd}


# def f_cd(N_trials_cd, p, cd_span):
#     print("Running a few trials to compute the CD mode")
#
#     dt = p["dt"]
#     tau = p["tau"]
#     f0 = p["f0"]
#     beta0 = p["beta0"]
#     theta0 = p["theta0"]
#     M = p["M"]
#     h = p["h"]
#     amp_stim = p["amp_stim"]
#     sigma_stim = p["sigma_stim"]
#     ramp_mean = p["ramp_mean"]
#     ramp_sigma = p["ramp_sigma"]
#     ramp_dur = p["ramp_dur"]
#     sigma_noise = p["sigma_noise_cd"]
#     T_test = p["T_test"]
#     endpoint = p["endpoint"]
#
#     simtime_test = np.arange(0, T_test, dt)
#     simtime_test_len = len(simtime_test)
#
#     noise_sigma_eff = np.sqrt(dt) * sigma_noise / tau
#
#     RNN_fr_cd = np.zeros((N, simtime_test_len, N_trials_cd))
#
#     for i in range(N_trials_cd):
#         print(f"Trial # {i + 1}")
#
#         x = np.zeros(N)
#         r = f0 / (1.0 + np.exp(-beta0 * (x - theta0)))
#         r = r.flatten()  # Convert (1, N) -> (N,)
#
#         for t in range(simtime_test_len):
#             x += (-x + M @ r + h) * dt + noise_sigma_eff * np.random.randn(N)
#             r = f0 / (1.0 + np.exp(-beta0 * (x - theta0)))
#             RNN_fr_cd[:, t, i] = r
#
#     return {"RNN_fr_cd": RNN_fr_cd}

def f_cd(N_trials_cd, p, cd_span):
    print("Running trials to compute the CD mode with ramping input and chirp")
    dt, tau, f0, beta0, theta0, M, h = p['dt'], p['tau'], p['f0'], p['beta0'], p['theta0'], p['M'], p['h']
    noise_sigma_eff = np.sqrt(dt) * p['sigma_noise'] / tau
    simtime_test = np.arange(0, p['T_test'], dt)
    simtime_len = len(simtime_test)

    inp_chirp_temp = np.zeros(simtime_len)
    inp_chirp_temp[int(500/dt):int(650/dt)] = 1
    inp_chirp_temp[int(1350/dt):int(1500/dt)] = 1
    inp_chirp_temp = moving_average(inp_chirp_temp, int(p['fr_smooth'])) #gaussian_filter1d(inp_chirp_temp, p['fr_smooth'])

    RNN_fr_cd = np.zeros((N, simtime_len, N_trials_cd))
    r_in_cd = np.zeros((3, simtime_len))
    r_in_cd[0, :] = p['amp_chirp'] * inp_chirp_temp

    x_init = np.mean([np.mean(params['params']['des_out_left'][:, :10], axis=1), np.mean(params['params']['des_out_right'][:, :10], axis=1)], axis=0)
    init_sigma = 0.05
    stm_trials = np.concatenate([np.zeros(N_trials_cd // 2), p['amp_stim'] * np.ones(N_trials_cd // 2)])

    for i in range(N_trials_cd):
        print(f"Trial #{i+1}")
        x = x_init * (1 + init_sigma * np.random.randn(N))
        r = f0 / (1.0 + np.exp(-beta0 * (x - theta0)))
        for t in range(simtime_len):
            ramp_input = np.zeros(N)
            if t >= p['t_ramp_start']:
                ramp_input = p['ramp_mean'] * p['ramp_train'][:N] * ((t - p['t_ramp_start']) / p['ramp_dur']) * (1 + p['ramp_sigma'] * np.random.randn()) + p['ramp_bsln']
            stim_input = np.zeros(N)
            if t in p['t_stim_interval']:
                stim_input = stm_trials[i] * (1 + p['sigma_stim'] * np.random.randn())
            x += (-x + M @ np.hstack([r, r_in_cd[:, t]]) + h) * p['eff_dt'] + noise_sigma_eff * np.random.randn(N)
            r = f0 / (1.0 + np.exp(-beta0 * (x - theta0)))
            RNN_fr_cd[:, t, i] = r

    return {"RNN_fr_cd": RNN_fr_cd}

cd_struct = f_cd(N_trials_cd, params_dict, cd_span)

# # Plot trials
# plt.figure()
# for i in range(N_trials_cd):
#     plt.plot(np.mean(cd_struct["RNN_fr_cd"][:, :, i], axis=0), 'k')
# plt.ylabel('Spike Rate (Hz)')
# plt.xlabel('Time to Go cue (s)')
# plt.title('Mean Firing rate (Network)')
# # plt.show()
# plt.savefig('test_plots/mean_firing_rates_RNN_ALM.png',dpi=200)

# Generate trials without distractors
def f_up(N_trials_up, params, cd_struct):
    print("Generating trials without distractors...")
    return {"RNN_fr_up": np.zeros((N, params["T_test"], N_trials_up))}


N_trials_up = 200
up_struct = f_up(N_trials_up, params_dict, cd_struct)


# Generate trials with distractors
def f_dist(N_trials_dist, params, cd_struct, up_struct, input_vector):
    print(f"Generating trials with distractors using input vector: {input_vector}")
    return {"RNN_fr_dist": np.zeros((N, params["T_test"], N_trials_dist))}


N_trials_dist = 100
input_vector = 's'
dist_struct = f_dist(N_trials_dist, params_dict, cd_struct, up_struct, input_vector)

# Save results
os.makedirs(Path(output_file_path).parent, exist_ok=True)
# sio.savemat(output_file_path, {'cd_struct': cd_struct, 'up_struct': up_struct, 'dist_struct': dist_struct})

