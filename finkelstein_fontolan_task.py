import numpy as np
import scipy.io as sio

class FinkelsteinFontolanTask:
    def __init__(self, dt, T_test, N_trials_cd, ramp_train, t_ramp_start, ramp_dur,
                 ramp_mean, ramp_sigma, ramp_bsln, amp_stim, sigma_stim,
                 t_stim_interval, amp_chirp, fr_smooth):
        """
        Parameters:
          dt              : time step (ms)
          T_test          : total trial duration (ms)
          N_trials_cd     : number of trials
          ramp_train      : 1D numpy array of ramp train values (length should be T_test/dt)
          t_ramp_start    : time index at which the ramp begins (assumed 0-indexed)
          ramp_dur        : duration (in time steps) of the ramp input
          ramp_mean       : mean slope of the ramp input
          ramp_sigma      : standard deviation for ramp noise (applied as a scalar factor per trial)
          ramp_bsln       : baseline added to the ramp input
          amp_stim        : amplitude of the stimulus input (for “right” trials)
          sigma_stim      : noise level for the stimulus input
          t_stim_interval : list or numpy array of time indices for stimulus presentation
          amp_chirp       : amplitude of the chirp input
          fr_smooth       : smoothing window length (integer)
        """
        self.dt = dt
        self.T_test = T_test
        self.N_trials_cd = N_trials_cd
        self.ramp_train = ramp_train
        self.t_ramp_start = t_ramp_start
        self.ramp_dur = ramp_dur
        self.ramp_mean = ramp_mean
        self.ramp_sigma = ramp_sigma
        self.ramp_bsln = ramp_bsln
        self.amp_stim = amp_stim
        self.sigma_stim = sigma_stim
        self.t_stim_interval = np.array(t_stim_interval)
        self.amp_chirp = amp_chirp
        self.fr_smooth = fr_smooth

        # Calculate number of time steps (assumes T_test is divisible by dt)
        self.num_steps = int(T_test / dt)

    # def smooth(self, x, window_len):
    #     """Simple moving average smoothing using a uniform window."""
    #     window = np.ones(window_len) / window_len
    #     return np.convolve(x, window, mode='same')
    def smooth(self, x, window_len):
        """Smooth a 1D signal using a moving average with a window of length window_len,
        while preserving the original length.

        This function pads the signal with different numbers of elements on the left and right.
        """
        window = np.ones(window_len) / window_len
        # Calculate asymmetric padding: total padding = window_len - 1
        left_pad = int((window_len - 1) // 2)
        right_pad = int(window_len - 1 - left_pad)
        x_padded = np.pad(x, (left_pad, right_pad), mode='edge')
        return np.convolve(x_padded, window, mode='valid')

    def __call__(self):
        """
        Returns:
          r_in_cd: a numpy array of shape (num_steps, N_trials_cd, 3) with:
                   - Channel 0: chirp input (same for all trials)
                   - Channel 1: stimulus input (trial-specific)
                   - Channel 2: ramp input (trial-specific)
        """
        num_steps = self.num_steps
        N_trials = self.N_trials_cd

        # -------------------------------
        # Compute Chirp Input (common to all trials)
        # -------------------------------
        chirp_input = np.zeros(num_steps)
        # Define chirp intervals (assuming times are in ms and dt is in ms)
        start1 = int(500 / self.dt)
        end1 = int(650 / self.dt)
        start2 = int(1350 / self.dt)
        end2 = int(1500 / self.dt)
        chirp_input[start1:end1] = 1
        chirp_input[start2:end2] = 1
        # Smooth the chirp signal and scale by amplitude
        chirp_input = self.smooth(chirp_input, self.fr_smooth)
        chirp_input = self.amp_chirp * chirp_input

        # Preallocate r_in_cd with shape (time_steps, N_trials, 3)
        r_in_cd = np.zeros((num_steps, N_trials, 3))
        # Assign chirp input to channel 0 (for all trials)
        r_in_cd[:, :, 0] = np.tile(chirp_input[:, np.newaxis], (1, N_trials))

        # -------------------------------
        # Prepare Stimulus Trial Mask
        # -------------------------------
        # First half of trials get 0 stimulus ("left"), second half get amp_stim ("right")
        stm_trials = np.concatenate((np.zeros(N_trials // 2),
                                     self.amp_stim * np.ones(N_trials // 2)))

        # -------------------------------
        # Loop over each trial to generate stimulus and ramp inputs
        # -------------------------------
        for i in range(N_trials):
            # ---- Stimulus Input ----
            stim_input = np.zeros(num_steps)
            # Draw a single noise scalar for the trial's stimulus
            noise_stim = 1 + self.sigma_stim * np.random.randn()
            stim_input[self.t_stim_interval] = stm_trials[i] * noise_stim
            stim_input = self.smooth(stim_input, self.fr_smooth)

            # ---- Ramping Input ----
            ramp_input = np.zeros(num_steps)
            inp_ramp_test = np.zeros(num_steps)
            # Define indices for the ramp portion
            start_ramp = self.t_ramp_start
            end_ramp = self.t_ramp_start + self.ramp_dur
            # Create a normalized ramp (linearly increasing from 1/ramp_dur to 1)
            inp_ramp_test[start_ramp:end_ramp] = np.arange(1, self.ramp_dur + 1) / self.ramp_dur
            # Use a single noise factor for the entire ramp on this trial
            noise_factor = 1 + self.ramp_sigma * np.random.randn()
            # Compute ramp input during the ramp period
            ramp_input[:end_ramp] = (self.ramp_mean * self.ramp_train[:end_ramp] *
                                     inp_ramp_test[:end_ramp] * noise_factor +
                                     self.ramp_bsln)
            # After the ramp period, hold the value constant (last computed value)
            ramp_input[end_ramp:] = ramp_input[end_ramp - 1]
            ramp_input = self.smooth(ramp_input, self.fr_smooth)

            # Assign stimulus and ramp inputs to their respective channels
            r_in_cd[:, i, 1] = stim_input
            r_in_cd[:, i, 2] = ramp_input

        targets = np.zeros_like(r_in_cd)[:,:,0:1]
        return r_in_cd,targets


def initialize_task(input_file_path="./RNN_ALM_gating/input_data/",N_trials_cd = 10):
    """
    Loads input and parameter data from the specified path and initializes the task object.

    Parameters:
      input_file_path (str): Path to the folder containing the MAT files.

    Returns:
      data (dict): The loaded input data from 'input_data_wramp.mat'.
      params (dict): Dictionary of RNN parameters loaded from 'params_data_wramp.mat'.
      task_obj (CodingDirectionInputs): Task object initialized with the parameters.
    """
    # Load input data (e.g., ramp train or any other input-specific data)
    file_name = "input_data_wramp"  # name of the input data file (without extension)
    data = sio.loadmat(f"{input_file_path}{file_name}.mat")

    # Load parameters data
    params_file_name = "params_data_wramp"
    params_mat = sio.loadmat(f"{input_file_path}{params_file_name}.mat")

    # Extract and prepare parameters from the MAT file.
    N = int(params_mat['params']['N'][0][0])
    params = {
        "N": N,
        "t_ramp_start": 500,
        "t_stim_interval": np.arange(1000, 1400),  # Adjusted for Python indexing
        "T_test": 5000,
        "ramp_dur": 3000,
        "sigma_noise_cd": 100. / N,
        "ramp_mean": 1.0,
        "ramp_sigma": 0.05,
        "amp_stim": 1,
        "sigma_stim": 0.1,
        "endpoint": 3500,
        "amp_chirp": 1,
        "dt": params_mat['params']['dt'][0][0][0, 0],
        "tau": params_mat['params']['tau'][0][0][0, 0],
        "f0": params_mat['params']['f0'][0][0][0, 0],
        "beta0": float(params_mat['params']['beta0'][0][0]),
        "theta0": float(params_mat['params']['theta0'][0][0]),
        "M": params_mat['params']['M'][0, 0][:N, :N],
        "h": params_mat['params']['h'][0, 0][:N].flatten(),
        "sigma_noise": params_mat['params']['tau_noise'][0][0][0, 0],
        "ramp_train": params_mat['params']['ramp_train'],
        "fr_smooth": params_mat['params']['fr_smooth'][0][0],
        "ramp_bsln": params_mat['params']['ramp_bsln'][0][0],
        "eff_dt": params_mat['params']['eff_dt'][0][0],
        "des_out_left": params_mat['params']['des_out_left'],
        "des_out_right": params_mat['params']['des_out_right'],
        "des_r_left_norm": params_mat['params']['des_r_left_norm'],
        "des_r_right_norm": params_mat['params']['des_r_right_norm'],
    }


    # Ensure ramp_train is a 1D numpy array.
    ramp_train = params['ramp_train'].flatten() if params['ramp_train'].ndim > 1 else params['ramp_train']

    # Initialize the task object with the required parameters.
    task_obj = FinkelsteinFontolanTask(
        dt=params['dt'],
        T_test=params['T_test'],
        N_trials_cd=N_trials_cd,
        ramp_train=ramp_train,
        t_ramp_start=params['t_ramp_start'],
        ramp_dur=params['ramp_dur'],
        ramp_mean=params['ramp_mean'],
        ramp_sigma=params['ramp_sigma'],
        ramp_bsln=params['ramp_bsln'],
        amp_stim=params['amp_stim'],
        sigma_stim=params['sigma_stim'],
        t_stim_interval=params['t_stim_interval'],
        amp_chirp=params['amp_chirp'],
        fr_smooth=params['fr_smooth'][0][0]
    )
    return task_obj

if __name__ == "__main__":
    input_path = "./RNN_ALM_gating/input_data/"  # adjust the path as needed
    task = initialize_task(input_path)
    # Now you can generate the inputs for a trial by calling the task object:
    r_in_cd,_ = task()  # shape: (num_steps, N_trials_cd, 3)
    print("r_in_cd shape:", r_in_cd.shape)

    import matplotlib.pyplot as plt
    plt.plot(r_in_cd[:,9,:])
    plt.show()