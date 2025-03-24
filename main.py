from config_utils import instantiate
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf_utils import omegaconf_resolvers
from learn_koopman_eig import train
from pathlib import Path
from functools import partial
import os
from compose import compose
from plotting import plot_model_contour,plot_kinetic_energy

from sklearn.decomposition import PCA
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import seaborn as sns
mpl.rcParams['agg.path.chunksize'] = 10000

# mpl.rcParams['text.usetex'] = True
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


from separatrixLocator import SeparatrixLocator

import torch
from torchdiffeq import odeint
import numpy as np

PATH_TO_FIXED_POINT_FINDER = f'{os.getenv("PROJECT_PATH")}/fixed_point_finder'
import sys
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from fixed_point_finder.FixedPointFinderTorch import FixedPointFinderTorch


CONFIG_PATH = "configs"
# CONFIG_NAME = "test"
CONFIG_NAME = "main"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_path = os.getenv("PROJECT_PATH")


@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def decorated_main(cfg):
    # return main(cfg)
    return main_multimodel(cfg)

def main_multimodel(cfg):
    """
    Uses the SeparatrixLocator class.
    """
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)

    # OmegaConf.resolve(cfg.model)

    # print(OmegaConf.to_yaml(cfg))

    dynamics_function = instantiate(cfg.dynamics.function)
    input_distribution = instantiate(cfg.dynamics.external_input_distribution) if hasattr(cfg.dynamics, 'external_input_distribution') else None
    distribution = instantiate(cfg.dynamics.IC_distribution)
    if hasattr(cfg.dynamics, 'IC_distribution_fit'):
        distribution_fit = instantiate(cfg.dynamics.IC_distribution_fit)
    else:
        distribution_fit = distribution

    if hasattr(cfg.dynamics, 'external_input_distribution_fit'):
        input_distribution_fit = instantiate(cfg.dynamics.external_input_distribution_fit)
    else:
        input_distribution_fit = input_distribution

    

    if input_distribution is not None:
        cfg.model.input_size = cfg.dynamics.dim + cfg.dynamics.external_input_dim
        OmegaConf.resolve(cfg.model)

    SL = instantiate(cfg.separatrix_locator)
    SL.models = [instantiate(cfg.model).to(SL.device) for _ in range(cfg.separatrix_locator.num_models)]

    if cfg.load_KEF_model:
        new_format_path = Path(cfg.savepath)/cfg.experiment_details
        print(new_format_path)
        if os.path.exists(new_format_path):
            load_path = new_format_path
        else:
            load_path = Path(cfg.savepath)
            print(new_format_path, 'does not exist, loading', load_path, 'instead.')
        SL.load_models(load_path)

    # with torch.no_grad():
    #     SL.models[0][0].weight[:] = 1.0
    # print('skipping fit')
    SL.fit(
        dynamics_function,
        distribution_fit,
        external_input_dist=input_distribution_fit,
        **instantiate(cfg.separatrix_locator_fit_kwargs),
        # **{
        #     **instantiate(cfg.separatrix_locator_fit_kwargs),
        #     "learning_rate":1e-5,
        # },
    )
    # print('SL.models[0][0].weight',SL.models[0][0].weight)
    # x = torch.arange(-2.0, 2.0, 0.01)
    # plt.plot(x,dynamics_function(x))
    # plt.plot(x, SL.models[0](x[:,None])[:,0].detach())
    # plt.show()

    if cfg.save_KEF_model:
        SL.save_models(Path(cfg.savepath)/cfg.experiment_details)

    scores = SL.score(
        dynamics_function,
        distribution,
        external_input_dist=input_distribution,
        **instantiate(cfg.separatrix_locator_score_kwargs)
    )
    print('Scores:\n', scores.detach().cpu().numpy())
    if hasattr(cfg,'separatrix_locator_score_kwargs_2'):
        scores2 = SL.score(
            dynamics_function,
            distribution,
            external_input_dist=input_distribution,
            **instantiate(cfg.separatrix_locator_score_kwargs_2)
        )
        print('Scores over 2x scaled distribution:\n',scores2.detach().cpu().numpy())

    SL.models = [model.to('cpu') for model in SL.models]
    #SL.filter_models(0.1)

    all_below_threshold_points = None
    if cfg.runGD:
        external_inputs = None
        if input_distribution is not None:
            external_inputs = input_distribution.sample(sample_shape=(cfg.separatrix_find_separatrix_kwargs.batch_size,))
        _, all_below_threshold_points = SL.find_separatrix(
            distribution,
            external_inputs = external_inputs,
            dist_needs_dim = cfg.dynamics.dist_requires_dim if hasattr(cfg.dynamics,"dist_requires_dim") else True,
            **instantiate(cfg.separatrix_find_separatrix_kwargs)
        )
        print('all_below_threshold_points',all_below_threshold_points)

    if cfg.run_fixed_point_finder:
        assert hasattr(cfg.dynamics,"loaded_RNN_model")
        rnn_model = instantiate(cfg.dynamics.loaded_RNN_model_partial)(device=SL.device)
        # cfg.dynamics.RNN_dataset.batch_size = 5000
        # cfg.dynamics.RNN_dataset.n_trials = 1000
        dataset = instantiate(cfg.dynamics.RNN_dataset)
        inp, targ = dataset()

        torch_inp = torch.from_numpy(inp).type(torch.float)  # .to(device)
        outputs, hidden_traj = rnn_model(torch_inp, return_hidden=True, deterministic=False)
        outputs, hidden_traj = outputs.detach().cpu().numpy(), hidden_traj.detach().cpu().numpy()


        FPF = FixedPointFinderTorch(
            rnn_model.rnn if hasattr(rnn_model,"rnn") else rnn_model,
            **instantiate(cfg.fpf_hps)
        )
        num_trials = 500
        # initial_conditions = dist.sample(sample_shape=(num_trials,)).detach().cpu().numpy()
        # inputs = np.zeros((1, cfg.dynamics.RNN_model.act_size))
        # inputs[...,2] = 1.0
        torch_inp[...,:2] = 0.0
        fp_inputs = torch_inp.reshape(-1, torch_inp.shape[-1]).detach().cpu().numpy()

        # inputs[...,0] = 1
        initial_conditions = hidden_traj.reshape(-1, hidden_traj.shape[-1])
        select = np.random.choice(initial_conditions.shape[0], size=num_trials, replace=False)
        initial_conditions = initial_conditions[select]
        fp_inputs = fp_inputs[select]
        # fp_inputs[:,:2] = 0
        initial_conditions += np.random.normal(size=initial_conditions.shape) * 2.0 #0.5 #2.0
        # print('initial_conditions', initial_conditions.shape)
        unique_fps, all_fps = FPF.find_fixed_points(
            deepcopy(initial_conditions),
            fp_inputs
        )

        # print(all_fps.shape)
        KEF_val_at_fp = {}
        for i in range(SL.num_models):
            # below_threshold_points = all_below_threshold_points[i] if all_below_threshold_points is not None else None
            mod_model = compose(
                torch.log,
                lambda x: x + 1,
                torch.exp,
                partial(torch.sum, dim=-1, keepdims=True),
                torch.log,
                torch.abs,
                SL.models[i]
            )
            # KEF_val_at_fp[f'KEF{i}'] = mod_model(torch.from_numpy(unique_fps.xstar).to(SL.device)).detach().cpu().numpy().flatten()

        fixed_point_data = {
            'stability': unique_fps.is_stable,
            'q': unique_fps.qstar,
        }
        fixed_point_data.update(KEF_val_at_fp)
        # print(fixed_point_data)
        fixed_point_data = pd.DataFrame(fixed_point_data)
        fixed_point_data.to_csv(Path(cfg.savepath) / 'fixed_point_data.csv', index=False)

    if cfg.run_analysis:

        #### Plotting log prob vs KEF amplitude
        # num_samples = 1000
        # needs_dim = True
        # if hasattr(cfg.dynamics, 'dist_requires_dim'):
        #     needs_dim = cfg.dynamics.dist_requires_dim
        #
        # samples = distribution.sample(
        #     sample_shape=[num_samples] + ([cfg.dynamics.dim] if needs_dim else []))
        #
        # samples.requires_grad_(True)
        #
        # fig,axs = plt.subplots(2,1,sharex=True,figsize=(6,8))
        # for j in range(SL.num_models):
        #     mod_model = compose(
        #         lambda x: x.sum(axis=-1,keepdims=True),
        #         torch.log,
        #         torch.abs,
        #         SL.models[j]
        #     )
        #     log_probs = distribution.log_prob(samples).detach().cpu().numpy()
        #     phi_x = mod_model(samples.to(SL.device))#.detach().cpu().numpy()
        #     # print(log_probs.shape,phi_x.shape)
        #     # from learn_koopman_eig import eval_loss
        #     # losses = eval_loss(
        #     #     model,
        #     #     normaliser=lambda x,y:(x - y) ** 2
        #     # )
        #     # Compute phi'(x)
        #     phi_x_prime = torch.autograd.grad(
        #         outputs=phi_x,
        #         inputs=samples,
        #         grad_outputs=torch.ones_like(phi_x),
        #         create_graph=True
        #     )[0]
        #     # Compute F(x_batch)
        #     F_x = dynamics_function(samples)
        #
        #     # Main loss term: ||phi'(x) F(x) - phi(x)||^2
        #     dot_prod = (phi_x_prime * F_x).sum(axis=-1, keepdim=True)
        #     errors = torch.abs(dot_prod - phi_x).detach().cpu().numpy()
        #
        #     phi_x = phi_x.detach().cpu().numpy()
        #
        #     ax = axs[0]
        #     ax.scatter(np.repeat(log_probs[...,None],repeats=phi_x.shape[-1],axis=-1),np.abs(phi_x),s=10)
        #     ax.set_ylabel(r'$|$KEF$(x)|$')
        #     ax.set_yscale('log')
        #     ax.set_xlabel(r'$\log p(x)$')
        #
        #     ax = axs[1]
        #     ax.scatter(log_probs[..., None], errors, s=10)
        #     ax.set_ylabel('PDE error')
        #     ax.set_yscale('log')
        #     ax.set_xlabel(r'$\log p(x)$')
        # fig.tight_layout()
        # fig.savefig(Path(cfg.savepath) / 'log_prob_and_KEF_amplitude.png')



        if cfg.model.input_size == 1:
            pass
        elif cfg.model.input_size == 2:
            fig,axs = plt.subplots(7,4,figsize=np.array([4,7])*2.3,sharey=True,sharex=True)
            for j in range(SL.num_models):
                for i in range(cfg.model.output_size):
                    mod_model = compose(
                        lambda x: x**0.01,
                        torch.log,
                        lambda x: x + 1,
                        torch.exp,
                        # partial(torch.sum, dim=-1, keepdims=True),
                        lambda x: x[...,i:i+1],
                        torch.log,
                        torch.abs,
                        SL.models[j]
                    )
                    ax = axs[i,j]

                    x_limits = (-2, 2)  # Limits for x-axis
                    y_limits = (-2, 2)  # Limits for y-axis
                    if hasattr(cfg.dynamics, 'lims'):
                        x_limits = cfg.dynamics.lims.x
                        y_limits = cfg.dynamics.lims.y
                    plot_model_contour(
                        mod_model,
                        ax,
                        x_limits=x_limits,
                        y_limits=y_limits,
                    )
                    below_threshold_points = all_below_threshold_points[j] if all_below_threshold_points is not None else None
                    # print(below_threshold_points.shape)
                    if below_threshold_points is not None:
                        xlim = ax.get_xlim()  # Store current x limits
                        ylim = ax.get_ylim()  # Store current y limits

                        ax.scatter(below_threshold_points[:, 0], below_threshold_points[:, 1], c='red', s=10)

                        ax.set_xlim(xlim)  # Reset x limits
                        ax.set_ylim(ylim)  # Reset y limits
                    # ax.set_aspect('auto')
                    ax.set_title(f'Model-{j},output{i}'+'\n'+f", loss:{scores[j, i]:.5f}")
                    ax.set_xlabel('')
                    ax.set_ylabel('')

                    # ax.scatter(*unique_fps.xstar[unique_fps.is_stable, :].T, c='blue',marker='x',s=100,zorder=1001)
                    # ax.scatter(*unique_fps.xstar[~unique_fps.is_stable, :].T, c='red',marker='x',s=100,zorder=1000)
            fig.tight_layout()
            fig.savefig(Path(cfg.savepath)/"all_KEF_contours.png",dpi=300)
            plt.close(fig)

            x_limits = (-2, 2)  # Limits for x-axis
            y_limits = (-2, 2)  # Limits for y-axis
            if hasattr(cfg.dynamics, 'lims'):
                x_limits = cfg.dynamics.lims.x
                y_limits = cfg.dynamics.lims.y

            fig,ax = plt.subplots(1,1, figsize=(5,5))
            plot_kinetic_energy(
                dynamics_function,
                ax,
                x_limits=x_limits,
                y_limits=y_limits,
                below_threshold_points = np.concatenate(all_below_threshold_points,axis=0) if all_below_threshold_points is not None else None,
            )
            if cfg.run_fixed_point_finder:
                ax.scatter(*unique_fps.xstar[unique_fps.is_stable, :].T, c='blue', marker='x', s=100, zorder=1001)
                ax.scatter(*unique_fps.xstar[~unique_fps.is_stable, :].T, c='red', marker='x', s=100, zorder=1000)
            fig.tight_layout()
            fig.savefig(Path(cfg.savepath) / "kinetic_energy.png",dpi=300)
            plt.close(fig)

        if 'hypercube' in cfg.dynamics.name or 'bistable' in cfg.dynamics.name:
            # Number of models and number of random vectors
            num_models = SL.num_models
            num_positions = cfg.dynamics.dim
            n_trials = 20
            num_random_vectors = 10

            # Create subplots with 10 rows and 10 columns
            fig, axs = plt.subplots(num_models, num_positions, figsize=(num_positions, num_models+1), sharex=True, sharey=True)

            # Iterate over each model
            for model_idx, model in enumerate(SL.models):
                # Iterate over each position for x
                for pos in range(num_positions):
                    ax = axs[model_idx, pos] if SL.num_models>1 else axs[pos]
                    # Generate multiple random vectors for n_1 to n_10
                    for trial in range(n_trials):
                        n_values = np.random.uniform(-1, 1, cfg.dynamics.dim)
                        x_values = np.linspace(-2, 2, 100)

                        # Create an array to store the results for this trial
                        trial_results = []

                        # Create a batch of input arrays with x values swept from -2 to 2
                        input_batch = np.tile(n_values, (len(x_values), 1))
                        input_batch[:, pos] = x_values

                        # Convert input_batch to torch tensor
                        input_tensor = torch.from_numpy(input_batch).float()

                        # Run the model on the input tensor
                        with torch.no_grad():
                            output = model(input_tensor)

                        # Store the results
                        trial_results = output[:, 0].tolist()  # Assuming single output

                        # Plot the results for this trial
                        ax.plot(x_values, trial_results, lw=1, alpha=0.5)

                    # Evaluate when all n_values are zero
                    zero_values = np.zeros(cfg.dynamics.dim)
                    zero_input_batch = np.tile(zero_values, (len(x_values), 1))
                    zero_input_batch[:, pos] = x_values

                    zero_input_tensor = torch.from_numpy(zero_input_batch).float()

                    with torch.no_grad():
                        zero_output = model(zero_input_tensor)

                    ax.plot(x_values, zero_output[:, 0].tolist(), lw=1, alpha=1, color='black', ls='dashed')
            # Set labels for the first column and first row
            # Set titles for the top row and labels for the first column
            for pos in range(num_positions):
                ax = axs[0, pos] if SL.num_models>1 else axs[pos]
                ax.set_title(f'Position {pos}')
            for model_idx in range(len(SL.models)):
                ax = axs[model_idx, 0] if SL.num_models > 1 else axs[0]
                ax.set_ylabel(f'Model {model_idx}')

            plt.tight_layout()
            fig.savefig(Path(cfg.savepath) / "model_output_vs_x_positions.png", dpi=50)
            # fig.savefig(Path(cfg.savepath) / "model_output_vs_x_positions.pdf")
            plt.close(fig)


        elif cfg.dynamics.dim > 2:
            # pass
            num_trials = 50
            times = torch.linspace(0, 500, 100)
            needs_dim = True
            if hasattr(cfg.dynamics, 'dist_requires_dim'):
                needs_dim = cfg.dynamics.dist_requires_dim

            distribution_relevant = instantiate(cfg.dynamics.IC_distribution_task_relevant)

            initial_conditions = distribution_relevant.sample(
                sample_shape=[num_trials] + ([cfg.dynamics.dim] if needs_dim else []))
            external_inputs = input_distribution.sample(sample_shape=[num_trials] + ([cfg.dynamics.dim] if needs_dim else []))

            trajectories = odeint(lambda t, x: dynamics_function(x, external_inputs), initial_conditions, times)
            trajectories = trajectories.detach().cpu().numpy()

            # Instantiate the dataset
            dataset = instantiate(cfg.dynamics.RNN_analysis_dataset)
            inputs, targets = dataset()
            inputs = torch.from_numpy(inputs).type(torch.float)

            inputs = inputs

            # Run trajectories using rnn and inputs from dataset
            rnn = instantiate(cfg.dynamics.loaded_RNN_model)
            outputs, hidden_trajectories = rnn(inputs, return_hidden=True, deterministic=False)
            hidden_trajectories = hidden_trajectories.detach().cpu().numpy()

            # Perform another odeint run using the first time step from hidden_trajectories as the initial conditions
            hidden_initial_conditions = torch.from_numpy(hidden_trajectories[0]).type(torch.float)
            external_inputs_last_step = inputs[-1000]  # Use the last time step of inputs as external inputs
            new_trajectories = odeint(lambda t, x: dynamics_function(x, external_inputs_last_step), hidden_initial_conditions, times)
            new_trajectories = new_trajectories.detach().cpu().numpy()


            # Option to fit PCA on one set of trajectories, the other, or both
            fit_on = 'both'  # Options: 'hidden', 'trajectories', 'both'

            if fit_on == 'hidden':
                pca = PCA(n_components=2)
                hidden_trajectories_reshaped = hidden_trajectories.reshape(-1, hidden_trajectories.shape[-1])
                pca.fit(hidden_trajectories_reshaped)
            elif fit_on == 'trajectories':
                pca = PCA(n_components=2)
                trajectories_reshaped = trajectories.reshape(-1, trajectories.shape[-1])
                pca.fit(trajectories_reshaped)
            else:  # fit_on == 'both'
                combined_trajectories = np.concatenate((hidden_trajectories.reshape(-1, hidden_trajectories.shape[-1]),
                                                        trajectories.reshape(-1, trajectories.shape[-1])), axis=0)
                pca = PCA(n_components=2)
                pca.fit(combined_trajectories)

            # Transform the new trajectories using PCA
            pca_new_trajectories = pca.transform(new_trajectories.reshape(-1, new_trajectories.shape[-1]))
            pca_new_trajectories = pca_new_trajectories.reshape(new_trajectories.shape[0], new_trajectories.shape[1], -1)

            # Transform both sets of trajectories
            pca_hidden_trajectories = pca.transform(hidden_trajectories.reshape(-1, hidden_trajectories.shape[-1]))
            pca_hidden_trajectories = pca_hidden_trajectories.reshape(hidden_trajectories.shape[0], hidden_trajectories.shape[1], -1)

            pca_trajectories = pca.transform(trajectories.reshape(-1, trajectories.shape[-1]))
            pca_trajectories = pca_trajectories.reshape(trajectories.shape[0], trajectories.shape[1], -1)

            from_t = 0

            # Plot combined PCA of both sets of trajectories
            plt.figure(figsize=(10, 6))
            for i in range(pca_hidden_trajectories.shape[1]):
                plt.plot(pca_hidden_trajectories[:, i, 0], pca_hidden_trajectories[:, i, 1], lw=1, alpha=0.5, label='Hidden Trajectories' if i == 0 else "")
                plt.scatter(pca_hidden_trajectories[0, i, 0], pca_hidden_trajectories[0, i, 1], c='blue', marker='o', zorder=5)
                plt.scatter(pca_hidden_trajectories[-1, i, 0], pca_hidden_trajectories[-1, i, 1], c='orange', marker='o', zorder=5)

            for i in range(num_trials):
                plt.plot(pca_trajectories[from_t:, i, 0], pca_trajectories[from_t:, i, 1], lw=1, alpha=0.5, label='Trajectories' if i == 0 else "")
                plt.scatter(pca_trajectories[from_t, i, 0], pca_trajectories[from_t, i, 1], c='green', marker='o', zorder=5)
                plt.scatter(pca_trajectories[-1, i, 0], pca_trajectories[-1, i, 1], c='red', marker='o', zorder=5)

            for i in range(pca_new_trajectories.shape[1]):
                plt.plot(pca_new_trajectories[:, i, 0], pca_new_trajectories[:, i, 1], lw=1, alpha=0.5, label='New Trajectories' if i == 0 else "")
                plt.scatter(pca_new_trajectories[0, i, 0], pca_new_trajectories[0, i, 1], c='purple', marker='o', zorder=5)
                plt.scatter(pca_new_trajectories[-1, i, 0], pca_new_trajectories[-1, i, 1], c='yellow', marker='o', zorder=5)

            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('Combined PCA of Hidden and Regular Trajectories')
            plt.legend()
            plt.tight_layout()
            # plt.show()
            plt.savefig(Path(cfg.savepath) / "pca_trajectories.png", dpi=300)
            plt.close()


        if hasattr(cfg.dynamics,'RNN_dataset'):
            dataset = instantiate(cfg.dynamics.RNN_analysis_dataset)
            dataset.N_trials_cd = 20
            rnn = instantiate(cfg.dynamics.loaded_RNN_model)
            dist = instantiate(cfg.dynamics.IC_distribution)
            inputs, targets = dataset()
            inputs = torch.from_numpy(inputs).type(torch.float)
            targets = torch.from_numpy(targets)
            # inputs = inputs * 0
            # print('inputs.shape',inputs.shape)
            # print('batch first',rnn.batch_first)
            outputs,hidden = rnn(inputs,return_hidden=True,deterministic=False)
            hidden = hidden.detach()



            ###
            # fp_inputs = torch_inp.reshape(-1, torch_inp.shape[-1]).detach().cpu().numpy()
            # initial_conditions = hidden_traj.reshape(-1, hidden_traj.shape[-1])
            ###

            P = PCA(n_components=3)
            # non_linearity = lambda x: rnn_model.f0 / (1.0 + torch.exp(-rnn_model.beta0 * (x - rnn_model.theta0)))
            # rates = non_linearity(hidden)
            hidden_last = hidden[-3000:]
            P.fit(hidden_last.reshape(-1,hidden_last.shape[-1]).detach().cpu())
            pc_hidden = P.transform(hidden.reshape(-1,hidden.shape[-1]).detach().cpu()).reshape(*hidden.shape[:2],P.n_components)

            plt.figure()
            for i in range(pc_hidden.shape[1]):
                plt.plot(inputs[:,i,2],pc_hidden[:,i,0],lw=1,alpha=0.5)

            if cfg.run_fixed_point_finder:
                unique_fps_rates = torch.from_numpy(unique_fps.xstar)
                pc_fps = P.transform(unique_fps_rates)
                # pc_IC  = P.transform(initial_conditions)
                plt.scatter(unique_fps.inputs[unique_fps.is_stable,2],pc_fps[unique_fps.is_stable, 0], c='blue', marker='x', s=100, zorder=1001)
                plt.scatter(unique_fps.inputs[~unique_fps.is_stable,2],pc_fps[~unique_fps.is_stable, 0], c='red', marker='x', s=100, zorder=1000)
                # plt.scatter(fp_inputs[:,2],pc_IC[:,0],c='green')

            plt.savefig(Path(cfg.savepath) / "PCA_traj.png",dpi=300)
            plt.close()

            KEFvals = []
            for i in range(SL.num_models):
                mod_model = compose(
                    # lambda x: x**0.1,
                    torch.log,
                    lambda x: x + 1,
                    torch.exp,
                    partial(torch.sum, dim=-1, keepdims=True),
                    torch.log,
                    torch.abs,
                    SL.models[i] #.to('cpu')
                )
                samples_for_normalisation = 1000
                needs_dim = True
                if hasattr(cfg.dynamics, 'dist_requires_dim'):
                    needs_dim = cfg.dynamics.dist_requires_dim

                dist_option = dist
                if hasattr(cfg.dynamics,"combined_distribution"):
                    dist_option = instantiate(cfg.dynamics.combined_distribution)

                samples = dist_option.sample(sample_shape=[samples_for_normalisation])

                norm_val = float(
                    torch.mean(torch.sum(mod_model(samples) ** 2, axis=-1)).sqrt().detach().numpy())

                mod_model = compose(
                    lambda x: x / norm_val,
                    mod_model
                )
                KEFval = mod_model(torch.concat([hidden,inputs],dim=-1)).detach()
                KEFvals.append(KEFval)

            KEFvals = torch.concat(KEFvals,dim=-1).detach().cpu()

            fig,axs = plt.subplots(2,5, figsize=(10,4), sharex=True, sharey=True)
            for i in range(SL.num_models):
                ax = axs.flatten()[i]
                scatter = ax.scatter(inputs[:,:,2], pc_hidden[:,:,0], c=KEFvals[:,:,i], s=10, cmap='viridis')
                fig.colorbar(scatter, ax=ax)
            fig.tight_layout()
            fig.savefig(Path(cfg.savepath) / "PCA1_inputs_KEFvals.png",dpi=300)


            # threshold = np.quantile(inputs[:,:,2].flatten(), 0.96)
            # top_indices = np.where(inputs[:,:,2] >= threshold)
            top_indices = np.where(
                (0.9 <= inputs[:, :, 2]) & (inputs[:, :, 2] <= 0.92)
            )

            # Extract corresponding hidden states for those indices
            top_hidden = hidden[top_indices[0], top_indices[1], :].detach().cpu().numpy()

            # Compute pairwise euclidean distances between all points in top_hidden using torch
            distances = torch.cdist(
                torch.from_numpy(top_hidden), 
                torch.from_numpy(top_hidden)
            )
            
            # Get indices of points with maximum distance
            max_dist_idx = np.unravel_index(torch.argmax(distances), distances.shape)
            
            # Get the actual points with maximum distance
            point1 = top_hidden[max_dist_idx[0]]
            point2 = top_hidden[max_dist_idx[1]]
            max_distance = distances[max_dist_idx[0], max_dist_idx[1]].item()
            
            print(f"Maximum distance: {max_distance}")

            
            # print(f"Point 1: {point1}")
            # print(f"Point 2: {point2}")

            plt.figure()
            plt.hist(point1.flatten())
            plt.hist(point2.flatten())
            plt.savefig(Path(cfg.savepath) / "fixedpoint_histograms.png",dpi=300)
            plt.close()

            # Create 100 linearly interpolated points between point1 and point2
            n_grid = 1000
            alpha = torch.linspace(-.5, 1.5, n_grid)
            interpolated_points = alpha[:, None] * point1[None, :] + (1 - alpha)[:, None] * point2[None, :]
            interpolated_inputs = (inputs[top_indices[0][max_dist_idx[0]], top_indices[1][max_dist_idx[0]]] + 
                                 inputs[top_indices[0][max_dist_idx[1]], top_indices[1][max_dist_idx[1]]]) / 2
            
            concat_interpolated = torch.concat([interpolated_points,interpolated_inputs.repeat(n_grid,1)],dim=-1)

            KEFvals = []
            for i in range(SL.num_models):
                mod_model = compose(
                    # lambda x: x**0.1,
                    torch.log,
                    lambda x: x + 1,
                    # torch.exp,
                    # partial(torch.sum, dim=-1, keepdims=True),
                    # torch.log,
                    torch.abs,
                    SL.models[i]#.to('cpu')
                )
                samples_for_normalisation = 1000
                needs_dim = True


                samples = dist_option.sample(sample_shape=[samples_for_normalisation])

                norm_val = float(
                    torch.mean(torch.sum(mod_model(samples) ** 2, axis=-1)).sqrt().detach().numpy())

                mod_model = compose(
                    lambda x: x / norm_val,
                    mod_model
                )
                KEFval = mod_model(concat_interpolated).detach()
                KEFvals.append(KEFval)

            KEFvals = torch.stack(KEFvals, dim=0).detach().cpu()

            fig,ax = plt.subplots()
            for i in range(SL.num_models):
                ax.plot(KEFvals[i],c=f'C{i}')
            ax.set_ylabel(r'KEF value')
            ax.set_xlabel('Position along decision axis')
            plt.savefig(Path(cfg.savepath) / "KEFs_interpolated.png", dpi=300)
            plt.close()

            plt.figure()
            for i in range(pc_hidden.shape[1]):
                plt.plot(inputs[:,i,2],pc_hidden[:,i,0],lw=1,alpha=0.5)
            pc_interpolated_points = P.transform(interpolated_points)
            plt.plot(concat_interpolated[:,-1],pc_interpolated_points[:,0],lw=1,ls='dotted',c='black')
            plt.savefig(Path(cfg.savepath) / "PCA_interpolation_line.png", dpi=300)

            plt.figure()
            for i in range(pc_hidden.shape[1]):
                plt.plot(inputs[:, i, :])
            plt.savefig(Path(cfg.savepath) / "inputs.png", dpi=300)
            plt.close()


            # Run from interpolated line
            n_grid = 100
            alpha = torch.linspace(-0.4, 1.4, n_grid)
            interpolated_points = alpha[:, None] * point1[None, :] + (1 - alpha)[:, None] * point2[None, :]
            interpolated_inputs = (inputs[top_indices[0][max_dist_idx[0]], top_indices[1][max_dist_idx[0]]] +
                                   inputs[top_indices[0][max_dist_idx[1]], top_indices[1][max_dist_idx[1]]]) / 2

            # Set simulation length for interpolated points
            run_T = 1000

            # Expand interpolated inputs over time dimension
            interpolated_inputs_expanded = interpolated_inputs.repeat(n_grid, 1).unsqueeze(0).expand(run_T, -1, -1)

            use_odeint = True

            if use_odeint:
                dynamics_function = instantiate(cfg.dynamics.function)

                def ode_dynamics(t, x):
                    return dynamics_function(x, interpolated_inputs_expanded[0, :, :])

                times = torch.linspace(0, run_T, run_T)
                interpolated_trajectories = odeint(ode_dynamics, interpolated_points, times)
            else:
                # Run RNN from interpolated initial conditions
                interpolated_trajectories = rnn(
                    interpolated_inputs_expanded,
                    x_init=interpolated_points[None],
                    deterministic=True,
                    return_hidden=True
                )[1]  # Only take second output, ignoring hidden states


            # Reshape trajectories
            interpolated_trajectories_r = interpolated_trajectories.reshape(run_T, n_grid, -1)

            # setattr(cfg.separatrix_locator_fit_kwargs, 'num_epochs', 2000)
            # SL.fit(
            #     dynamics_function,
            #     distribution,
            #     external_input_dist=input_distribution,
            #     fixed_x_batch = interpolated_points,
            #     fixed_external_inputs = interpolated_inputs_expanded[0],
            #     **instantiate(cfg.separatrix_locator_fit_kwargs)
            # )


            #### Computing PDE error
            from learn_koopman_eig import shuffle_normaliser

            # Define three sets of points
            point_sets = [
                instantiate(cfg.dynamics.IC_distribution_full).sample(sample_shape=(interpolated_points.shape[0],)),
                instantiate(cfg.dynamics.IC_distribution_task_relevant).sample(sample_shape=(interpolated_points.shape[0],)),
                instantiate(cfg.dynamics.IC_distribution_task_relevant_PC1).sample(sample_shape=(interpolated_points.shape[0],)),
                instantiate(cfg.dynamics.IC_interpolation_line_1).sample(sample_shape=(interpolated_points.shape[0],)),
                interpolated_points,
                interpolated_points + torch.normal(mean=0.0, std=0.4, size=interpolated_points.shape),
            ]

            #dist_PC1 = instantiate(cfg.dynamics.IC_distribution_task_relevant_PC1)

            colors = ['g', 'b', 'y',  'purple', 'r', 'orange']  # Colors for each set
            labels = ['Isotropic','Task data', 'Task data PC1', 'interpolated_points', 'line', 'line+noise0.4',]

            from mpl_toolkits.mplot3d import Axes3D

            plt.close()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i, (points, label) in enumerate(zip(point_sets, labels)):
                if i<2:
                    continue
                # Transform points using PCA
                transformed_points = P.transform(points.detach().cpu().numpy())
                # Plot PC1, PC2, and PC3
                ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], label=label, alpha=0.4, c=colors[i])
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.legend()
            plt.show()

            vector_1 = point_sets[2][1] - point_sets[2][0]
            vector_2 = point_sets[4][1] - point_sets[4][0]  # Adjusted index to match the provided point_sets
            from scipy.stats import pearsonr
            pearson_corr, _ = pearsonr(vector_1.detach().cpu(), vector_2.detach().cpu())
            print(f"Pearson's correlation of the two vectors: {pearson_corr}")
            

            plt.close()
            plt.figure()
            for i, (points, label) in enumerate(zip(point_sets,labels)):
                if i<3:
                    continue
                if i>4:
                    continue
                points.requires_grad_(True)
                inputs = torch.concat([points, interpolated_inputs_expanded[0]], axis=-1)
                phi_x = SL.predict(inputs, no_grad=False)

                phi_x_prime = torch.autograd.grad(
                    outputs=phi_x,
                    inputs=points,
                    grad_outputs=torch.ones_like(phi_x),
                    create_graph=True
                )[0]

                # Compute F(x_batch)
                F_x = dynamics_function(points, interpolated_inputs_expanded[0])

                # Main loss term: ||phi'(x) F(x) - phi(x)||^2
                dot_prod = (phi_x_prime * F_x).sum(axis=-1, keepdim=True)

                # Use shuffle_normaliser to compute the normalized loss
                normalised_loss = shuffle_normaliser(dot_prod, phi_x)
                print(f"Normalised loss for set {i+1}:", normalised_loss)

                # # Evaluate the log likelihood at the points
                # log_likelihoods = dist.log_prob(points)
                # print(f"Log likelihoods at points for set {i+1}:", log_likelihoods)

                # Evaluate the log likelihood at zero
                # zero_point = torch.zeros_like(points)
                # log_likelihoods_zero = dist.log_prob(zero_point)
                # print(f"Log likelihoods at zero for set {i+1}:", log_likelihoods_zero)

                # Plot PDE errors for each set
                plt.scatter(
                    dot_prod.detach().cpu().repeat(1, phi_x.shape[-1]), phi_x.detach().cpu(), color=colors[i], label=label, alpha=0.4
                )

            plt.xlabel(r'$\nabla \psi(x) \cdot f(x)$')
            plt.ylabel(r'$\lambda \psi(x)$')
            plt.legend()
            # plt.aspect('equal')
            plt.tight_layout()
            plt.show()
            plt.savefig(Path(cfg.savepath) / "PDE_scatter.png", dpi=300)




            # Transform interpolated trajectories using PCA
            interpolated_trajectories_pca = P.transform(interpolated_trajectories_r.reshape(-1, interpolated_trajectories_r.shape[-1]).detach().cpu().numpy())
            interpolated_trajectories_pca = interpolated_trajectories_pca.reshape(*interpolated_trajectories_r.shape[:-1], -1)

            # Create figure for PCA trajectories over time
            fig, ax = plt.subplots(figsize=(10, 6))
            # Create colormap based on alpha values
            norm = plt.Normalize(alpha.min(), alpha.max())
            cmap = plt.cm.viridis
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            for i in range(n_grid):
                ax.plot(interpolated_trajectories_pca[:, i, 0],
                        color=cmap(norm(alpha[i])),
                        alpha=0.5, lw=1)
            # Adjust layout and add colorbar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            fig.colorbar(mappable, cax=cbar_ax, label='Alpha')
            ax.set_xlabel('time')
            ax.set_ylabel('PC1')
            ax.set_title('PCA Trajectories from Interpolated Initial Conditions')
            plt.show()
            fig.savefig(Path(cfg.savepath) / "interpolated_trajectories_pca.png", dpi=300)
            plt.close(fig)

            # Concatenate points and inputs for KEF evaluation
            concat_traj = torch.cat([interpolated_trajectories_r, interpolated_inputs_expanded], dim=-1)

            # Evaluate KEFs
            KEFvals_traj = []
            for i in range(SL.num_models):
                mod_model = compose(
                    # torch.log,
                    # lambda x: x + 1,
                    # torch.abs,
                    SL.models[i]
                )
                samples_for_normalisation = 1000
                samples = dist_option.sample(sample_shape=[samples_for_normalisation])
                norm_val = float(
                    torch.mean(torch.sum(mod_model(samples) ** 2, axis=-1)).sqrt().detach().numpy())
                # mod_model = compose(
                #     lambda x: x / norm_val,
                #     mod_model
                # )
                KEFval = mod_model(concat_traj).detach()
                KEFvals_traj.append(KEFval)

            KEFvals_traj = torch.stack(KEFvals_traj, dim=0).detach().cpu()

            # Plot KEFs
            fig, axs = plt.subplots(2, 5, figsize=(10, 4))
            axs = axs.ravel()
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            for i in range(SL.num_models):
                for t in range(n_grid):
                    axs[i].plot(KEFvals_traj[i, :, t],
                              color=cmap(norm(alpha[t])),
                              alpha=0.5, lw=1)
                axs[i].set_title(f'KEF {i+1}')
                axs[i].set_xlabel('Time')
                axs[i].set_ylabel('KEF value')
                axs[i].set_xlim(0,20)
            # Adjust layout and add colorbar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            # fig.tight_layout()
            fig.colorbar(mappable, cax=cbar_ax, label='Alpha')
            fig.savefig(Path(cfg.savepath) / "interpolated_trajectories_KEF.png", dpi=300)
            plt.close(fig)
            
            
            ### KEFs vs PCA
            # Create a figure with two subplots stacked vertically
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

            # Plot KEFs in top subplot
            for i in range(SL.num_models):
                ax1.plot(alpha, np.abs(KEFvals_traj[i, 0]), label=f'KEF {i+1}', c=f'C{i}')
            ax1.set_xlabel('Alpha')
            ax1.set_ylabel('KEF value')
            # ax1.legend()

            # Plot final PCA timepoint in bottom subplot
            pca_final = interpolated_trajectories_pca[-1,:,0]  # Get final timepoint
            ax2.plot(alpha, pca_final)
            ax2.set_xlabel('Alpha')
            ax2.set_ylabel('Final PC1')

            # plt.tight_layout()
            fig.savefig(Path(cfg.savepath) / "KEF_pca_vs_alpha.png", dpi=300)
            plt.show()
            plt.close(fig)




            # Sample every 100 timesteps
            n_trials = 40
            sample_every = 100
            sampled_hidden = hidden[::sample_every,:n_trials].detach().clone()
            sampled_inputs = inputs[::sample_every,:n_trials].detach().clone()

            # Add multivariate Gaussian noise with PCA covariance statistics
            pca_cov = torch.from_numpy(P.get_covariance()).float()
            mvn = torch.distributions.MultivariateNormal(
                loc=torch.zeros(sampled_hidden.shape[-1]),
                covariance_matrix=pca_cov
            )
            noise = mvn.sample(sampled_hidden.shape[:-1])
            perturbed_hidden = sampled_hidden + noise * 0.5

            # Reshape sampled tensors to 2D (combining first two dimensions)
            sampled_hidden_2d = sampled_hidden.reshape(-1, sampled_hidden.shape[-1])
            sampled_inputs_2d = sampled_inputs.reshape(-1, sampled_inputs.shape[-1])
            perturbed_hidden_2d = perturbed_hidden.reshape(-1, perturbed_hidden.shape[-1])

            # Set simulation length
            run_T = 100
            
            # Expand inputs by repeating along new middle dimension
            sampled_inputs_2d = sampled_inputs_2d.unsqueeze(0).expand(run_T, -1, -1)

            # Run RNN on perturbed initial conditions
            perturbed_trajectories = rnn(
                sampled_inputs_2d, #.to(rnn.device),
                x_init=perturbed_hidden_2d[None], #.to(rnn.device),
                deterministic=True,
                return_hidden=True,
            )[1]  # Only take second output, ignoring hidden states

            perturbed_trajectories_r = perturbed_trajectories.reshape(run_T, *perturbed_hidden.shape)
            # Expand sampled_inputs to match perturbed_trajectories_r shape
            sampled_inputs_expanded = sampled_inputs.unsqueeze(0).expand(perturbed_trajectories_r.shape[0], -1, -1, -1)
            # Concatenate perturbed trajectories and sampled inputs along last dimension
            perturbed_trajectories_r_inp = torch.cat([perturbed_trajectories_r, sampled_inputs_expanded], dim=-1)

            # Transform perturbed trajectories using PCA
            # P.transform(hidden)
            sampled_hidden_pca = P.transform(sampled_hidden.reshape(-1, sampled_hidden.shape[-1]).detach().cpu().numpy())
            sampled_hidden_pca = sampled_hidden_pca.reshape(*sampled_hidden.shape[:-1], -1)
            perturbed_hidden_pca = P.transform(perturbed_hidden.reshape(-1, perturbed_hidden.shape[-1]).detach().cpu().numpy())
            perturbed_hidden_pca = perturbed_hidden_pca.reshape(*perturbed_hidden.shape[:-1], -1)
            perturbed_trajectories_pca = P.transform(perturbed_trajectories_r.reshape(-1, perturbed_trajectories_r.shape[-1]).detach().cpu().numpy())
            perturbed_trajectories_pca = perturbed_trajectories_pca.reshape(*perturbed_trajectories_r.shape[:-1], -1)

            all_KEF_vals = []
            for i in range(SL.num_models):
                KEFvals = SL.models[i](perturbed_trajectories_r_inp.reshape(-1, perturbed_trajectories_r_inp.shape[-1]))
                KEFvals = KEFvals.reshape(*perturbed_trajectories_r_inp.shape[:-1],-1)
                all_KEF_vals.append(KEFvals)
            all_KEF_vals = torch.stack(all_KEF_vals, dim=0).detach().cpu()
            print(all_KEF_vals.shape)

            # Create subplots with 5 rows and 10 columns
            fig, axs = plt.subplots(5, 10, figsize=(20, 10), sharey='row',sharex='col')
            
            # Plot PCA trajectories in first row
            for j in range(10):  # For each trajectory
                axs[0,j].plot(perturbed_trajectories_pca[...,5*j,:,0])
                if j == 0:  # Only leftmost column gets y labels
                    axs[0,j].set_ylabel('PCA 1')
            
            # Plot each KEF value in remaining rows
            for i in range(4):  # For each model
                for j in range(10):  # For each trajectory
                    axs[i+1,j].plot(all_KEF_vals[i,:,5*j,:,0])
                    if i == 3:  # Only bottom row gets x labels
                        axs[i+1,j].set_xlabel('time')
                    if j == 0:  # Only leftmost column gets y labels
                        axs[i+1,j].set_ylabel(f'KEF {i+1}')
            
            plt.tight_layout()
            plt.savefig(Path(cfg.savepath) / "KEF_evolution.png", dpi=300)
            plt.close()


            # Reshape perturbed trajectories to match original shape
            plt.figure()
            neuron_id = 2
            for i in range(hidden.shape[1]):
                plt.plot(hidden[:, i, :].mean(-1).detach().cpu().numpy())
            plt.savefig(Path(cfg.savepath) / "hidden_traj.png",dpi=300)
            plt.close()
            print(
                'hidden[:5,0,0]:',hidden[:5,0,0]
            )

            # inputs = inputs[:2]
            # hidden = hidden[:2]
            #
            # select_every = 1
            # external_inputs_select = inputs.reshape(-1,inputs.shape[-1]).detach().clone()[::select_every]
            # hidden_select = hidden.reshape(-1, hidden.shape[-1]).detach().clone()[::select_every]
            # GD_traj, all_below_threshold_points, all_below_threshold_masks  = SL.find_separatrix(
            #     distribution,
            #     initial_conditions = hidden_select,
            #     external_inputs = external_inputs_select,
            #     external_input_dist = input_distribution,
            #     dist_needs_dim=cfg.dynamics.dist_requires_dim if hasattr(cfg.dynamics,
            #                                                              "dist_requires_dim") else True,
            #     return_indices = False,
            #     return_mask = True,
            #     **instantiate(cfg.separatrix_find_separatrix_kwargs)
            # )
            #
            #
            # KEFvals = []
            # delta_dists = []
            # delta_hiddens = []
            # for i in range(SL.num_models):
            #     mod_model = compose(
            #         torch.log,
            #         lambda x: x + 1,
            #         torch.exp,
            #         partial(torch.sum, dim=-1, keepdims=True),
            #         torch.log,
            #         torch.abs,
            #         SL.models[i]
            #     )
            #     samples_for_normalisation = 1000
            #     needs_dim = True
            #     if hasattr(cfg.dynamics, 'dist_requires_dim'):
            #         needs_dim = cfg.dynamics.dist_requires_dim
            #
            #     samples = dist.sample(
            #         sample_shape=[samples_for_normalisation] + ([cfg.dynamics.dim] if needs_dim else []))
            #     input_samples = input_distribution.sample(sample_shape=[samples_for_normalisation])
            #     samples = torch.cat((samples,input_samples),dim=-1)
            #     norm_val = float(
            #         torch.mean(torch.sum(mod_model(samples) ** 2, axis=-1)).sqrt().detach().numpy())
            #
            #     mod_model = compose(
            #         lambda x: x / norm_val,
            #         mod_model
            #     )
            #     input_to_KEF = hidden
            #     input_to_KEF = torch.cat([hidden,inputs],dim=-1)
            #     KEFval = mod_model(input_to_KEF).detach()
            #     KEFvals.append(KEFval)
            #
            #     below_threshold_points = all_below_threshold_points[i]
            #     below_threshold_mask = all_below_threshold_masks[i]
            #     hidden_reshaped = hidden.clone().detach().reshape(-1, hidden.shape[-1]).detach()
            #     hidden_reshaped[~below_threshold_mask] = torch.nan
            #     delta_hidden = torch.zeros_like(hidden_reshaped)
            #     delta_hidden[below_threshold_mask] = below_threshold_points[...,:hidden.shape[-1]] - hidden_reshaped[below_threshold_mask]
            #     delta_hidden[~below_threshold_mask] = torch.nan
            #     delta_hidden = delta_hidden.reshape(*hidden.shape)
            #     delta_hiddens.append(delta_hidden)
            #     delta_dist = torch.nanmean(
            #         (delta_hidden)**2,
            #         axis = -1
            #     )
            #     delta_dists.append(delta_dist)
            #
            #     # hidden_onlyvalid = hidden_reshaped.reshape(*hidden.shape)
            # print('len(delta_dists)',len(delta_dists))
            #
            # ### perturbation
            # scale = 1.0 #3.0
            # pert_rnn = instantiate(cfg.dynamics.perturbable_RNN_model)
            # delta_dists_st = torch.stack(delta_dists, axis=-1)
            # min_ids = np.argmin(np.nanmin(np.array(delta_dists_st), axis=-1), axis=0)
            # pert_inputs = torch.zeros((*delta_dists_st.shape[:2], hidden.shape[-1]))
            # random_pert_inputs = pert_inputs.clone()
            # for i in range(len(min_ids)):
            #     pert_vector = delta_hiddens[0][min_ids[i], i, :] * scale
            #     pert_inputs[min_ids[i]:min_ids[i]+3, i, :] = pert_vector[None]
            #     random_pert_inputs[min_ids[i]:min_ids[i]+3, i, :] = pert_vector[np.random.permutation(len(pert_vector))][None]
            # concat_inputs = torch.concat((inputs, pert_inputs), dim=-1)
            # random_concat_inputs = torch.concat((inputs, random_pert_inputs), dim=-1)
            # pert_outputs, pert_hidden = pert_rnn(concat_inputs, return_hidden=True)
            # random_pert_outputs, random_pert_hidden = pert_rnn(random_concat_inputs, return_hidden=True)
            #
            # KEFvals = torch.concatenate(KEFvals,axis=-1).detach().cpu().numpy()
            # inputs = inputs.detach().cpu().numpy()
            # targets = targets.detach().cpu().numpy()
            # outputs = outputs.detach().cpu().numpy()
            # pert_outputs = pert_outputs.detach().cpu().numpy()
            # random_pert_outputs = random_pert_outputs.detach().cpu().numpy()
            #
            #
            #
            #
            # fig, axes = plt.subplots(5, 10, sharex=True, sharey='row', figsize=(15, 12))
            #
            # for trial_num in range(axes.shape[1]):
            #     axs = axes[:, trial_num]
            #
            #     # Column Titles (Above First Row)
            #     axs[0].set_title(f"Trial-{trial_num}")
            #
            #     # First Row: Inputs
            #     ax = axs[0]
            #     ax.plot(inputs[:, trial_num])
            #     ax.spines['top'].set_visible(False)
            #     ax.spines['right'].set_visible(False)
            #     ax.spines['left'].set_bounds(-1, 1)
            #
            #     ax = axs[1]
            #     ax.plot(np.linalg.norm(pert_inputs[:,trial_num], axis=-1))
            #     ax.spines['top'].set_visible(False)
            #     ax.spines['right'].set_visible(False)
            #     ax.spines['left'].set_bounds(0, 1)
            #     ax.set_ylim(-0.1,1.1)
            #
            #
            #     # Second Row: Outputs/Targets
            #     # ax = axs[2]
            #     # ax.plot(targets[:, trial_num])
            #     # ax.plot(outputs[:, trial_num], ls='solid',label='No pert', alpha=0.7)
            #     # ax.plot(pert_outputs[:, trial_num], ls='dashed', label='Calc pert', alpha=0.7)
            #     # ax.plot(random_pert_outputs[:, trial_num], ls='dashed', label='Random pert', alpha=0.7)
            #     # ax.spines['top'].set_visible(False)
            #     # ax.spines['right'].set_visible(False)
            #     # ax.spines['left'].set_bounds(-1,1)
            #     # # ax.set_ylim(-0.1, 1.1)
            #
            #     # Third Row: KEF values
            #     ax = axs[3]
            #     ax.plot(KEFvals[:, trial_num])
            #     ax.spines['top'].set_visible(False)
            #     ax.spines['right'].set_visible(False)
            #     ax.set_yscale('log')
            #
            #     ## Fourth Row: dist to separatrix
            #     ax = axs[4]
            #     ax.plot(torch.stack(delta_dists,axis=-1)[:,trial_num],marker='o',markersize=1,alpha=0.5)
            #     # ax.plot(dist.log_prob(hidden[:, trial_num]).detach().cpu().numpy())
            #     # ax.set_ylabel('Log prob(hidden)')
            #     ax.spines['top'].set_visible(False)
            #     ax.spines['right'].set_visible(False)
            #     ax.set_yscale('log')
            #
            #
            #
            # # Set y-axis labels only for the first column
            # ylabel_texts = ['inputs', 'Norm of Pert input', 'outputs/targets', 'KEF values', 'Dist to Separatrix']
            # for row, label in enumerate(ylabel_texts):
            #     axes[row, 0].set_ylabel(label)
            #
            # axes[2,1].legend(fontsize=8)
            # # for ax in axes.flatten():
            # #     ax.set_xlim(230,300)
            #
            # fig.tight_layout()
            # fig.savefig(Path(cfg.savepath) / "RNN_task_KEFvals_sweep.png", dpi=300)
            # plt.close(fig)



            # P = PCA(n_components=3)
            # pc_hidden = P.fit_transform(hidden.reshape(-1,hidden.shape[-1]).detach().cpu().numpy())
            # pc_hidden = pc_hidden.reshape(*hidden.shape[:-1],P.n_components)
            #
            #
            # # fig,ax = plt.subplots()
            # fig = plt.figure(figsize=(6, 5))
            # ax = fig.add_subplot(111, projection='3d')
            #
            # n_lines = pc_hidden.shape[1]
            # import matplotlib.cm as cm
            # colors = cm.Purples(np.linspace(0.3, 0.9, n_lines))
            #
            # # fig, ax = plt.subplots()
            # for i in range(n_lines):
            #     ax.plot(*pc_hidden[100:, i, :].T, color=colors[i])
            #
            # for i in range(SL.num_models):
            #     below_threshold_points = all_below_threshold_points[i].detach().cpu().numpy()
            #     below_threshold_points = below_threshold_points[~np.isnan(below_threshold_points).any(axis=-1)]
            #     if len(below_threshold_points) == 0:
            #         continue
            #     pc_below_threshold_points = P.transform(below_threshold_points)
            #     ax.scatter(*pc_below_threshold_points[:,:].T,c=f'C{i}',s=10)
            #
            #
            # pc_unique_fps = P.transform(unique_fps.xstar)
            # ax.scatter(*pc_unique_fps[unique_fps.is_stable, :].T, c='blue',marker='x',s=100,zorder=1001)
            # ax.scatter(*pc_unique_fps[~unique_fps.is_stable, :].T, c='red',marker='x',s=100,zorder=1000) #
            #
            # fig.tight_layout()
            # fig.savefig(Path(cfg.savepath) / "trajectory_PCA.png", dpi=300)
            # # plt.close(fig)
            #
            # ##### Function to rotate the plot ######
            # def rotate(angle):
            #     ax.view_init(elev=30, azim=angle)
            #
            # # Create animation
            # num_frames = 360  # Number of frames for a full rotation
            # rotation_animation = animation.FuncAnimation(fig, rotate, frames=num_frames, interval=1000 / 30)
            #
            # # Save the animation to a file
            # rotation_animation.save(Path(cfg.savepath) / 'PCA_3d_rotation.mp4', writer='ffmpeg', fps=30, dpi=100)
            # plt.close(fig)


def main(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)


    OmegaConf.resolve(cfg.model)

    print(OmegaConf.to_yaml(cfg))

    F = instantiate(cfg.dynamics.function)
    dist = instantiate(cfg.dynamics.IC_distribution)
    model = instantiate(cfg.model)

    print(model)
    print(dict(model.named_parameters()))

    path = Path(cfg.savepath)
    path.mkdir(parents=True,exist_ok=True)

    if cfg.load_KEF_model:
        model = torch.load( path / (cfg.model.name+'_KEFmodel.torch'))
        #.load_state_dict(torch.load(os.path.join(cfg.savepath,'KEFmodel.torch'),weights_only=True))

    model.to(device)
    dist_kwargs = {}
    if hasattr(cfg.dynamics, 'dist_requires_dim'):
        dist_kwargs = {'dist_requires_dim': cfg.dynamics.dist_requires_dim}


    if hasattr(cfg,'train_func_teacher'):
        if hasattr(cfg.dynamics, 'analytical_eigenfunction'):
            train_func_teacher = instantiate(cfg.train_func_teacher)
            model.train()
            train_func_teacher(
                model,
                instantiate(cfg.dynamics.analytical_eigenfunction),
                dist,
                **dist_kwargs
            )

    if hasattr(cfg,'train_func'):
        train_func = instantiate(cfg.train_func)
        model.train()
        param_specific_hyperparams = []
        if hasattr(cfg.model, 'param_specific_hyperparams'):
            param_specific_hyperparams = instantiate(cfg.model.param_specific_hyperparams)
        train_func(
            model,
            F,
            dist,
            device=device,
            param_specific_hyperparams=param_specific_hyperparams,
            **dist_kwargs
        )
        model.eval()
    model.to('cpu')

    if hasattr(cfg,'train_func_trajectories'):
        train_func_trajectories = instantiate(cfg.train_func_trajectories)
        needs_dim = True
        if hasattr(cfg.dynamics,'dist_requires_dim'):
            needs_dim = cfg.dynamics.dist_requires_dim

        initial_conditions = dist.sample(sample_shape=[cfg.train_trajectories]+([cfg.dynamics.dim] if needs_dim else []))
        print(initial_conditions.shape)
        times = torch.linspace(0, cfg.train_trajectory_duration, cfg.train_points_per_trajectory)

        trajectories = odeint(lambda t, x: F(x), initial_conditions, times)
        trajectories = trajectories.swapaxes(0,1)
        train_func_trajectories(trajectories,model,times)

    if cfg.save_KEF_model:
        torch.save(model, path / (cfg.model.name+'_KEFmodel.torch'))

    test_func = instantiate(cfg.test_func)
    # with torch.no_grad():
    test_losses = torch.stack([test_func(model,F,dist,**dist_kwargs) for _ in range(20)])
    test_losses_mean = torch.mean(test_losses,axis=0).detach().cpu().numpy()
    test_losses_std = torch.std(test_losses,axis=0).detach().cpu().numpy()
    if cfg.save_results:
        results = {
            'test_losses_mean': list(test_losses_mean),
            'test_losses_std': list(test_losses_std),
            'test_loss_type': cfg.test_func.normaliser._target_,
            'model_name'    : cfg.model.name,
            **cfg.hyperparams_to_record_in_results
        }
        pd.DataFrame(results).to_csv(path / (cfg.model.name+'_results.csv'))

    GD_on_KEF_trajectories = None
    KEFvalues_GDtraj = None
    below_threshold_points = None
    KEFvalues_below_threshold_points = None

    model_to_GD_on = compose(
        # lambda x: x ** 0.1,
        # lambda x: x ** 2,
        torch.log,
        lambda x: x + 1,
        torch.exp,
        partial(torch.sum, dim=-1, keepdims=True),
        torch.log,
        torch.abs,
        model
    )

    samples_for_normalisation = 1000
    needs_dim = True
    if hasattr(cfg.dynamics, 'dist_requires_dim'):
        needs_dim = cfg.dynamics.dist_requires_dim

    samples = dist.sample(sample_shape=[samples_for_normalisation] + ([cfg.dynamics.dim] if needs_dim else []))
    norm_val = float(torch.mean(torch.sum(model_to_GD_on(samples)**2,axis=-1)).sqrt().detach().numpy())

    model_to_GD_on = compose(
        lambda x: x/norm_val,
        model_to_GD_on
    )

    if cfg.plot_KEF_of_traj:
        needs_dim = True
        if hasattr(cfg.dynamics,'dist_requires_dim'):
            needs_dim = cfg.dynamics.dist_requires_dim
        initial_conditions = dist.sample(sample_shape=[100]+([cfg.dynamics.dim] if needs_dim else []))
        times = torch.linspace(0, 5, 50)
        trajectories = odeint(lambda t, x: F(x), initial_conditions, times)
        fig, ax = plt.subplots()
        model_eval_phi_t = compose(
            # torch.log,
            # lambda x: x + 1,
            # torch.exp,
            partial(torch.sum, dim=-1, keepdims=True),
            torch.log,
            torch.abs,
            model
        )
        phi_vals = model_eval_phi_t(
            trajectories.reshape(-1,trajectories.shape[-1])
        ).reshape(*trajectories.shape[:2],-1).detach().cpu().numpy()
        print(phi_vals.shape)
        for i in range(trajectories.shape[1]):
            ax.plot(times, phi_vals[:, i, 0])
        # ax.plot([0, 5], [0, 5], ls='dashed', color='black')
        ax.set_ylabel(r'$\log \phi(t)$')
        ax.set_xlabel(r'$t$')
        plt.savefig( path / 'phi_t.png', dpi=300)

    if cfg.runGD:
        print('Running gradient descent on KEF landscape.')
        GD_on_KEF = instantiate(cfg.GD_on_KEF)

        GD_on_KEF_trajectories,below_threshold_points = GD_on_KEF(
            model_to_GD_on,
            dist,
            dist_needs_dim = (cfg.dynamics.dist_requires_dim if hasattr(cfg.dynamics,'dist_requires_dim') else True),
        )
        KEFvalues_GDtraj = model_to_GD_on(GD_on_KEF_trajectories.reshape(-1,GD_on_KEF_trajectories.shape[-1])).detach().cpu().numpy().reshape(*GD_on_KEF_trajectories.shape[:-1],-1)
        # KEFvalues_below_threshold_points = model_to_GD_on(below_threshold_points).detach().cpu().numpy()
        print('Gradient descent on KEF landscape complete.')
        print(f'Found {below_threshold_points.shape[0]} points below threshold.')

    print("Test loss:",test_losses_mean)
    if hasattr(cfg.dynamics,'analytical_eigenfunction'):
        analytical_eigenfunction = instantiate(cfg.dynamics.analytical_eigenfunction)
        test_loss_analytical = torch.mean(torch.tensor([test_func(analytical_eigenfunction, F, dist) for _ in range(20)]))
        print("Test loss analytical:", test_loss_analytical)


    if hasattr(cfg.dynamics,'analytical_eigenfunction'):
        num_trials = 10
        initial_conditions = dist.sample(sample_shape=(num_trials, cfg.dynamics.dim))
        times = torch.linspace(0, 10, 1000)

        trajectories = odeint(lambda t,x: F(x), initial_conditions, times)
        # trajectories = trajectories.detach().cpu().numpy()

        analytical_eigenfunction = instantiate(cfg.dynamics.analytical_eigenfunction)
        fig, ax = plt.subplots()
        phi_vals = analytical_eigenfunction(trajectories).detach().cpu() #.numpy()
        for i in range(trajectories.shape[1]):
            ax.plot(times, np.log(phi_vals[:, i, 0]))
        ax.plot([0, 5], [0, 5], ls='dashed', color='black')
        ax.set_ylabel(r'$\phi(t)$')
        ax.set_xlabel(r'$t$')
        plt.savefig(path / 'analytical_phi_t.png', dpi=300)
        plt.close(fig)


        if cfg.runGD_analytical:
            fig, axs = plt.subplots(2, 2, figsize=(8, 8),sharex=True)

            GD_on_KEF = instantiate(cfg.GD_on_KEF)

            mu_vals = [-0.5,0,0.5,1]
            for i,mu in enumerate(mu_vals):
                ax=axs.flatten()[i]
                cfg.dynamics.analytical_eigenfunction.mu = mu
                analytical_eigenfunction = instantiate(cfg.dynamics.analytical_eigenfunction)
                trajectories = GD_on_KEF(analytical_eigenfunction, dist)
                ax.plot(trajectories[...,0], trajectories[...,1], color='grey', alpha=0.3)
                ax.scatter(trajectories[0, :, 0], trajectories[0, :, 1], color='green', alpha=0.3)
                ax.scatter(trajectories[-1, :, 0], trajectories[-1, :, 1], color='red')
                ax.set_title(rf'$\mu={mu}$')

            x_limits = (-2, 2)  # Limits for x-axis
            y_limits = (-2, 2)  # Limits for y-axis
            if hasattr(cfg.dynamics, 'lims'):
                x_limits = cfg.dynamics.lims.x
                y_limits = cfg.dynamics.lims.y
            for ax in axs.flatten():
                ax.set_xlim(*x_limits)
                ax.set_ylim(*y_limits)
            fig.tight_layout()
            fig.savefig(path / 'GD_on_KEF_trajectories.png', dpi=300)


    if not cfg.run_analysis:
        return

    if cfg.dynamics.dim == 1:
        x = torch.linspace(-15, 15, 1000,dtype=torch.float32)
        x.requires_grad_(True)
        phi_val = model(x[:, None])
        F_val = F(x[:, None])

        F_val = F_val.detach().cpu().numpy()[...,0]


        phi_x_prime = torch.autograd.grad(
            outputs=phi_val[:,:].sum(axis=-1),
            inputs=x,
            grad_outputs=torch.ones_like(phi_val[:,0]),
            create_graph= True
        )[0]
        phi_val = phi_val.detach().cpu().numpy()[...,0]
        x = x.detach().cpu().numpy()
        phi_x_prime = phi_x_prime.detach().cpu().numpy()

        fig,axs = plt.subplots(4 + int(below_threshold_points is not None),1,figsize=(4,7),sharex=True)
        ax = axs[0]
        ax.plot(x, 0 * x, c='grey', lw=1)
        ax.plot(x, F_val, label='F')

        ax = axs[1]
        ax.plot(x, np.abs(phi_val)/np.sqrt((phi_val**2).mean()), label=f'$\phi$')
        if hasattr(cfg.dynamics, 'analytical_eigenfunction'):
            analytical_eigenfunction = instantiate(cfg.dynamics.analytical_eigenfunction)
            ana_phi_val = analytical_eigenfunction(x)
            ax.plot(x,np.abs(ana_phi_val)/np.sqrt(ana_phi_val**2).mean(),ls='dashed',color='black',alpha=0.5)

        ax = axs[2]
        print(phi_x_prime.shape,F_val.shape,phi_val.shape)
        diff = (phi_x_prime * F_val) - 1 * phi_val
        diff /= diff.std()
        ax.plot(
            x,
            # (phi_x_prime*F_val.flatten())[...,None]-1*phi_val,
            np.abs(diff),
            label=r'$\nabla \phi-\lambda \phi$',
            lw=1
        )

        ax = axs[3]
        # print(phi_x_prime.shape,F_val.shape,phi_val.shape)
        dot_prods = (phi_x_prime * F_val) # - 1 * phi_val

        ax.plot(
            x,
            np.abs(dot_prods),
            label=r'$\nabla \phi \cdot F$',
            lw=1,
            c='red'
        )
        ax.plot(
            x,
            np.abs(phi_val),
            label=r'$\phi$',
            lw=1,
            c='blue'
        )
        # ax.set_yscale('log')
        ax.legend()


        if below_threshold_points is not None:
            ax = axs[4]
            ax.hist(below_threshold_points,density=True,bins=100)

        # plt.legend()
        for ax in axs.flatten():
            ax.set_xlim(-5, 5)
            ax.set_ylim(0, 5)
        axs[0].set_ylim(-2,2)
        fig.tight_layout()
        fig.savefig(path / 'F_and_phi.png',dpi=300)
        plt.close(fig)


        if model.__class__.__name__ == "RBFLayer":
            fig,ax = plt.subplots()
            D = pd.DataFrame({
                'kernels_centers': model.get_kernels_centers.detach().cpu().numpy().flatten(),
                'weights': torch.abs(model.get_weights.detach().cpu()).numpy().flatten(),
                'log shapes': torch.log(model.get_shapes.detach()).cpu().numpy().flatten(),
                'shapes': (model.get_shapes.detach()).cpu().numpy().flatten(),
            })
            sns.scatterplot(x='kernels_centers',y='log shapes',size='weights',data=D,ax=ax)
            fig.savefig(path / 'RBF_data.png',dpi=300)
            # plt.show()


    elif cfg.dynamics.dim == 2:
        if KEFvalues_GDtraj is not None:
            fig,ax = plt.subplots()
            t = np.arange(KEFvalues_GDtraj.shape[0])
            ax.plot(t,KEFvalues_GDtraj[:,:,0])
            fig.tight_layout()
            fig.savefig(path / 'KEFvalues_GDtraj.png')



        # Configurable parameters
        heatmap_resolution = 500  # Resolution for the heatmap
        quiver_resolution = 25  # Resolution for the quiver plot
        laplace_resolution = 50

        x_limits = (-2, 2)  # Limits for x-axis
        y_limits = (-2, 2)  # Limits for y-axis
        if hasattr(cfg.dynamics,'lims'):
            x_limits = cfg.dynamics.lims.x
            y_limits = cfg.dynamics.lims.y

        num_trials = 100
        initial_conditions = torch.concat([
            torch.rand(size=(num_trials, 1)) * (x_limits[1] - x_limits[0]) + x_limits[0],
            torch.rand(size=(num_trials, 1)) * (y_limits[1] - y_limits[0]) + y_limits[0]
        ],axis=-1)
        times = torch.linspace(0, 10, 100)
        trajectories = None
        if hasattr(cfg.dynamics,'run_traj'):
            if cfg.dynamics.run_traj:
                trajectories = odeint(lambda t,x: F(x), initial_conditions, times)
                trajectories = trajectories.detach().cpu().numpy()

        # Define grid for heatmap (higher resolution)
        x_heatmap = torch.linspace(x_limits[0], x_limits[1], heatmap_resolution)
        y_heatmap = torch.linspace(y_limits[0], y_limits[1], heatmap_resolution)
        X_heatmap, Y_heatmap = torch.meshgrid(x_heatmap, y_heatmap, indexing='ij')
        heatmap_grid = torch.stack([X_heatmap.flatten(), Y_heatmap.flatten()], dim=-1)

        # Define grid for laplace (higher resolution)
        x_laplace = torch.linspace(x_limits[0], x_limits[1], laplace_resolution)
        y_laplace = torch.linspace(y_limits[0], y_limits[1], laplace_resolution)
        X_laplace, Y_laplace = torch.meshgrid(x_laplace, y_laplace, indexing='ij')
        laplace_grid = torch.stack([X_laplace.flatten(), Y_laplace.flatten()], dim=-1)

        # Define grid for quiver plot (lower resolution)
        x_quiver = torch.linspace(x_limits[0], x_limits[1], quiver_resolution)
        y_quiver = torch.linspace(y_limits[0], y_limits[1], quiver_resolution)
        X_quiver, Y_quiver = torch.meshgrid(x_quiver, y_quiver, indexing='ij')
        quiver_grid = torch.stack([X_quiver.flatten(), Y_quiver.flatten()], dim=-1)

        # Evaluate trajectories from IC and compute Laplace integral
        tau = 0.1
        # times = torch.linspace(0, 5, 100)
        # laplace_trajectories = odeint(lambda t,x: F(x), laplace_grid, times,rtol=1e-10, atol=1e-10)
        # observable = lambda x: x[...,1]
        center = torch.tensor([0.0,-6.0])
        center = torch.tensor([0.0, 3.0])
        observable = lambda x: (torch.log((x - center[None,None,:])**2)).mean(axis=-1)
        # laplace_integrals = (observable(laplace_trajectories) * torch.exp(-tau * times[:,None])).mean(axis=0)
        # laplace_integrals = laplace_integrals.detach().cpu().numpy().reshape(laplace_resolution, laplace_resolution)

        # Compute F values for the quiver grid
        F_val_quiver = F(quiver_grid).detach().cpu().numpy()
        Fx_quiver = F_val_quiver[:, 0].reshape(quiver_resolution, quiver_resolution)
        Fy_quiver = F_val_quiver[:, 1].reshape(quiver_resolution, quiver_resolution)

        # Compute F values for the heatmap grid (for kinetic energy)
        F_val_heatmap = F(heatmap_grid).detach().cpu().numpy()
        Fx_heatmap = F_val_heatmap[:, 0].reshape(heatmap_resolution, heatmap_resolution)
        Fy_heatmap = F_val_heatmap[:, 1].reshape(heatmap_resolution, heatmap_resolution)
        kinetic_energy = Fx_heatmap ** 2 + Fy_heatmap ** 2

        # Compute phi values for the heatmap grid
        # phi_val = torch.prod( model(heatmap_grid),dim=-1).detach().cpu().numpy()
        phi_val = model_to_GD_on(heatmap_grid).detach().cpu().numpy()
        phi_val = phi_val.reshape(heatmap_resolution, heatmap_resolution)

        # heatmap_grid.requires_grad_(True)
        # phi_val_2 = model(heatmap_grid).detach().cpu().numpy()
        # phi_val_2_prime = torch.autograd.grad(
        #     outputs=phi_val_2.sum(axis=-1),
        #     inputs=heatmap_grid,
        #     grad_outputs=torch.ones_like(phi_val_2.sum(axis=-1)),
        #     create_graph=False  # True
        # )[0]
        # phi_val_2_prime


        # Set up the figure with two side-by-side subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Left subplot: Kinetic energy of F with quiver plot
        im1 = axes[0].imshow(
            np.log(kinetic_energy).T, extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
            origin='lower', aspect='auto', cmap='plasma'
        )
        axes[0].quiver(
            X_quiver, Y_quiver, Fx_quiver, Fy_quiver, color='white', scale=50, width=0.002, headwidth=3,
            pivot='middle'
        )
        if trajectories is not None:
            axes[0].plot(trajectories[..., 0], trajectories[..., 1],c='grey', lw=1)
        axes[0].set_title('Kinetic Energy and Vector Field of $F(x, y)$')
        axes[0].set_xlabel('$x$')
        axes[0].set_ylabel('$y$')
        if below_threshold_points is not None:
            # print('below_threshold_points',below_threshold_points)
            axes[0].scatter(below_threshold_points[ :, 0],
                            below_threshold_points[ :, 1], c='red', zorder=1000)
        fig.colorbar(im1, ax=axes[0], label='Kinetic Energy $||F(x, y)||^2$')

        # Right subplot: Heatmap of phi with contour for phi(x, y) = 0
        # im2 = axes[1].imshow(
        #     phi_val.T, extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
        #     origin='lower', aspect='auto', cmap='viridis'
        # )
        # contour = axes[1].contour(
        #     X_heatmap, Y_heatmap, phi_val, levels=[0,1,2,3], colors='red', linewidths=2
        # )
        contour = axes[1].contourf(
            X_heatmap, Y_heatmap, phi_val,
        )
        # if GD_on_KEF_trajectories is not None:
        #     axes[1].plot(GD_on_KEF_trajectories[..., 0],
        #                  GD_on_KEF_trajectories[..., 1], c='grey', lw=0.5,alpha=0.3)
        #     axes[1].scatter(GD_on_KEF_trajectories[-1, :, 0],
        #                GD_on_KEF_trajectories[-1, :, 1], c='red',zorder=1000)
        if below_threshold_points is not None:
            # print('below_threshold_points',below_threshold_points)
            axes[1].scatter(below_threshold_points[ :, 0],
                            below_threshold_points[ :, 1], c='red', zorder=1000)
            # print('Plotting below_threshold_points:',below_threshold_points)
        # axes[1].clabel(contour, inline=True, fontsize=10)
        axes[1].set_title('Contour plot of $\log \phi(x, y)$')
        axes[1].set_xlabel('$x$')
        axes[1].set_ylabel('$y$')


        fig.colorbar(contour, ax=axes[1], label='$\phi(x, y)$')

        contour1 = axes[2].contourf(
            X_heatmap, Y_heatmap, np.abs(phi_val) ** 0.1,
        )
        # axes[2].clabel(contour1, inline=True, fontsize=10)
        axes[2].set_title('Contour plot of $\phi(x, y)$')
        axes[2].set_xlabel('$x$')
        axes[2].set_ylabel('$y$')

        fig.colorbar(contour1, ax=axes[2], label='$\phi(x, y)^{0.1}$')

        if hasattr(cfg.dynamics,"analytical_eigenfunction"):
            analytical_phi_func = instantiate(cfg.dynamics.analytical_eigenfunction)
            phi_val = analytical_phi_func(heatmap_grid).detach().cpu().numpy()
            phi_val = phi_val.reshape(heatmap_resolution, heatmap_resolution)
            print('phi_val range',phi_val.min(), phi_val.max(), phi_val.mean())
            phi_val = np.clip(phi_val, 0, 1.0)
            label = r'$\phi^{ana}(x, y)$'
        else:
            phi_val = np.abs(phi_val)**0.05
            label = r'$\phi(x, y)^{0.05}$'



        contour2 = axes[3].contourf(
            X_heatmap, Y_heatmap, (np.abs(phi_val)),
        )
        # axes[3].clabel(contour2, inline=True, fontsize=10)
        axes[3].set_title('Contour plot of $\phi(x, y)$')
        axes[3].set_xlabel('$x$')
        axes[3].set_ylabel('$y$')

        for ax in axes:
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
        fig.colorbar(contour2, ax=axes[3], label=label)


        # plot laplace integrals
        # contour = axes[2].contourf(
        #     X_laplace, Y_laplace, laplace_integrals, linewidths=2
        # )
        # axes[2].set_title('Contour plot of Laplace integral $f^*_1(x, y)$')
        # axes[2].set_xlabel('$x$')
        # axes[2].set_ylabel('$y$')
        # for ax in axes:
        #     ax.set_xlim(*x_limits)
        #     ax.set_ylim(*y_limits)
        # fig.colorbar(contour, ax=axes[2], label='$f^*_1(x, y)$')


        # Save the figure with both subplots
        fig.tight_layout()
        fig.savefig(path / 'F_and_phi_subplots.png',dpi=300)
        fig.savefig(path / 'F_and_phi_subplots.pdf')
        plt.close(fig)

        ### Plotting all
        phi_vals = model(heatmap_grid).detach().cpu().numpy()

        if phi_vals.shape[1]>=10:
            fig,axs = plt.subplots(2,5,figsize=(10, 8),sharey=True,sharex=True)
            for i,ax in enumerate(axs.flatten()):
                phi_val = phi_vals[..., i]
                phi_val = phi_val.reshape(heatmap_resolution, heatmap_resolution)
                contour = ax.contourf(
                    X_heatmap, Y_heatmap, np.log(np.abs(phi_val)),
                )
            fig.tight_layout()
            fig.savefig(path / 'all_phi.png',dpi=300)

        x = torch.linspace(-1,1,100)
        inp = torch.stack([x,torch.zeros_like(x)], dim=-1)
        phi_val = model(inp).detach().cpu().numpy()

        plt.figure()
        plt.plot(x,phi_val, label=f'$\phi$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\phi(x,0)$')
        plt.tight_layout()
        plt.savefig(path / 'phi(x,0).png',dpi=300)










    elif cfg.dynamics.dim > 2:
        trajectories = None
        KEFvaltraj = None
        if hasattr(cfg.dynamics, 'run_traj'):
            if cfg.dynamics.run_traj:
                num_trials = 50
                times = torch.linspace(0, 5, 100)
                needs_dim = True
                if hasattr(cfg.dynamics, 'dist_requires_dim'):
                    needs_dim = cfg.dynamics.dist_requires_dim

                initial_conditions = dist.sample(
                    sample_shape=[num_trials] + ([cfg.dynamics.dim] if needs_dim else []))

                trajectories = odeint(lambda t, x: F(x), initial_conditions, times)
                KEFvaltraj = compose(torch.abs, model)(trajectories).detach().cpu().numpy()
                trajectories = trajectories.detach().cpu().numpy()

        if trajectories is not None:
            fig,ax = plt.subplots()
            ax.plot(times, KEFvaltraj[...,0], c='grey', lw=1)
            fig.savefig(path / "KEF_of_trajectories.png",dpi=300)

        rnn_model = instantiate(cfg.dynamics.loaded_RNN_model)
        # cfg.dynamics.RNN_dataset.batch_size = 5000
        cfg.dynamics.RNN_dataset.n_trials = 1000
        dataset = instantiate(cfg.dynamics.RNN_dataset)
        inp, targ = dataset()

        torch_inp = torch.from_numpy(inp).type(torch.float) #.to(device)
        outputs,hidden_traj = rnn_model(torch_inp,return_hidden=True)
        outputs,hidden_traj = outputs.detach().cpu().numpy(), hidden_traj.detach().cpu().numpy()


        # print(model)
        fpf_hps = {
            'max_iters': 1000, #10000
            'n_iters_per_print_update': 1000,
            'lr_init': .1,
            'outlier_distance_scale': 10.0,
            'verbose': True,
            'super_verbose': True,
            # 'tol_q':1e-6,
            # 'tol_q': 1e-15,
            # 'tol_dq': 1e-15,
        }
        FPF = FixedPointFinderTorch(
            rnn_model.rnn,
            **fpf_hps
        )
        num_trials = 1000
        # initial_conditions = dist.sample(sample_shape=(num_trials,)).detach().cpu().numpy()
        inputs = np.zeros((1, cfg.dynamics.RNN_model.act_size))
        # inputs[...,0] = 1
        initial_conditions = hidden_traj.reshape(-1,hidden_traj.shape[-1])
        initial_conditions = initial_conditions[np.random.choice(initial_conditions.shape[0],size=num_trials,replace=False)]
        initial_conditions += np.random.normal(size=initial_conditions.shape) * 0.05
        # print('initial_conditions', initial_conditions.shape)
        unique_fps,all_fps = FPF.find_fixed_points(
            initial_conditions,
            inputs
        )
        # print(all_fps.shape)
        fixed_point_data = pd.DataFrame({
            'stability': unique_fps.is_stable,
            'q': unique_fps.qstar,
            'KEF': model_to_GD_on(torch.from_numpy(unique_fps.xstar).type(torch.float)).detach().cpu().numpy()[...,0],
        })
        fixed_point_data.to_csv(path / 'fixed_point_data.csv',index=False)


        P = PCA(n_components=3)
        pc_traj = P.fit_transform(hidden_traj.reshape(-1, hidden_traj.shape[-1])).reshape(*hidden_traj.shape[:-1],P.n_components)
        # pc_initial_conditions = P.fit_transform(initial_conditions)
        pc_initial_conditions = P.transform(initial_conditions)
        pc_fps = P.transform(all_fps.xstar)
        pc_unique_fps = P.transform(unique_fps.xstar)
        # where_best = (unique_fps.qstar < np.median(unique_fps.qstar))
        where_best = True
        # pc_traj = P.transform(hidden_traj.reshape(-1,hidden_traj.shape[-1])).reshape(*hidden_traj.shape[:-1],P.n_components)





        # fig,axs = plt.subplots(figsize=(4,3))
        # ax = axs
        # # ax.scatter(pc_initial_conditions[:,0],pc_initial_conditions[:,1],size=10)
        # # Plot and capture scatter plot artists
        #
        # scatter1 = ax.scatter(pc_unique_fps[unique_fps.is_stable & where_best, 0], pc_unique_fps[unique_fps.is_stable & where_best, 1], s=5,
        #                        c='C0')
        # scatter2 = ax.scatter(pc_unique_fps[(~unique_fps.is_stable) & where_best, 0], pc_unique_fps[(~unique_fps.is_stable) & where_best, 1], s=5,
        #                       c='C1')
        #
        # ax.plot(pc_traj[:, :100, 0], pc_traj[:, :100, 1], lw=0.5, c='grey', alpha=0.1)
        #
        # where_decide_1 = np.argmax(outputs, axis=-1) == 1
        # where_decide_2 = np.argmax(outputs, axis=-1) == 2
        #
        #
        # scatter3 = ax.scatter(pc_traj[..., 0][where_decide_1][:100], pc_traj[..., 1][where_decide_1][:100], c='C3',s=10)
        # scatter4 = ax.scatter(pc_traj[..., 0][where_decide_2][:100], pc_traj[..., 1][where_decide_2][:100], c='C4',s=10)
        #
        # if GD_on_KEF_trajectories is not None:
        #     GD_on_KEF = instantiate(cfg.GD_on_KEF)
        #     GD_on_KEF_trajectories = GD_on_KEF(
        #         compose(torch.abs, model),
        #         dist,
        #         initial_conditions = torch.from_numpy(initial_conditions).type(torch.float),
        #     )
        #
        #
        #     pc_GD_on_KEF_trajectories = P.transform(GD_on_KEF_trajectories.reshape(-1,GD_on_KEF_trajectories.shape[-1])).reshape(*GD_on_KEF_trajectories.shape[:-1],P.n_components)
        #     ax.plot(pc_GD_on_KEF_trajectories[:, :, 0], pc_GD_on_KEF_trajectories[:, :, 1], c='C5',lw=0.5,alpha=0.5)
        #     scatter5 = ax.scatter(pc_GD_on_KEF_trajectories[-1,:,0],pc_GD_on_KEF_trajectories[-1,:,1],c='C5',s=10)

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        # Plot and capture scatter plot artists in 3D
        scatter1 = ax.scatter(
            pc_unique_fps[unique_fps.is_stable & where_best, 0],
            pc_unique_fps[unique_fps.is_stable & where_best, 1],
            pc_unique_fps[unique_fps.is_stable & where_best, 2],
            s=5, c='C0'
        )
        scatter2 = ax.scatter(
            pc_unique_fps[(~unique_fps.is_stable) & where_best, 0],
            pc_unique_fps[(~unique_fps.is_stable) & where_best, 1],
            pc_unique_fps[(~unique_fps.is_stable) & where_best, 2],
            s=5, c='C1'
        )

        ax.plot(
            pc_traj[:, :10, 0],
            pc_traj[:, :10, 1],
            pc_traj[:, :10, 2],
            lw=0.5, c='grey', alpha=0.5
        )

        # where_decide_1 = np.argmax(outputs, axis=-1) == 1
        # where_decide_2 = np.argmax(outputs, axis=-1) == 2

        # scatter3 = ax.scatter(
        #     pc_traj[..., 0][where_decide_1][:100],
        #     pc_traj[..., 1][where_decide_1][:100],
        #     pc_traj[..., 2][where_decide_1][:100],
        #     c='C3', s=10
        # )
        # scatter4 = ax.scatter(
        #     pc_traj[..., 0][where_decide_2][:100],
        #     pc_traj[..., 1][where_decide_2][:100],
        #     pc_traj[..., 2][where_decide_2][:100],
        #     c='C4', s=10
        # )




        # Handle GD_on_KEF_trajectories if not None
        if GD_on_KEF_trajectories is not None:
            # GD_on_KEF = instantiate(cfg.GD_on_KEF)
            # GD_on_KEF_trajectories = GD_on_KEF(
            #     compose(torch.abs, model),
            #     dist,
            #     initial_conditions=torch.from_numpy(initial_conditions).type(torch.float),
            # )
            final_KEFvals = compose(torch.log,torch.abs, model)(GD_on_KEF_trajectories[-1]).detach()
            select_best = final_KEFvals.flatten()<torch.quantile(final_KEFvals,.05)
            print(
                'Max KEF val', final_KEFvals.max(),
                'Min KEF val', final_KEFvals.min(),
                'Median KEF val', torch.quantile(final_KEFvals,.05),
            )

            pc_GD_on_KEF_trajectories = P.transform(
                GD_on_KEF_trajectories.reshape(-1, GD_on_KEF_trajectories.shape[-1])).reshape(
                *GD_on_KEF_trajectories.shape[:-1], P.n_components
            )

            if trajectories is not None:
                pc_trajectories = P.transform(
                    trajectories.reshape(-1, trajectories.shape[-1])).reshape(
                    *trajectories.shape[:-1], P.n_components
                )
                for i in np.arange(pc_trajectories.shape[1]):
                    traj_l,=ax.plot(
                        pc_trajectories[:, i, 0],
                        pc_trajectories[:, i, 1],
                        pc_trajectories[:, i, 2],
                        c='C6', lw=0.5, alpha=0.5
                    )

            # for i in np.where(select_best)[0]:
            #     ax.plot(
            #         pc_GD_on_KEF_trajectories[[0,-1], i, 0],
            #         pc_GD_on_KEF_trajectories[[0,-1], i, 1],
            #         pc_GD_on_KEF_trajectories[[0,-1], i, 2],
            #         c='C5', lw=0.5, alpha=0.5
            #     )

            # scatter5 = ax.scatter(
            #     pc_GD_on_KEF_trajectories[-1, select_best, 0],
            #     pc_GD_on_KEF_trajectories[-1, select_best, 1],
            #     pc_GD_on_KEF_trajectories[-1, select_best, 2],
            #     c='C5', s=10,alpha=0.5
            # )
            if below_threshold_points is not None:
                pc_below_threshold_points = P.transform(below_threshold_points)
                scatter5 = ax.scatter(
                    pc_below_threshold_points[:, 0],
                    pc_below_threshold_points[:, 1],
                    pc_below_threshold_points[:, 2],
                    c='red', s=10,alpha=0.5
                )
            # scatter5 = ax.scatter(
            #     pc_GD_on_KEF_trajectories[-1, select_best, 0],
            #     pc_GD_on_KEF_trajectories[-1, select_best, 1],
            #     pc_GD_on_KEF_trajectories[-1, select_best, 2],
            #     c='red', s=10, alpha=0.5
            # )


        # Add legend using scatter plot artists
        ax.legend(
            [scatter1, scatter2], #, scatter5],#, scatter3, scatter4],
            ['Stable Fixed Points', 'Unstable Fixed Points','KEF minima'], #'Report Decision 1', 'Report Decision 2'
            loc='best',
            fontsize='small'
        )

        # Add legend to the plot
        # ax.legend(handles=legend_handles, loc='best', fontsize='small')
        fig.tight_layout()

        fig.savefig(path / 'PCA_3d.png',dpi=300)




        ##### Function to rotate the plot ######
        def rotate(angle):
            ax.view_init(elev=30, azim=angle)

        # Create animation
        num_frames = 360  # Number of frames for a full rotation
        rotation_animation = animation.FuncAnimation(fig, rotate, frames=num_frames, interval=1000/30)

        # Save the animation to a file
        rotation_animation.save(path / 'PCA_3d_rotation.mp4', writer='ffmpeg', fps=30, dpi=100)
        plt.close(fig)



        # fig,ax = plt.subplots(figsize=(4,3))
        # ax.hist(unique_fps.qstar, bins=np.logspace(-7,-3,50))
        # ax.axvline(np.median(unique_fps.qstar), color='black',linestyle='--',lw=1,label='median')
        # ax.legend()
        # ax.set_xlabel(r'$q$')
        # ax.set_xscale('log')
        # fig.tight_layout()
        # fig.savefig(path / 'qstar_hist.png',dpi=200)

        # fig,ax = plt.subplots(figsize=(4,3))
        # bins = np.linspace(-3.5,2.5,101)
        # ax.hist(pc_traj[..., 0][where_decide_1|where_decide_2], bins=bins, color='C0')  # , bins=np.logspace(-7,-3,50))
        # # ax.hist(pc_traj[..., 0][where_decide_1], bins=bins, label='Report decision 1', color='C3',alpha=0.5) #, bins=np.logspace(-7,-3,50))
        # # ax.hist(pc_traj[..., 0][where_decide_2], bins=bins, label='Report decision 2', color='C4',alpha=0.5)  # , bins=np.logspace(-7,-3,50))
        # # ax.axvline(np.median(unique_fps.qstar), color='black',linestyle='--',lw=1,label='median')
        # # ax.legend()
        # ax.set_xlabel(r'PC1 decision points')
        # # ax.set_xscale('log')
        # fig.tight_layout()
        # fig.savefig(path / 'PC1_decision_hist.png',dpi=200)




        # # A = F.functions[0].keywords['A']
        # # initial_conditions = 15 * torch.randn(size=(num_trials, A.shape[-1]))[:,:2] @ A.T[:2]
        # initial_conditions = dist.sample(sample_shape=(num_trials, cfg.dynamics.dim))
        # times = torch.linspace(0, 2, 500)
        # trajectories = odeint(lambda t, x: F(x), initial_conditions, times).detach()
        # # print(torch.mean(trajectories**2,dim=(-1))[-1])
        # # print(trajectories[-1,0,:])
        #
        # # Reshape the data for PCA (combine timesteps and trials into one axis)
        # data = trajectories.reshape(-1, trajectories.shape[-1]).numpy()  # Shape: [6000, 20]
        #
        # # Perform PCA to reduce the dimensionality to 2 for 2D visualization
        # pca = PCA(n_components=2)
        # # data_pca = pca.fit_transform(data)  # Shape: [6000, 2]
        # pca.fit(initial_conditions.detach().cpu().numpy())
        # data_pca = pca.transform(data)
        #
        # # Reshape back to [500, 100, 2] for plotting
        # data_pca = data_pca.reshape(times.shape[0], num_trials, 2)
        #
        # # Define the range for the PCA plane
        # pc1_min, pc1_max = np.min(data_pca[:, :, 0]), np.max(data_pca[:, :, 0])
        # pc2_min, pc2_max = np.min(data_pca[:, :, 1]), np.max(data_pca[:, :, 1])
        #
        # # Calculate the range for each principal component
        # pc1_range = pc1_max - pc1_min
        # pc2_range = pc2_max - pc2_min
        #
        # # Extend the range by 5%
        # pc1_min = pc1_min - 0.05 * pc1_range
        # pc1_max = pc1_max + 0.05 * pc1_range
        # pc2_min = pc2_min - 0.05 * pc2_range
        # pc2_max = pc2_max + 0.05 * pc2_range
        #
        # # Create a grid in the PCA space
        # resolution = 100
        # pc1 = np.linspace(pc1_min, pc1_max, resolution)
        # pc2 = np.linspace(pc2_min, pc2_max, resolution)
        # PC1, PC2 = np.meshgrid(pc1, pc2)
        #
        # # Flatten the grid for evaluation
        # pc_points = np.stack([PC1.ravel(), PC2.ravel()], axis=1)  # Shape: [resolution^2, 2]
        #
        # # Map PCA points back to the original N-dimensional space
        # original_space_points = pca.inverse_transform(pc_points)  # No need for zero-padding
        #
        # # Ensure the model evaluates only the grid points
        # with torch.no_grad():
        #     model_output = model(torch.tensor(original_space_points, dtype=torch.float32)).numpy()
        #
        # # Ensure the model output size matches the grid
        # assert model_output.size == PC1.size, f"Expected {PC1.size}, but got {model_output.size}"
        #
        # # Reshape model output to match the grid
        # model_output = model_output.reshape(PC1.shape)
        #
        # # print(model_output.min())
        #
        # # Plot the trajectories in the first two principal components
        # plt.figure(figsize=(10, 8))
        #
        # levels = np.arange(0,1.5,0.1) #np.linspace(model_output.min(),model_output.max(),4)
        # # Plot the contour of the model
        # zero_contour = plt.contour(
        #     PC1, PC2, model_output, levels=[0], colors='red', linewidths=2
        # )
        # plt.clabel(zero_contour, inline=True, fontsize=10)
        # contour = plt.contourf(PC1, PC2, np.abs(model_output), levels=levels, cmap='viridis', alpha=0.7)
        # plt.colorbar(contour, label=r"$|\phi|$")
        #
        # for trial in range(data_pca.shape[1]):  # Iterate over trials
        #     plt.plot(
        #         data_pca[:, trial, 0],  # PC1
        #         data_pca[:, trial, 1],  # PC2
        #         alpha=0.7,
        #         c='grey',
        #         lw=1
        #     )
        #     plt.scatter(
        #         data_pca[0, trial, 0],  # PC1
        #         data_pca[0, trial, 1],  # PC2
        #         c='green'
        #     )
        #     plt.scatter(
        #         data_pca[-1, trial, 0],  # PC1
        #         data_pca[-1, trial, 1],  # PC2
        #         c='red'
        #     )
        #
        # # Set labels and title
        # plt.xlabel("PC1")
        # plt.ylabel("PC2")
        # plt.title("Trajectories and Model Contour in PCA-reduced Space (2D)")
        #
        # # Save the plot
        # plt.savefig(path / 'trajectories_with_contour.png', dpi=300)
        # plt.savefig(path / 'trajectories_with_contour.pdf')










        # num_trials = 100
        # A = F.functions[0].keywords['A']
        # # initial_conditions = 10*A.T #2 *
        # initial_conditions = 15 * torch.randn(size=(num_trials, A.shape[-1])) @ A.T
        # # initial_conditions = 2 * torch.randn(size=(num_trials, cfg.dynamics.dim))# @ A.T
        # times = torch.linspace(0, 2, 500)
        # trajectories = odeint(lambda t,x: F(x), initial_conditions, times)
        #
        #
        # # Reshape the data for PCA (combine timesteps and trials into one axis)
        # data = trajectories.reshape(-1, trajectories.shape[-1]).numpy()  # Shape: [6000, 20]
        #
        # # Perform PCA to reduce the dimensionality to 2 for 2D visualization
        # pca = PCA(n_components=2)
        # data_pca = pca.fit_transform(data)  # Shape: [6000, 2]
        #
        # # Reshape back to [30, 200, 2] for plotting
        # data_pca = data_pca.reshape(times.shape[0], num_trials, 2)
        #
        # # Plot the trajectories in the first two principal components
        # plt.figure(figsize=(10, 8))
        #
        # for trial in range(data_pca.shape[1]):  # Iterate over trials
        #     plt.plot(
        #         data_pca[:, trial, 0],  # PC1
        #         data_pca[:, trial, 1],  # PC2
        #         alpha=0.7
        #     )
        #     plt.scatter(
        #         data_pca[0, trial, 0],  # PC1
        #         data_pca[0, trial, 1],  # PC2
        #         c='green'
        #     )
        #     plt.scatter(
        #         data_pca[-1, trial, 0],  # PC1
        #         data_pca[-1, trial, 1],  # PC2
        #         c='red'
        #     )
        # # Set labels and title
        # plt.xlabel("PC1")
        # plt.ylabel("PC2")
        # plt.title("Trajectories in PCA-reduced space (2D)")
        #
        # plt.savefig(path / 'trajectories.png', dpi=300)
        # plt.savefig(path / 'trajectories.pdf')


if __name__ == '__main__':
    decorated_main()

