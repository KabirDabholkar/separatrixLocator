from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
from functools import partial
from pathlib import Path
from compose import compose
import os
from learn_koopman_eig import train_with_logger, eval_loss, runGD


class KoopmanEigenfunctionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_single_model(model, X, y, device, lr, epochs):
    """Trains a single Koopman model on a given device."""
    model.to(device)
    X, y = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    return model.cpu().state_dict()


class SeparatrixLocator(BaseEstimator):
    def __init__(self, num_models=10, dynamics_dim=1, model_class=KoopmanEigenfunctionModel, lr=1e-3, epochs=100, use_multiprocessing=True, verbose=False, device="cpu"):
        super().__init__()
        # self.func = func
        # self.distribution = distribution
        self.num_models = num_models
        self.lr = lr
        self.epochs = epochs
        self.use_multiprocessing = use_multiprocessing
        self.device = device
        self.verbose = verbose
        self.dynamics_dim = dynamics_dim
        self.model_class = model_class
        self.models = []
        self.scores = None

    def init_models(self):
        self.models = [self.model_class(self.dynamics_dim) for _ in range(self.num_models)]

    def fit(self, func, distribution, **kwargs):
        train_single_model_ = partial(train_with_logger,F=func,dist=distribution, dynamics_dim=self.dynamics_dim,**kwargs)

        if len(self.models)==0:
            self.init_models()

        print(self.models)
        if self.use_multiprocessing:
            mp.set_start_method('spawn', force=True)
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]

            with mp.Pool(processes=len(devices)) as pool:
                results = [
                    pool.apply_async(
                        train_single_model_,
                        args = (self.models[i]),
                        kwds = dict(verbose=self.verbose,device=self.device),
                    ) for i in range(self.num_models)
                ]

                # for i, result in enumerate(results):
                #     self.models[i].load_state_dict(result.get())
        else:
            for model in self.models:
                train_single_model_(model,verbose=self.verbose,device=self.device)
                # model.load_state_dict(trained_state_dict)

        return self

    def score(self, func, distribution, **kwargs):
        scores = []
        for model in self.models:
            score = eval_loss(model, func, distribution, dynamics_dim=self.dynamics_dim, **kwargs)
            scores.append(score)
        self.scores = scores
        return scores

    def save_models(self,savedir):
        os.makedirs(Path(savedir)/"models", exist_ok=True)
        for i,model in enumerate(self.models):
            torch.save(model.state_dict(), Path(savedir) / "models" / f"{model.__class__.__name__}_{i}.pt")

    def load_models(self,savedir):
        for i,model in enumerate(self.models):
            state_dict = torch.load(Path(savedir) / "models" / f"{model.__class__.__name__}_{i}.pt",weights_only=True)
            self.models[i].load_state_dict(state_dict)

    def filter_models(self, threshold):
        assert (self.scores is not None)
        scores = self.scores
        self.models = [m for m, s in zip(self.models, scores) if s < threshold]
        return self

    def find_separatrix(self, distribution, dist_needs_dim=True, **kwargs):
        all_trajectories = []
        all_below_threshold_points = []
        for model in self.models:
            model_to_GD_on = compose(
                torch.log,
                lambda x: x + 1,
                torch.exp,
                partial(torch.sum, dim=-1, keepdims=True),
                torch.log,
                torch.abs,
                model
            )
            samples_for_normalisation = 1000

            needs_dim = dist_needs_dim

            samples = distribution.sample(sample_shape=[samples_for_normalisation] + ([self.dynamics_dim] if needs_dim else []))
            norm_val = float(torch.mean(torch.sum(model_to_GD_on(samples) ** 2, axis=-1)).sqrt().detach().numpy())

            model_to_GD_on = compose(
                lambda x: x / norm_val,
                model_to_GD_on
            )

            trajectories, below_threshold_points = runGD(
                model_to_GD_on,
                distribution,
                input_dim = self.dynamics_dim,
                dist_needs_dim = dist_needs_dim,
                **kwargs
            )
            all_trajectories.append(trajectories)
            all_below_threshold_points.append(below_threshold_points)
        return all_trajectories, all_below_threshold_points

if __name__ == '__main__':
    # model_class = KoopmanEigenfunctionModel
    from learn_koopman_eig import create_phi_network as model_class
    model_class = partial(model_class, num_layers=7, output_dim=10)

    SL = SeparatrixLocator(
        num_models = 2,
        dynamics_dim = 2,
        use_multiprocessing = False,
        verbose = True,
        model_class = model_class
    )
    from torch.distributions import Normal, Uniform
    SL.fit(
        func = lambda x: x-x**3,
        distribution = Normal(0, 1),
        dist_requires_dim = True,
        batch_size = 2000
    )