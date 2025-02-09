from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
from functools import partial

from learn_koopman_eig import train_with_logger, eval_loss


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

    def fit(self, func, distribution, **kwargs):
        train_single_model_ = partial(train_with_logger,F=func,dist=distribution, dynamics_dim=self.dynamics_dim,**kwargs)
        self.models = [self.model_class(self.dynamics_dim) for _ in range(self.num_models)]
        print(self.models)
        if self.use_multiprocessing:
            mp.set_start_method('spawn', force=True)
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]

            with mp.Pool(processes=len(devices)) as pool:
                results = [
                    pool.apply_async(
                        train_single_model_,
                        args=(self.models[i], devices[i % len(devices)], self.lr, self.epochs)
                    ) for i in range(self.num_models)
                ]

                for i, result in enumerate(results):
                    self.models[i].load_state_dict(result.get())
        else:
            for model in self.models:
                train_single_model_(model,verbose=self.verbose)
                # model.load_state_dict(trained_state_dict)

        return self

    def score(self, func, distribution, **kwargs):
        scores = []
        for model in self.models:
            # y_pred = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
            score = eval_loss(model, func, distribution, **kwargs)
            scores.append(score)
        self.scores = scores
        return scores

    def filter_models(self, threshold):
        assert (self.scores is not None)
        scores = self.scores
        self.models = [m for m, s in zip(self.models, scores) if s < threshold]
        return self

    def find_separatrix(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        for model in self.models:
            optimizer = optim.SGD([X_tensor], lr=0.01)
            for _ in range(100):
                optimizer.zero_grad()
                loss = model(X_tensor).abs().mean()
                loss.backward()
                optimizer.step()
        return X_tensor.detach().numpy()

if __name__ == '__main__':
    # model_class = KoopmanEigenfunctionModel
    from learn_koopman_eig import create_phi_network as model_class
    model_class = partial(model_class, num_layers=7, output_dim=10)

    SL = SeparatrixLocator(
        num_models = 2,
        dynamics_dim = 2,
        use_multiprocessing = False,
        verbose = True,
        model_class=model_class
    )
    from torch.distributions import Normal, Uniform
    SL.fit(
        func = lambda x: x-x**3,
        distribution= Normal(0, 1),
        dist_requires_dim = True,
        batch_size=2000
    )