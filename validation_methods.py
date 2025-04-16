from torchdiffeq import odeint
import torch
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from functools import partial
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
import numpy as np

def run_initial_conditions(dynamics_function, initial_conditions, T=10, dt=0.05):
    times = torch.arange(0, T + dt, dt)  # Create a time vector from 0 to T with step dt
    trajectories = odeint(lambda t,y : dynamics_function(y), initial_conditions, times)  # Run the dynamics using odeint
    return trajectories
    

class ClassifierBasedSeparatrixLocator(BaseEstimator):
    def __init__(self, 
                 num_models=1, 
                 dynamics_dim=1, 
                 model_class = partial(make_pipeline, StandardScaler(), SVC(gamma='auto', kernel='linear')),
                 num_clusters = None,
                 lr=1e-3, 
                 epochs=100, 
                 use_multiprocessing=True, 
                 verbose=False, 
                 device="cpu"):
        
        super().__init__()
        self.model_class = model_class
        self.num_clusters = num_clusters
        self.dynamics_dim = dynamics_dim
        self.lr = lr
        self.num_models = num_models
        self.epochs = epochs
        self.verbose = verbose
        self.device = device
        self.use_multiprocessing = use_multiprocessing
        self.init_models()
        self.model_specs = {
            'num_models': num_models,
            'dynamics_dim': dynamics_dim,
            'num_clusters': num_clusters,
            'learning_rate': lr,
            'epochs': epochs,
            'model_class': model_class
        }
        self.results = {
            'accuracies': [],
            'optimal_clusters': None,
            'silhouette_score': None,
            'num_clusters_provided': num_clusters is not None
        }

    def sample(self, distribution, batch_size):
        if hasattr(distribution, '__iter__'):
            samples = [dist.sample(sample_shape=(batch_size,)) for dist in distribution]
        else:
            samples = distribution.sample(sample_shape=(batch_size,))
        return samples

    def init_models(self):
        self.models = [self.model_class() for _ in range(self.num_models)]  # Initialize models

    def fit(self, func, distribution, log_dir=None, batch_size=100, **kwargs):
        # samples = distribution.sample(sample_shape=(batch_size,))  # Sample from the provided distribution
        samples = self.sample(distribution, batch_size)  # Sample from the provided distribution
        trajectories = run_initial_conditions(func, samples).detach().cpu()  # Get trajectories using the sampled initial conditions

        # Extract the last time points of trajectories
        last_time_points = trajectories[-1].numpy()  # Get the last time point for each trajectory

        # Perform KMeans clustering with cross-validation to find the optimal K if num_clusters is None
        if self.num_clusters is None:
            self.find_optimal_num_clusters(last_time_points)
        
        self.kmeans = KMeans(n_clusters=self.num_clusters)
        self.kmeans.fit(last_time_points)
        labels = self.kmeans.labels_
        
        # Fit the SVM model to the samples and labels
        for model in self.models:
            model.fit(samples.cpu().numpy(), labels)


    def find_optimal_num_clusters(self, data, max_k=10):
        best_k = 2
        best_score = -1

        for k in range(2, max_k + 1):
            kf = KFold(n_splits=5)
            scores = []

            for train_index, test_index in kf.split(data):
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(data[train_index])
                score = silhouette_score(data[test_index], kmeans.predict(data[test_index]))
                scores.append(score)

            avg_score = sum(scores) / len(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_k = k

        self.num_clusters = best_k  # Update num_models to the best K found
        self.results['optimal_clusters'] = best_k
        self.results['silhouette_score'] = best_score
        if self.verbose:
            print(f"Optimal number of clusters (K) found: {best_k} with silhouette score: {best_score}")
    
    def predict(self, inputs, no_grad=True):
        pass

    def score(self, func, distribution, batch_size=100, **kwargs):
        # samples = distribution.sample(sample_shape=(batch_size,))  # Sample from the provided distribution
        samples = self.sample(distribution, batch_size)  # Sample from the provided distribution
        trajectories = run_initial_conditions(func, samples)  # Get trajectories using the sampled initial conditions

        # Extract the last time points of trajectories
        last_time_points = trajectories[-1]  # Get the last time point for each trajectory
        labels = self.kmeans.predict(last_time_points.cpu().numpy())

        accuracies = []  # List to collect accuracies from all models
        # Test the classifier on the samples
        for model in self.models:
            predictions = model.predict(samples.cpu().numpy())
            accuracy = float(np.mean(predictions == labels))
            self.results['accuracies'].append(accuracy)  # Collect accuracy for each model
            if self.verbose:
                print(f"Model accuracy: {accuracy}")

        return self.results['accuracies']  # Return the list of accuracies

    def save_models(self, savedir):
        pass

    def load_models(self, savedir):
        pass

    def filter_models(self, threshold):
        pass

    def find_separatrix(self, distribution, dist_needs_dim=True, return_indices=False, return_mask=False, **kwargs):
        pass



if __name__ == "__main__":
    from dynamical_functions import bistable_ND
    from torch.distributions import MultivariateNormal

    dim = 2
    training_samples = 100

    test_func = lambda z: bistable_ND(z, dim=dim, pos=0)

    mean = torch.zeros(dim)  # Mean vector
    covariance = torch.eye(dim) * 2
    mvn = MultivariateNormal(mean, covariance)

    # Example usage
    z = torch.tensor([0.5])  # Example input tensor for bistable_ND
    result = test_func(z)


    samples = mvn.sample(sample_shape=(10,))
    print(samples.shape)
    
    SL = ClassifierBasedSeparatrixLocator(num_models=10,num_clusters=2, verbose=True)

    # classifier_instance.models[0]
    SL.fit(test_func, mvn, batch_size=training_samples)
    classification_acccuracy = SL.score(test_func, mvn, batch_size=1000)
    print(classification_acccuracy)

    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    # Sample initial conditions
    num_samples = 10
    initial_conditions = mvn.sample(sample_shape=(num_samples,))  # Sample from the multivariate normal distribution

    trajectories = run_initial_conditions(test_func, initial_conditions)

    # Flatten the trajectories for clustering
    flattened_trajectories = trajectories[-1]

    # Perform KMeans clustering

    labels = SL.kmeans.predict(flattened_trajectories.cpu().numpy())

    # Plot trajectories colored by their kmeans label
    plt.figure(figsize=(10, 6))
    for i in range(num_samples):
        plt.plot(trajectories[:, i, 0].cpu().numpy(), color='C' + str(labels[i]), alpha=0.5)  # Color by label
    plt.title('Trajectories Colored by KMeans Label')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


    if dim==2:
        num_samples = 5000
        samples = mvn.sample(sample_shape=(num_samples,))  # Generate 1000 samples from the multivariate normal distribution
        labels = SL.models[0].predict(samples.cpu().numpy())  # Predict labels using the classifier

        # Plot the generated samples colored by their kmeans label
        plt.figure(figsize=(10, 6))
        for i in range(num_samples):
            plt.scatter(samples[i, 0].cpu().numpy(), samples[i, 1].cpu().numpy(), color='C' + str(labels[i]), alpha=0.5)  # Color by label
        plt.title('Generated Samples Colored by SVM Label')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()