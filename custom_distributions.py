import torch
import torch.distributions as D
from torch.distributions import Distribution
import math
from sklearn.decomposition import PCA
from rnn import reshape_hidden
from typing import Union, Callable, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

def sample(dist,shape):
    return dist.sample(sample_shape=shape)

class CubicHermiteSampler:
    """
    A distribution that samples points along a cubic Hermite curve with added noise.

    Attributes:
        x: Starting point of the curve
        y: Ending point of the curve
        scale: Scale of noise to add to sampled points
        alpha_dist: torch.distribution to sample alpha values from. Defaults to uniform.
    """

    def __init__(self, x, y=None, scale=0.1, alpha_dist=None):
        super().__init__()
        if y is None:
            if x.shape[0] != 2:
                raise ValueError("If y is not provided, x must be a (2,dim) tensor")
            self.x = x[0]
            self.y = x[1]
        else:
            self.x = x
            self.y = y
        self.dim = self.x.shape[-1]
        self.scale = scale
        self.alpha_dist = alpha_dist if alpha_dist is not None else D.Uniform(0, 1)
        self.vector_noise = torch.distributions.MultivariateNormal(torch.zeros(self.dim),
                                                                   torch.eye(self.dim) * self.scale ** 2)

    def sample(self, sample_shape=torch.Size()):
        """
        Sample points along the cubic Hermite curve with added noise.

        Args:
            sample_shape: Shape of the sample to generate

        Returns:
            samples: Tensor of sampled points with shape sample_shape + (dim,)
        """
        dim = self.x.shape[-1]

        # Generate alpha values for interpolation
        alpha = self.alpha_dist.sample(sample_shape)

        # Generate noisy tangent vectors around the x-y vector
        diff = -self.x + self.y
        norm = (diff ** 2).mean().sqrt()
        xy_vector = (diff).expand(*sample_shape, dim)
        # print('alpha term shape', (2 * alpha ** 3 - 3 * alpha ** 2 + 1).unsqueeze(-1).shape)
        m_x = xy_vector
        m_y = xy_vector

        m_x = m_x + self.vector_noise.sample(sample_shape=(*sample_shape,))/norm
        m_y = m_y + self.vector_noise.sample(sample_shape=(*sample_shape,))/norm

        # Expand x and y points to match sample shape
        x_expanded = self.x.expand(*sample_shape, dim)
        y_expanded = self.y.expand(*sample_shape, dim)

        # Compute points on the curve using cubic Hermite interpolation
        points = (2 * alpha ** 3 - 3 * alpha ** 2 + 1).unsqueeze(-1) * x_expanded + \
                 (-2 * alpha ** 3 + 3 * alpha ** 2).unsqueeze(-1) * y_expanded + \
                 (alpha ** 3 - 2 * alpha ** 2 + alpha).unsqueeze(-1) * m_x + \
                 (alpha ** 3 - alpha ** 2).unsqueeze(-1) * m_y

        return points


def create_hermite_samplers_from_three_points(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                                              scale1: float, scale2: float,
                                              alpha_dist1: D.Distribution = None,
                                              alpha_dist2: D.Distribution = None) -> tuple[
    CubicHermiteSampler, CubicHermiteSampler]:
    """
    Create two CubicHermiteSampler objects from three points.

    Args:
        a: First point
        b: Second point (middle point)
        c: Third point
        scale1: Scale parameter for first sampler (a to b)
        scale2: Scale parameter for second sampler (b to c)
        alpha_dist1: Alpha distribution for first sampler
        alpha_dist2: Alpha distribution for second sampler

    Returns:
        Tuple of two CubicHermiteSampler objects
    """
    sampler1 = CubicHermiteSampler(a, b, scale=scale1, alpha_dist=alpha_dist1)
    sampler2 = CubicHermiteSampler(b, c, scale=scale2, alpha_dist=alpha_dist2)
    return [sampler1, sampler2]


def create_hermite_samplers_from_three_points_stacked(ac: torch.Tensor, b: torch.Tensor,
                                              scale1: float, scale2: float,
                                              alpha_dist1: D.Distribution = None,
                                              alpha_dist2: D.Distribution = None) -> tuple[
    CubicHermiteSampler, CubicHermiteSampler]:
    """
    Create two CubicHermiteSampler objects from three points, where the first and third points are stacked in a single tensor.

    Args:
        ac: Stacked tensor containing first and third points [a, c]
        b: Second point (middle point)
        scale1: Scale parameter for first sampler (a to b)
        scale2: Scale parameter for second sampler (b to c)
        alpha_dist1: Alpha distribution for first sampler
        alpha_dist2: Alpha distribution for second sampler

    Returns:
        Tuple of two CubicHermiteSampler objects
    """
    return create_hermite_samplers_from_three_points(ac[0], b, ac[1], scale1, scale2, alpha_dist1, alpha_dist2)

def isotropic_gaussian(mean, scale=1.0):
    """
    Create a multivariate isotropic Gaussian distribution.

    Args:
        mean: Mean vector of the distribution
        scale: Scale factor for the covariance matrix (default: 1.0)

    Returns:
        dist: torch.distributions.MultivariateNormal with isotropic covariance
    """
    dim = len(mean)
    cov = torch.eye(dim) * scale
    return D.MultivariateNormal(mean, cov)

def beta_from_mean_var(mean: float, var: float) -> D.Beta:
    """
    Create a Beta distribution with specified mean and variance.
    
    Args:
        mean: Mean of the Beta distribution (between 0 and 1)
        var: Variance of the Beta distribution (must be less than mean*(1-mean))
        
    Returns:
        dist: torch.distributions.Beta with the specified mean and variance
        
    Raises:
        ValueError: If mean is not between 0 and 1, or if variance is invalid
    """
    if not 0 < mean < 1:
        raise ValueError("Mean must be between 0 and 1")
    if not 0 < var < mean * (1 - mean):
        raise ValueError("Variance must be between 0 and mean*(1-mean)")
        
    # Solve for alpha and beta parameters
    # mean = alpha/(alpha + beta)
    # var = (alpha*beta)/((alpha + beta)^2 * (alpha + beta + 1))
    
    alpha = mean * (mean * (1 - mean) / var - 1)
    beta = (1 - mean) * (mean * (1 - mean) / var - 1)
    
    return D.Beta(alpha, beta)


def iid_beta(mean, scale) -> D.Beta:
    """
    Create a Beta distribution with specified mean and variance.
    
    Args:
        mean: Mean vector of the distribution
        scale: Std of the Beta distribution (must be less than sqrt(mean*(1-mean)))
    """
    return ConcatIIDDistribution([beta_from_mean_var(mean[i], scale**2) for i in range(len(mean))])

def list_of_iid_betas(mean, scales):
    return [iid_beta(mean, scale) for scale in scales]

def isotropic_gaussians(mean, scales):
    """
    Create a list of multivariate isotropic Gaussian distributions with the same mean but different scales.

    Args:
        mean: Mean vector of the distributions
        scales: List of scale factors for the covariance matrices

    Returns:
        dists: List of torch.distributions.MultivariateNormal with isotropic covariance
    """
    return [isotropic_gaussian(mean, scale) for scale in scales]

def random_gaussians(dim: int, num_distributions: int, mean_range: tuple = (-1.0, 1.0), scale_range: tuple = (0.1, 2.0)) -> list:
    """
    Create a list of multivariate isotropic Gaussian distributions with random means and scales.

    Args:
        dim: Dimension of each Gaussian distribution
        num_distributions: Number of Gaussian distributions to create
        mean_range: Tuple (min, max) for sampling means uniformly
        scale_range: Tuple (min, max) for sampling scales uniformly

    Returns:
        dists: List of torch.distributions.MultivariateNormal with random means and scales
    """
    dists = []
    for _ in range(num_distributions):
        # Sample random mean vector
        mean = torch.rand(dim) * (mean_range[1] - mean_range[0]) + mean_range[0]
        # Sample random scale
        scale = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
        dists.append(isotropic_gaussian(mean, scale))
    return dists



def makeIIDMultiVariate(dist, dim):
    # Expand the distribution so that its batch_shape becomes (dim,)
    # Then wrap it with Independent to treat these as event dimensions.
    return torch.distributions.Independent(dist.expand([dim]), 1)


class MultiGapNormal(D.Distribution):
    """
    A modified Normal distribution that is identical to N(loc, scale)
    except that the probability density is zero for x in any of the gaps.

    Each gap is specified by its center point and a half-width epsilon,
    so that the gap around center c is (c - epsilon, c + epsilon).

    Note: The gap intervals should be disjoint.
    """
    arg_constraints = {}
    support = D.constraints.real
    has_rsample = False  # Using standard (non-reparameterized) sampling

    def __init__(self, gap_points, epsilon, loc=0.0, scale=1.0, validate_args=None):
        """
        Args:
            gap_points (list or tensor): A list or 1D tensor of centers for the gaps.
            epsilon (float): Half-width of each gap.
            loc (float or Tensor): Mean of the underlying Normal.
            scale (float or Tensor): Standard deviation of the underlying Normal.
        """
        self.epsilon = epsilon
        # Create gap intervals: list of (low, high) pairs.
        # For a gap centered at c, the interval is (c - epsilon, c + epsilon)
        self.gap_intervals = [(c - epsilon, c + epsilon) for c in gap_points]

        # Create the underlying normal distribution.
        self.normal = D.Normal(loc, scale)

        # Determine device and dtype from the normal's parameters.
        # (If loc is a float, we'll default to CPU and torch.float.)
        if isinstance(self.normal.loc, torch.Tensor):
            device = self.normal.loc.device
            dtype = self.normal.loc.dtype
        else:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()

        # Compute the total probability mass removed by the gaps.
        removed_mass = 0.0
        for (low, high) in self.gap_intervals:
            # Convert low and high to tensors
            low_tensor = torch.tensor(low, dtype=dtype, device=device)
            high_tensor = torch.tensor(high, dtype=dtype, device=device)
            removed_mass += self.normal.cdf(high_tensor) - self.normal.cdf(low_tensor)

        # Renormalization constant: probability mass remaining.
        self.Z = 1.0 - removed_mass
        if self.Z <= 0:
            raise ValueError("The total probability mass outside the gaps must be positive.")

        super().__init__(validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the modified distribution via rejection sampling.
        """
        shape = self._extended_shape(sample_shape)
        samples = self.normal.sample(shape)
        # Build a mask that is True for values that fall in any gap.
        mask = torch.zeros_like(samples, dtype=torch.bool)
        for (low, high) in self.gap_intervals:
            mask |= ((samples > low) & (samples < high))

        # Resample any values that fell inside a gap.
        while mask.any():
            new_samples = self.normal.sample(shape)
            samples[mask] = new_samples[mask]
            mask = torch.zeros_like(samples, dtype=torch.bool)
            for (low, high) in self.gap_intervals:
                mask |= ((samples > low) & (samples < high))
        return samples

    def log_prob(self, value):
        """
        Compute the log-probability of a given value.
        Returns -infinity for values within any gap.
        """
        mask = torch.zeros_like(value, dtype=torch.bool)
        for (low, high) in self.gap_intervals:
            mask |= ((value > low) & (value < high))

        lp = torch.full_like(value, float('-inf'))
        valid_mask = ~mask
        if valid_mask.any():
            lp[valid_mask] = self.normal.log_prob(value[valid_mask]) - math.log(self.Z)
        return lp

    def cdf(self, value):
        """
        (Optional) Compute the cumulative distribution function.
        For values that lie in a gap, the CDF is continuous from the left.
        Here, we approximate it by subtracting the mass of any gaps below the value.
        """
        cdf_val = self.normal.cdf(value)
        for (low, high) in self.gap_intervals:
            gap_mass = self.normal.cdf(high) - self.normal.cdf(low)
            cdf_val = torch.where(value >= high, cdf_val - gap_mass, cdf_val)
            cdf_val = torch.where((value > low) & (value < high), self.normal.cdf(low), cdf_val)
        return cdf_val / self.Z




class ConcatIIDDistribution: #(Distribution)
    """
    A distribution that concatenates a list of distributions.
    When sampling, it independently samples from each distribution and
    concatenates the results along the last dimension.
    The log probability is computed as the sum of the log probabilities
    from each individual distribution.
    """

    def __init__(self, dists):
        # Save the list of distributions
        self.dists = dists

        # Check that all distributions have the same batch shape.
        batch_shapes = [dist.batch_shape for dist in dists]
        if not all(bs == batch_shapes[0] for bs in batch_shapes):
            raise ValueError("All distributions must have the same batch shape.")
        self._batch_shape = batch_shapes[0]

        # Determine the event size for each distribution.
        # If event_shape is empty, we treat the distribution as scalar (i.e. size 1).
        self.event_sizes = []
        for dist in dists:
            if len(dist.event_shape) == 0:
                size = 1
            else:
                size = 1
                for d in dist.event_shape:
                    size *= d
            self.event_sizes.append(size)
        total_size = sum(self.event_sizes)
        # We define the new event shape to be a 1D vector whose length is the sum
        # of the individual event sizes.
        self._event_shape = (total_size,)

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape

    def sample(self, sample_shape=torch.Size()):
        # For each distribution, sample and then ensure that the event dimension
        # is a 1D vector (flatten if necessary).
        samples = []
        for dist in self.dists:
            s = dist.sample(sample_shape)
            # If the distribution is scalar, unsqueeze to get a last dim.
            if len(dist.event_shape) == 0:
                s = s.unsqueeze(-1)
            else:
                # If the event is multi-dimensional, flatten it into one dimension.
                if len(dist.event_shape) > 1:
                    s = s.reshape(s.shape[:-len(dist.event_shape)] + (-1,))
            samples.append(s)
        # Concatenate along the last dimension (the event dimension).
        return torch.cat(samples, dim=-1)

    def log_prob(self, value):
        # 'value' is assumed to have shape sample_shape + (total_event_size,)
        # We split 'value' along the last dimension according to each distribution's event size.
        splits = torch.split(value, self.event_sizes, dim=-1)
        log_probs = []
        for split, dist in zip(splits, self.dists):
            # For scalar distributions, remove the extra dimension.
            if len(dist.event_shape) == 0:
                split = split.squeeze(-1)
            # Otherwise, if the event was flattened, we assume that log_prob accepts the flat vector.
            log_probs.append(dist.log_prob(split))
        # Since we assume independence, the overall log probability is the sum.
        return sum(log_probs)


def concat(dists):
    """Convenience function to create a ConcatDistribution."""
    return ConcatIIDDistribution(dists)


class MixtureDistribution(torch.distributions.Distribution):
    def __init__(self, distributions, weights=None):
        """
        A mixture of multiple distributions.

        Args:
            distributions (list of torch.distributions.Distribution): The component distributions.
            weights (list of float, optional): The weights for each component distribution. If None, all components are equally weighted.
        """
        self.distributions = distributions
        if weights is None:
            self.weights = torch.ones(len(distributions)) / len(distributions)
        else:
            self.weights = torch.tensor(weights) / sum(weights)
        self.categorical = torch.distributions.Categorical(self.weights)

    @property
    def batch_shape(self):
        return self.distributions[0].batch_shape

    @property
    def event_shape(self):
        return self.distributions[0].event_shape

    def sample(self, sample_shape=torch.Size()):
        # Sample from the categorical distribution to choose which component to sample from
        mixture_indices = self.categorical.sample(sample_shape)
        samples = []
        for i, dist in enumerate(self.distributions):
            mask = (mixture_indices == i).float().unsqueeze(-1)
            samples.append(dist.sample(sample_shape) * mask)
        return sum(samples)

    def log_prob(self, value):
        log_probs = torch.stack([dist.log_prob(value) for dist in self.distributions], dim=-1)
        weighted_log_probs = log_probs + torch.log(self.weights)
        return torch.logsumexp(weighted_log_probs, dim=-1)


class ProjectedDistribution:
    def __init__(self, base_distribution, projection_layer):
        """
        A distribution that samples from a base distribution and projects the samples to a higher dimension using a linear layer.

        Args:
            base_distribution (torch.distributions.Distribution): The base distribution to sample from.
            projection_layer (torch.nn.Linear): The linear layer to project the samples to a higher dimension.
        """
        self.base_distribution = base_distribution
        self.projection_layer = projection_layer

    # @property
    # def batch_shape(self):
    #     return self.base_distribution.batch_shape
    #
    # @property
    # def event_shape(self):
    #     return self.projection_layer.weight.shape[0],

    def sample(self, sample_shape=torch.Size()):
        base_samples = self.base_distribution.sample(sample_shape)
        projected_samples = self.projection_layer(base_samples)
        return projected_samples



def initialize_linear_layer(input_dim, output_dim, weights, biases):
    linear_layer = torch.nn.Linear(input_dim, output_dim)
    linear_layer.weight.data = torch.tensor(weights, dtype=torch.float32)
    # print('weights shape',torch.tensor(weights, dtype=torch.float32).shape)
    linear_layer.bias.data = torch.tensor(biases, dtype=torch.float32)
    return linear_layer

def singlePC_distribution_from_hidden(hidden, component_id=0,squeeze_first_two_dims=True,multiply_scale=1):
    if squeeze_first_two_dims:
        hidden = reshape_hidden(hidden)
    P = PCA()
    P.fit(hidden.detach().cpu().numpy())
    weights = P.components_[component_id][:, None]
    biases = P.mean_
    print(weights,biases)
    # layer = torch.nn.Linear(1,hidden.shape[-1])
    layer = initialize_linear_layer(hidden.shape[-1], 1, weights, biases)
    # Adjust the scale of the base distribution to reflect the variance along the principal component
    scale = math.sqrt(P.explained_variance_[component_id])
    dist = ProjectedDistribution(
        makeIIDMultiVariate(
            torch.distributions.Normal(loc=0.0, scale=scale * multiply_scale),
            dim=1),
        layer
    )
    return dist

def get_stacked_one_hot(pos=0,length=1):
    vec = torch.nn.functional.one_hot(torch.tensor(pos), num_classes=length)
    vec = vec.type(torch.float32)
    return torch.stack([vec,-vec])

class RejectionSamplerWithClassifier:
    """
    A class that combines a distribution with a classifier for rejection sampling.
    Samples from the base distribution are accepted or rejected based on the classifier's predictions.
    
    Attributes:
        base_distribution: A distribution-like object with a .sample() method
        classifier: A sklearn classifier with .predict() method
        target_class: The class label that should be accepted
        max_attempts: Maximum number of sampling attempts before raising an error
    """
    
    def __init__(self, 
                 base_distribution: Union[Distribution, Callable],
                 classifier: BaseEstimator,
                 target_class: int = 1,
                 max_attempts: int = 1000):
        """
        Initialize the rejection sampler.
        
        Args:
            base_distribution: Distribution to sample from
            classifier: Classifier to use for rejection
            target_class: Class label to accept
            max_attempts: Maximum sampling attempts before error
        """
        self.base_distribution = base_distribution
        self.classifier = classifier
        self.target_class = target_class
        self.max_attempts = max_attempts
        
    def sample(self, sample_shape: Union[torch.Size, tuple] = torch.Size()) -> torch.Tensor:
        """
        Sample from the distribution, rejecting points that don't match the target class.
        
        Args:
            sample_shape: Shape of the sample to generate
            
        Returns:
            samples: Tensor of accepted samples
            
        Raises:
            RuntimeError: If max_attempts is reached without getting enough samples
        """
        if isinstance(sample_shape, tuple):
            sample_shape = torch.Size(sample_shape)
            
        # Calculate total number of samples needed
        total_samples = 1
        for dim in sample_shape:
            total_samples *= dim
            
        samples = []
        attempts = 0
        
        while len(samples) < total_samples and attempts < self.max_attempts:
            # Sample from base distribution
            new_samples = self.base_distribution.sample(sample_shape)
            
            # Convert to numpy for classifier
            if isinstance(new_samples, torch.Tensor):
                new_samples_np = new_samples.detach().cpu().numpy()
            else:
                new_samples_np = np.array(new_samples)
                
            # Reshape if needed (classifier expects 2D array)
            if len(new_samples_np.shape) == 1:
                new_samples_np = new_samples_np.reshape(-1, 1)
                
            # Get predictions
            predictions = self.classifier.predict(new_samples_np)
            
            # Keep samples that match target class
            accepted_mask = predictions == self.target_class
            accepted_samples = new_samples[accepted_mask]
            
            samples.extend(accepted_samples)
            attempts += 1
            
        if len(samples) < total_samples:
            raise RuntimeError(f"Failed to generate enough samples after {attempts} attempts")
            
        # Convert to tensor and reshape
        samples = torch.stack(samples[:total_samples])
        return samples.reshape(sample_shape + samples.shape[-1:])
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of a value under the base distribution.
        Note: This does not account for the rejection sampling.
        
        Args:
            value: Value to compute log probability for
            
        Returns:
            log_prob: Log probability of the value
        """
        if hasattr(self.base_distribution, 'log_prob'):
            return self.base_distribution.log_prob(value)
        else:
            raise NotImplementedError("Base distribution does not support log_prob")
            
    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute the CDF of a value under the base distribution.
        Note: This does not account for the rejection sampling.
        
        Args:
            value: Value to compute CDF for
            
        Returns:
            cdf: CDF of the value
        """
        if hasattr(self.base_distribution, 'cdf'):
            return self.base_distribution.cdf(value)
        else:
            raise NotImplementedError("Base distribution does not support cdf")

# Example usage:
if __name__ == '__main__':

    # Demo locally_isotropic_beta with histograms
    import matplotlib.pyplot as plt
    #
    # # Define mean vector and scale
    # mean = torch.tensor([0.1, 0.1])
    # scale = np.sqrt(0.089)
    # dist = iid_beta(mean, scale)
    # samples = dist.sample((5000,))
    #
    # # Create 2D histogram plot
    # plt.figure(figsize=(8, 8))
    # plt.hist2d(samples[:, 0].numpy(), samples[:, 1].numpy(), bins=30, density=True)
    # plt.colorbar(label='Density')
    #
    # # Add mean point
    # plt.plot(mean[0], mean[1], 'r*', markersize=15, label='Mean')
    #
    # plt.title(f'2D Histogram of Locally Isotropic Beta\nμ={mean.numpy()}, σ={scale}')
    # plt.xlabel(r'$x_1$')
    # plt.ylabel(r'$x_2$')
    # plt.legend()
    # plt.grid(True)
    # plt.show()




    # Demo beta_from_mean_var with histograms
    import matplotlib.pyplot as plt
    
    print("Demonstrating beta_from_mean_var distribution:")
    
    # Create beta distributions with different means and variances
    beta1 = beta_from_mean_var(mean=0.1, var=np.sqrt(0.01)**2)
    beta2 = beta_from_mean_var(mean=0.7, var=0.01**2)
    
    # Sample from distributions
    samples1 = beta1.sample((1000,))
    samples2 = beta2.sample((1000,))
    
    # Print distribution parameters and sample statistics
    print("\nBeta Distribution 1:")
    print(f"Target mean: 0.3, Empirical mean: {samples1.mean():.3f}")
    print(f"Target var: 0.05, Empirical var: {samples1.var():.3f}")
    print(f"Alpha: {beta1.concentration1:.3f}, Beta: {beta1.concentration0:.3f}")
    
    print("\nBeta Distribution 2:")
    print(f"Target mean: 0.7, Empirical mean: {samples2.mean():.3f}")
    print(f"Target var: 0.02, Empirical var: {samples2.var():.3f}")
    print(f"Alpha: {beta2.concentration1:.3f}, Beta: {beta2.concentration0:.3f}")
    
    # Plot histograms
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(samples1.numpy(), bins=30, density=True, alpha=0.7)
    plt.title(f'Beta(μ=0.3, σ=0.1)\nα={beta1.concentration1:.1f}, β={beta1.concentration0:.1f}')
    plt.xlabel('x')
    plt.ylabel('Density')
    
    plt.subplot(1, 2, 2)
    plt.hist(samples2.numpy(), bins=30, density=True, alpha=0.7)
    plt.title(f'Beta(μ=0.7, σ=0.5)\nα={beta2.concentration1:.1f}, β={beta2.concentration0:.1f}')
    plt.xlabel('x')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.show()
    

    # dist1 = torch.distributions.Normal(loc=0.0, scale=1.0)
    # dist2 = torch.distributions.Normal(loc=5.0, scale=1.0)
    # mixture_dist = MixtureDistribution([dist1, dist2], weights=[0.3, 0.7])

    # Sample from the mixture distribution:
    # samples = mixture_dist.sample((1000,))
    # print(samples)

    # Compute the log probability of a value:
    # print(mixture_dist.log_prob(torch.tensor([1.0])))

    # import matplotlib.pyplot as plt
    # plt.hist(samples.numpy().flatten(), bins=100)
    # plt.show()



    # Define specific weights and biases
    # specific_weights = [[0.1] * 1] * 10  # Replace with your specific weights
    # specific_biases = [0.1] * 10         # Replace with your specific biases
    #
    # projected_dist = ProjectedDistribution(
    #     makeIIDMultiVariate(torch.distributions.Normal(loc=0.0, scale=1.0), 1),
    #     initialize_linear_layer(1, 10, specific_weights, specific_biases)
    # )
    # samples = projected_dist.sample((1000,))
    # print(samples.shape)

    # import matplotlib.pyplot as plt
    # plt.hist(samples[:, 0].numpy().flatten(), bins=100)
    # plt.show()
    # gap_points = [-2.0, 0.0, 2.0]
    # epsilon = 0.5
    # # dist = MultiGapNormal(gap_points, epsilon, loc=0.0, scale=1.0)
    #
    # # Sample from the modified distribution:
    # samples = dist.sample((10000,))
    # # print(samples)
    # #
    # # # Compute the log probability of a value:
    # # print(dist.log_prob(torch.tensor([1.0])))
    #
    # # import matplotlib.pyplot as plt
    # # plt.hist(samples, bins=100)
    # # plt.show()
    #
    # dist = torch.distributions.Normal(loc=0.0, scale=1.0)
    # print(
    #     dist.sample(sample_shape=(10,)).shape
    # )
    # multivariate_dist = makeIIDMultiVariate(dist,dim=4)
    # print(
    #     multivariate_dist.sample(sample_shape=(10,)).shape
    # )
    #
    # combined_dist = concat([multivariate_dist, multivariate_dist]*2)
    # sample = combined_dist.sample(sample_shape=(10,))
    # print("Combined sample shape):", sample.shape)

    # mixture_dist = MixtureDistribution(
    #     [
    #         torch.distributions.Normal(loc=-1.0, scale=1.0),
    #         torch.distributions.Normal(loc=1.0, scale=1.0),
    #     ],
    #     weights=[0.3, 0.7]
    # )
    # sample = mixture_dist.sample(sample_shape=(10,))
    # print("Mixture sample shape):", sample.shape)
    # x = torch.linspace(-3,3)
    #
    # plt.plot(x,mixture_dist.log_prob(x))
    # plt.show()
    #
    #


    # print(
    #     singlePC_distribution_from_hidden(torch.randn((3,4,5)))
    # )
    #
    # print(
    #     isinstance( torch.distributions.Normal(loc=0.0, scale=1.0) , ExtendedDistributions )
    # )
    # print(
    #     isinstance(torch.distributions.Normal(loc=0.0, scale=1.0), torch.distributions.Distribution )
    # )

    # dist1,dist2 = create_hermite_samplers_from_three_points(
    #     torch.ones(2) * 0,
    #     torch.ones(2) * 1,
    #     torch.ones(2) * 2,
    #     scale1=2.0,
    #     scale2=2.0,
    #     alpha_dist1=torch.distributions.Beta(20, 1),
    #     alpha_dist2=torch.distributions.Beta(1, 20),
    # )

    # import matplotlib.pyplot as plt

    # # Sample points from both distributions
    # samples1 = dist1.sample(sample_shape=(1000,))
    # samples2 = dist2.sample(sample_shape=(1000,))

    # # Convert to numpy for plotting
    # samples1_np = samples1.detach().cpu().numpy()
    # samples2_np = samples2.detach().cpu().numpy()

    # # Create scatter plot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(samples1_np[:, 0], samples1_np[:, 1], alpha=0.5, label='Distribution 1')
    # plt.scatter(samples2_np[:, 0], samples2_np[:, 1], alpha=0.5, label='Distribution 2')
    
    # # Plot the control points
    # plt.scatter([0, 1, 2], [0, 1, 2], c='red', s=100, marker='*', label='Control Points')
    
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.title('Samples from Cubic Hermite Samplers')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Demo of RejectionSamplerWithClassifier
    # import matplotlib.pyplot as plt
    # from sklearn.svm import SVC
    # from sklearn.datasets import make_moons
    # import numpy as np
    #
    # # Create a 2D dataset (two moons)
    # X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    #
    # # Train an SVM classifier
    # classifier = SVC(kernel='rbf', probability=True)
    # classifier.fit(X, y)
    #
    # # Create a base distribution (2D Gaussian)
    # base_dist = D.MultivariateNormal(
    #     loc=torch.zeros(2),
    #     covariance_matrix=torch.eye(2) * 2.0
    # )
    #
    # # Create rejection sampler for class 1
    # sampler = RejectionSamplerWithClassifier(
    #     base_distribution=base_dist,
    #     classifier=classifier,
    #     target_class=1,
    #     max_attempts=10000
    # )
    #
    # # Sample points that are classified as class 1
    # samples = sampler.sample((1000,))
    # samples_np = samples.detach().cpu().numpy()
    #
    # # Plot the results
    # plt.figure(figsize=(10, 5))
    #
    # # Plot the original dataset
    # plt.subplot(121)
    # plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', alpha=0.5, label='Class 0')
    # plt.scatter(X[y==1, 0], X[y==1, 1], c='red', alpha=0.5, label='Class 1')
    # plt.title('Original Dataset')
    # plt.legend()
    #
    # # Plot the sampled points
    # plt.subplot(122)
    # plt.scatter(samples_np[:, 0], samples_np[:, 1], c='red', alpha=0.5, label='Sampled Class 1')
    # plt.title('Rejection Sampled Points')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # Print some statistics
    # print(f"Number of samples generated: {len(samples_np)}")
    # print(f"Mean of sampled points: {np.mean(samples_np, axis=0)}")
    # print(f"Covariance of sampled points:\n{np.cov(samples_np.T)}")
    #
    # # Test the log_prob method
    # test_point = torch.tensor([0.0, 0.0])
    # print(f"Log probability of test point: {sampler.log_prob(test_point)}")
    #
    #
