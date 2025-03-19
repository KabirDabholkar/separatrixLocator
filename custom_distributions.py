import torch
import torch.distributions as D
from torch.distributions import Distribution
import math
from sklearn.decomposition import PCA
from rnn import reshape_hidden
from typing import Union

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




class ConcatIIDDistribution(Distribution):
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

    @property
    def batch_shape(self):
        return self.base_distribution.batch_shape

    @property
    def event_shape(self):
        return self.projection_layer.weight.shape[0],

    def sample(self, sample_shape=torch.Size()):
        base_samples = self.base_distribution.sample(sample_shape)
        projected_samples = self.projection_layer(base_samples)
        return projected_samples

# Define the union type for extended distributions
from typing import Any

class ExtendedDistributions:
    @staticmethod
    def is_instance(obj: Any) -> bool:
        return isinstance(obj, (torch.distributions.Distribution, ProjectedDistribution))



def initialize_linear_layer(input_dim, output_dim, weights, biases):
    linear_layer = torch.nn.Linear(input_dim, output_dim)
    linear_layer.weight.data = torch.tensor(weights, dtype=torch.float32)
    # print('weights shape',torch.tensor(weights, dtype=torch.float32).shape)
    linear_layer.bias.data = torch.tensor(biases, dtype=torch.float32)
    return linear_layer

def singlePC_distribution_from_hidden(hidden, component_id=0,squeeze_first_two_dims=True):
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
            torch.distributions.Normal(loc=0.0, scale=scale),
            dim=1),
        layer
    )
    return dist



# Example usage:
if __name__ == '__main__':
    dist1 = torch.distributions.Normal(loc=0.0, scale=1.0)
    dist2 = torch.distributions.Normal(loc=5.0, scale=1.0)
    mixture_dist = MixtureDistribution([dist1, dist2], weights=[0.3, 0.7])

    # Sample from the mixture distribution:
    samples = mixture_dist.sample((1000,))
    # print(samples)

    # Compute the log probability of a value:
    # print(mixture_dist.log_prob(torch.tensor([1.0])))

    # import matplotlib.pyplot as plt
    # plt.hist(samples.numpy().flatten(), bins=100)
    # plt.show()



    # Define specific weights and biases
    specific_weights = [[0.1] * 1] * 10  # Replace with your specific weights
    specific_biases = [0.1] * 10         # Replace with your specific biases

    projected_dist = ProjectedDistribution(
        makeIIDMultiVariate(torch.distributions.Normal(loc=0.0, scale=1.0), 1),
        initialize_linear_layer(1, 10, specific_weights, specific_biases)
    )
    samples = projected_dist.sample((1000,))
    print(samples.shape)

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


    print(
        singlePC_distribution_from_hidden(torch.randn((3,4,5)))
    )

    print(
        isinstance( torch.distributions.Normal(loc=0.0, scale=1.0) , ExtendedDistributions )
    )
    print(
        isinstance(torch.distributions.Normal(loc=0.0, scale=1.0), torch.distributions.Distribution )
    )