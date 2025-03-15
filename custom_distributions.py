import torch
import torch.distributions as D
from torch.distributions import Distribution
import math

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


if __name__ == '__main__':
    gap_points = [-2.0, 0.0, 2.0]
    epsilon = 0.5
    dist = MultiGapNormal(gap_points, epsilon, loc=0.0, scale=1.0)

    # Sample from the modified distribution:
    samples = dist.sample((10000,))
    # print(samples)
    #
    # # Compute the log probability of a value:
    # print(dist.log_prob(torch.tensor([1.0])))

    # import matplotlib.pyplot as plt
    # plt.hist(samples, bins=100)
    # plt.show()

    dist = torch.distributions.Normal(loc=0.0, scale=1.0)
    print(
        dist.sample(sample_shape=(10,)).shape
    )
    multivariate_dist = makeIIDMultiVariate(dist,dim=4)
    print(
        multivariate_dist.sample(sample_shape=(10,)).shape
    )

    combined_dist = concat([multivariate_dist, multivariate_dist]*2)
    sample = combined_dist.sample(sample_shape=(10,))
    print("Combined sample shape):", sample.shape)