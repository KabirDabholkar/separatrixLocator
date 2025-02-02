import torch
import torch.distributions as D
import math


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


