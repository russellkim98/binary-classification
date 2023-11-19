import typing as T

import scipy.stats as stats
from data_generator import DataGenerator


class ContinuousSampleGenerator(DataGenerator):
    """implementation of the ContinuousSampleGenerator abstract class."""

    def __init__(self, distribution_name, params: T.Tuple):
        """Initialize the discrete sample generator with the specified distribution and parameters."""
        super().__init__(distribution_name, params)
        self.distribution = getattr(stats, distribution_name)
        self.params = params

    def sample(self, size):
        """Generate a sample of the specified size from the discrete distribution."""
        return self.distribution.rvs(size, *self.params)


# Implementations of ContinuousSampleGenerator for the 5 most commonly used continuous distributions


class UniformSampleGenerator(ContinuousSampleGenerator):
    def __init__(self, loc, scale):
        super().__init__(distribution_name="uniform", params=(loc, scale))


class NormalSampleGenerator(ContinuousSampleGenerator):
    def __init__(self, loc, scale):
        super().__init__(distribution_name="norm", params=(loc, scale))


class GammaSampleGenerator(ContinuousSampleGenerator):
    def __init__(self, alpha, loc, scale):
        super().__init__(distribution_name="gamma", params=(alpha, loc, scale))


class ExponentialSampleGenerator(ContinuousSampleGenerator):
    def __init__(self, loc, scale):
        super().__init__(distribution_name="expon", params=(loc, scale))


class BetaSampleGenerator(ContinuousSampleGenerator):
    def __init__(self, alpha, beta, loc, scale):
        super().__init__(distribution_name="beta", params=(alpha, beta, loc, scale))
