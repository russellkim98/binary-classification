import typing as T

import scipy.stats as stats
from data_generator import DataGenerator


class DiscreteSampleGenerator(DataGenerator):
    """Concrete implementation of the DiscreteSampleGenerator abstract class."""

    def __init__(self, distribution_name, params: T.Tuple):
        """Initialize the discrete sample generator with the specified distribution and parameters."""
        super().__init__(distribution_name, params)
        self.distribution = getattr(stats, distribution_name)
        self.params = params

    def sample(self, size):
        """Generate a sample of the specified size from the discrete distribution."""
        return self.distribution.rvs(size, *self.params)


# Concrete implementations of DiscreteSampleGenerator for the 5 most commonly used discrete distributions


class BernoulliSampleGenerator(DiscreteSampleGenerator):
    def __init__(self, p):
        super().__init__(distribution_name="bernoulli", params=(p,))


class BinomialSampleGenerator(DiscreteSampleGenerator):
    def __init__(self, n, p):
        super().__init__(distribution_name="binom", params=(n, p))


class PoissonSampleGenerator(DiscreteSampleGenerator):
    def __init__(self, mu):
        super().__init__(distribution_name="poisson", params=(mu,))


class GeometricSampleGenerator(DiscreteSampleGenerator):
    def __init__(self, p):
        super().__init__(distribution_name="geom", params=(p,))


class NegativeBinomialSampleGenerator(DiscreteSampleGenerator):
    def __init__(self, r, p):
        super().__init__(distribution_name="nbinom", params=(r, p))
