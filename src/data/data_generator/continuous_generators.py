class ContinuousSampleGenerator(abc.ABC):
    """Abstract class for generating samples from continuous probability distributions."""

    @abc.abstractmethod
    def __init__(self, distribution_name, *params):
        """Initialize the continuous sample generator with the specified distribution and parameters."""
        pass

    @abc.abstractmethod
    def sample(self, size):
        """Generate a sample of the specified size from the continuous distribution."""
        pass


class ConcreteContinuousSampleGenerator(ContinuousSampleGenerator):
    """Concrete implementation of the ContinuousSampleGenerator abstract class."""

    def __init__(self, distribution_name, *params):
        """Initialize the continuous sample generator with the specified distribution and parameters."""
        self.distribution = getattr(stats, distribution_name)
        self.params = params

    def sample(self, size):
        """Generate a sample of the specified size from the continuous distribution."""
        return self.distribution.rvs(size, *self.params)


# Concrete implementations of ContinuousSampleGenerator for the 5 most commonly used continuous distributions


class UniformSampleGenerator(ConcreteContinuousSampleGenerator):
    def __init__(self, loc, scale):
        super().__init__(distribution_name="uniform", params=(loc, scale))


class NormalSampleGenerator(ConcreteContinuousSampleGenerator):
    def __init__(self, loc, scale):
        super().__init__(distribution_name="norm", params=(loc, scale))


class GammaSampleGenerator(ConcreteContinuousSampleGenerator):
    def __init__(self, alpha, loc, scale):
        super().__init__(distribution_name="gamma", params=(alpha, loc, scale))


class ExponentialSampleGenerator(ConcreteContinuousSampleGenerator):
    def __init__(self, loc, scale):
        super().__init__(distribution_name="expon", params=(loc, scale))


class BetaSampleGenerator(ConcreteContinuousSampleGenerator):
    def __init__(self, alpha, beta, loc, scale):
        super().__init__(distribution_name="beta", params=(alpha, beta, loc, scale))
