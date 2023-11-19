import typing as T

from src.data.data_generator import ContinuousSampleGenerator

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
