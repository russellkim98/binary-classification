import typing as T

from src.data.data_generator.generator import ContinuousSampleGenerator

# Implementations of ContinuousSampleGenerator for the 5 most commonly used continuous distributions


class UniformSampleGenerator(ContinuousSampleGenerator):
    """
    Generates samples from a uniform distribution.

    Args:
        loc (float): The lower bound of the distribution. This represents the minimum value that the random variable can take.
        scale (float): The width of the distribution. This represents the range of values that the random variable can take. In other words, the difference between the upper bound and the lower bound is equal to the scale.
    """

    def __init__(self, loc: float, scale: float):
        super().__init__(distribution_name="uniform", params=dict(loc=loc, scale=scale))


class NormalSampleGenerator(ContinuousSampleGenerator):
    """
    Generates samples from a normal distribution.

    Args:
        loc (float): The mean of the distribution. This represents the center of the distribution and the value that the random variable is most likely to take.
        scale (float): The standard deviation of the distribution. This represents the spread of the distribution and the distance from the mean at which 68% of the samples fall.
    """

    def __init__(self, loc: float, scale: float):
        super().__init__(distribution_name="norm", params=dict(loc=loc, scale=scale))


class GammaSampleGenerator(ContinuousSampleGenerator):
    """
    Generates samples from a gamma distribution.

    Args:
        alpha (float): The shape parameter of the distribution. This parameter controls the skewness and tail of the distribution. Higher values of alpha lead to a more skewed distribution with a longer tail.
        loc (float): The location parameter of the distribution. This parameter controls the minimum value that the random variable can take.
        scale (float): The scale parameter of the distribution. This parameter controls the width of the distribution. A larger scale parameter leads to a wider distribution.
    """

    def __init__(self, alpha: float, loc: float, scale: float):
        super().__init__(
            distribution_name="gamma", params=dict(alpha=alpha, loc=loc, scale=scale)
        )


class ExponentialSampleGenerator(ContinuousSampleGenerator):
    """
    Generates samples from an exponential distribution.

    Args:
        loc (float): The location parameter of the distribution. This parameter controls the minimum value that the random variable can take.
        scale (float): The rate parameter of the distribution. This parameter is inversely proportional to the mean of the distribution. A larger scale parameter leads to a smaller mean and a faster decay rate for the distribution.
    """

    def __init__(self, loc: float, scale: float):
        super().__init__(distribution_name="expon", params=dict(loc=loc, scale=scale))


class BetaSampleGenerator(ContinuousSampleGenerator):
    """
    Generates samples from a beta distribution.

    Args:
        alpha (float): The first shape parameter of the distribution. This parameter controls the shape of the left tail of the distribution. Higher values of alpha lead to a more skewed distribution with a longer left tail.
        beta (float): The second shape parameter of the distribution. This parameter controls the shape of the right tail of the distribution. Higher values of beta lead to a more skewed distribution with a longer right tail.
        loc (float): The location parameter of the distribution. This parameter controls the minimum value that the random variable can take.
        scale (float): The scale parameter of the distribution. This parameter controls the width of the distribution. A larger scale parameter leads to a wider distribution.
    """

    def __init__(self, alpha: float, beta: float, loc: float, scale: float):
        super().__init__(
            distribution_name="beta",
            params=dict(alpha=alpha, beta=beta, loc=loc, scale=scale),
        )
