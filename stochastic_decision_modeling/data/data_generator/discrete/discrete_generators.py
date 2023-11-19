from stochastic_decision_modeling.data.data_generator.generator import (
    DiscreteSampleGenerator,
)

# Concrete implementations of DiscreteSampleGenerator for the 5 most commonly used discrete distributions


class BernoulliSampleGenerator(DiscreteSampleGenerator):
    """
    Generates samples from the Bernoulli distribution.

    Parameters:
        p (float): The probability of success. This must be a scalar value between 0 and 1.
            Mathematically, p represents the probability of a success in a single Bernoulli trial.
            A Bernoulli trial is a random experiment with two possible outcomes: success or failure.
    """

    def __init__(self, p: float) -> None:
        super().__init__(distribution_name="bernoulli", params=dict(p=p))


class BinomialSampleGenerator(DiscreteSampleGenerator):
    """
    Generates samples from the Binomial distribution.

    Parameters:
        n (int): The number of trials. This must be a positive integer value.
            Mathematically, n represents the number of independent Bernoulli trials.
        p (float): The probability of success on each trial. This must be a scalar value between 0 and 1.
            Mathematically, p represents the probability of success in each Bernoulli trial.
            A Bernoulli trial is a random experiment with two possible outcomes: success or failure.
    """

    def __init__(self, n: int, p: float) -> None:
        super().__init__(distribution_name="binom", params=dict(n=n, p=p))


class PoissonSampleGenerator(DiscreteSampleGenerator):
    """
    Generates samples from the Poisson distribution.

    Parameters:
      mu (float): The average number of successes. This must be a non-negative real number.
        Mathematically, mu represents the mean or expected value of the Poisson distribution.
        The Poisson distribution describes the number of events that occur in a fixed interval of time or space,
        given a known average rate of occurrence.
    """

    def __init__(self, mu: float) -> None:
        super().__init__(distribution_name="poisson", params=dict(mu=mu))


class GeometricSampleGenerator(DiscreteSampleGenerator):
    """
    Generates samples from the Geometric distribution.

    Parameters:
      p (float): The probability of success. This must be a scalar value between 0 and 1.
        Mathematically, p represents the probability of success in each trial of a geometric experiment.
        A geometric experiment is a sequence of independent Bernoulli trials, where the trials continue until the first success occurs.
    """

    def __init__(self, p: float) -> None:
        super().__init__(distribution_name="geom", params=dict(p=p))


class NegativeBinomialSampleGenerator(DiscreteSampleGenerator):
    """
    Generates samples from the Negative Binomial distribution.

    Parameters:
      r (int): The number of successes. This must be a positive integer value.
        Mathematically, r represents the number of successes in a negative binomial experiment.
        A negative binomial experiment is a sequence of independent Bernoulli trials, where the trials continue
        until the rth success occurs.
      p (float): The probability of success on each trial. This must be a scalar value between 0 and 1.
        Mathematically, p represents the probability of success in each Bernoulli trial.
        A Bernoulli trial is a random experiment with two possible outcomes: success or failure.
    """

    def __init__(self, r: int, p: float) -> None:
        super().__init__(distribution_name="nbinom", params=dict(r=r, p=p))
