import abc
import typing as T

import scipy.stats as stats


class DataGenerator(abc.ABC):
    """Abstract class for generating samples from discrete probability distributions."""

    @abc.abstractmethod
    def __init__(self, distribution_name: str, params: T.Dict[str, T.Any]):
        """Initialize the discrete sample generator with the specified distribution and parameters."""
        pass

    @abc.abstractmethod
    def sample(self, size: int):
        """Generate a sample of the specified n from the discrete distribution."""
        pass


class DiscreteSampleGenerator(DataGenerator):
    """Implementation of the DiscreteSampleGenerator abstract class."""

    def __init__(self, distribution_name: str, params: T.Dict[str, T.Any]):
        """Initialize the discrete sample generator with the specified distribution and parameters."""
        super().__init__(distribution_name, params)
        self.distribution = getattr(stats, distribution_name)(*params.values())
        self.params = params

    def sample(self, size: int):
        """Generate a sample of the specified n from the discrete distribution."""
        return self.distribution.rvs(size)


class ContinuousSampleGenerator(DataGenerator):
    """implementation of the ContinuousSampleGenerator abstract class."""

    def __init__(self, distribution_name: str, params: T.Dict[str, T.Any]):
        """Initialize the discrete sample generator with the specified distribution and parameters."""
        super().__init__(distribution_name, params)
        self.distribution = getattr(stats, distribution_name)(*params.values())
        self.params = params

    def sample(self, size: int):
        """Generate a sample of the specified n from the discrete distribution."""
        return self.distribution.rvs(size)