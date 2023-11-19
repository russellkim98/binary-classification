import abc


class DataGenerator(abc.ABC):
    """Abstract class for generating samples from discrete probability distributions."""

    @abc.abstractmethod
    def __init__(self, distribution_name: str, *params):
        """Initialize the discrete sample generator with the specified distribution and parameters."""
        pass

    @abc.abstractmethod
    def sample(self, size: int):
        """Generate a sample of the specified size from the discrete distribution."""
        pass
