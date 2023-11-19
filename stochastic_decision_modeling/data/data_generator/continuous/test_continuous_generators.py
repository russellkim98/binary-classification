import numpy as np
import pytest

from stochastic_decision_modeling.data.data_generator.continuous.continuous_generators import (
    BetaSampleGenerator,
    ExponentialSampleGenerator,
    GammaSampleGenerator,
    NormalSampleGenerator,
    UniformSampleGenerator,
)


@pytest.fixture
def uniform_generator():
    return UniformSampleGenerator(loc=0.0, scale=1.0)


@pytest.fixture
def normal_generator():
    return NormalSampleGenerator(loc=0.0, scale=1.0)


@pytest.fixture
def gamma_generator():
    return GammaSampleGenerator(alpha=1.0, loc=0.0, scale=1.0)


@pytest.fixture
def exponential_generator():
    return ExponentialSampleGenerator(loc=0.0, scale=1.0)


@pytest.fixture
def beta_generator():
    return BetaSampleGenerator(alpha=1.0, beta=1.0, loc=0.0, scale=1.0)


def test_uniform_sample_generator(uniform_generator):
    """
    Tests the UniformSampleGenerator class.
    """

    # Generate 100 samples from the generator
    samples = uniform_generator.sample(size=10000)

    # Check that the samples are within the range of the distribution
    assert np.min(samples) >= 0.0
    assert np.max(samples) <= 1.0


def test_normal_sample_generator(normal_generator):
    """
    Tests the NormalSampleGenerator class.
    """

    # Generate 100 samples from the generator
    samples = normal_generator.sample(size=10000)

    # Check that the mean of the samples is close to the specified mean
    assert np.isclose(np.mean(samples), 0, atol=0.1)

    # Check that the standard deviation of the samples is close to the specified standard deviation
    assert np.isclose(np.std(samples), 1, atol=0.1)


def test_gamma_sample_generator(gamma_generator):
    """
    Tests the GammaSampleGenerator class.
    """

    # Generate 100 samples from the generator
    samples = gamma_generator.sample(size=10000)

    # Check that the samples are positive
    assert np.all(samples > 0.0)


def test_exponential_sample_generator(exponential_generator):
    """
    Tests the ExponentialSampleGenerator class.
    """

    # Generate 100 samples from the generator
    samples = exponential_generator.sample(size=10000)

    # Check that the samples are positive
    assert np.all(samples > 0.0)


def test_beta_sample_generator(beta_generator):
    """
    Tests the BetaSampleGenerator class.
    """

    # Generate 100 samples from the generator
    samples = beta_generator.sample(size=10000)

    # Check that the samples are between 0.0 and 1.0
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)
