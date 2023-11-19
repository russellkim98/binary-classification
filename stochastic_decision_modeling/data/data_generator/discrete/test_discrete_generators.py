import numpy as np
import pytest

from stochastic_decision_modeling.data.data_generator.discrete.discrete_generators import (
    BernoulliSampleGenerator,
    BinomialSampleGenerator,
    GeometricSampleGenerator,
    NegativeBinomialSampleGenerator,
    PoissonSampleGenerator,
)


@pytest.fixture
def bernoulli_generator():
    return BernoulliSampleGenerator(p=0.5)


@pytest.fixture
def binomial_generator():
    return BinomialSampleGenerator(n=10, p=0.5)


@pytest.fixture
def poisson_generator():
    return PoissonSampleGenerator(mu=5.0)


@pytest.fixture
def geometric_generator():
    return GeometricSampleGenerator(p=0.5)


@pytest.fixture
def negative_binomial_generator():
    return NegativeBinomialSampleGenerator(r=3, p=0.5)


def test_bernoulli_sample_generator(bernoulli_generator):
    """
    Tests the BernoulliSampleGenerator class.
    """

    # Generate 100 samples from the generator
    samples = bernoulli_generator.sample(size=10000)

    # Check that the samples are either 0 or 1
    assert np.all(np.isin(samples, [0, 1]))

    # Check that the mean of the samples is close to the specified probability
    assert np.isclose(np.mean(samples), bernoulli_generator.params.get("p"), atol=0.1)


def test_binomial_sample_generator(binomial_generator):
    """
    Tests the BinomialSampleGenerator class.
    """

    # Generate 100 samples from the generator
    samples = binomial_generator.sample(size=10000)

    # Check that the samples are non-negative integers
    assert np.all(np.isin(samples, np.arange(binomial_generator.params.get("n") + 1)))

    # Check that the mean of the samples is close to the expected value
    assert np.isclose(
        np.mean(samples),
        binomial_generator.params.get("n") * binomial_generator.params.get("p"),
        atol=0.1,
    )


def test_poisson_sample_generator(poisson_generator):
    """
    Tests the PoissonSampleGenerator class.
    """

    # Generate 100 samples from the generator
    samples = poisson_generator.sample(size=10000)

    # Check that the samples are non-negative integers
    assert np.all(samples >= 0)

    # Check that the mean of the samples is close to the specified mean
    assert np.isclose(np.mean(samples), poisson_generator.params.get("mu"), atol=0.1)


def test_geometric_sample_generator(geometric_generator):
    """
    Tests the GeometricSampleGenerator class.
    """

    # Generate 100 samples from the generator
    samples = geometric_generator.sample(size=10000)

    # Check that the samples are non-negative integers
    assert np.all(samples >= 0)

    # Check that the expected value of the samples is close to the specified probability
    assert np.isclose(
        1 / geometric_generator.params.get("p"), np.mean(samples), atol=0.1
    )


def test_negative_binomial_sample_generator(negative_binomial_generator):
    """
    Tests the NegativeBinomialSampleGenerator class.
    """

    # Generate 100 samples from the generator
    samples = negative_binomial_generator.sample(size=10000)

    # Check that the samples are non-negative integers
    assert np.all(samples >= 0)

    # Check that the expected value of the samples is close to the specified mean
    assert np.isclose(
        (1 - negative_binomial_generator.params.get("p"))
        * negative_binomial_generator.params.get("r")
        / negative_binomial_generator.params.get("p"),
        np.mean(samples),
        atol=0.3,
    )
