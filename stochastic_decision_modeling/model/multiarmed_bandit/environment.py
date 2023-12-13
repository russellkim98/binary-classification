from tf_agents.bandits.environments import stationary_stochastic_py_environment as sspe
import numpy as np
import typing as T


class LinearNormalReward(object):
    """A class that acts as linear reward function when called."""

    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma

    def __call__(self, x):
        mu = np.dot(x, self.theta)
        return np.random.normal(mu, self.sigma)


def context_sampling_fn(batch_size):
    """Contexts from [-10, 10]^4."""

    def _context_sampling_fn():
        return np.random.randint(-10, 10, [batch_size, 4]).astype(np.float32)

    return _context_sampling_fn


class StationaryStochastic(sspe.StationaryStochasticPyEnvironment):
    def __init__(self, reward_arms: T.List[T.List[float]], batch_size: int):
        super().__init__(
            context_sampling_fn=context_sampling_fn(batch_size),
            reward_fns=[LinearNormalReward(arm, 1) for arm in reward_arms],
            batch_size=batch_size,
        )
