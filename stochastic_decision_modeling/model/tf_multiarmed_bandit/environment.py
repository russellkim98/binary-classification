from tf_agents.bandits.environments import stationary_stochastic_py_environment as sspe
import numpy as np
import typing as T


class LinearNormalReward(object):
    """A class that acts as linear reward function when called."""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        mu = np.dot(1, self.mu)
        val = np.random.normal(mu, self.sigma)
        print(f"mu: {mu}, sigma: {self.sigma}, x: {x}, val: {val}")
        return val


def context_sampling_fn(batch_size, size):
    """Contexts from [-10, 10]^1."""

    def _context_sampling_fn():
        return np.random.randint(-10, 10, [batch_size, size]).astype(np.float32)

    return _context_sampling_fn


class StationaryStochastic(sspe.StationaryStochasticPyEnvironment):
    def __init__(self, reward_arms: T.List[float], batch_size: int):
        super().__init__(
            context_sampling_fn=context_sampling_fn(batch_size, 1),
            reward_fns=[LinearNormalReward(arm, 1) for arm in reward_arms],
            batch_size=batch_size,
        )
