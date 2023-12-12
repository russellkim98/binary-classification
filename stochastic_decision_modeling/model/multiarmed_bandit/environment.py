from tf_agents.bandits.environments import stationary_stochastic_py_environment as sspe
import numpy as np


class LinearNormalReward(object):
    """A class that acts as linear reward function when called."""

    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma

    def __call__(self, x):
        mu = np.dot(x, self.theta)
        return np.random.normal(mu, self.sigma)


batch_size = 2
arms = [[-3, 0, 1, -2], [1, -2, 3, 0], [0, 0, 1, 1]]
arms_reward = [LinearNormalReward(arm, 1) for arm in arms]


def context_sampling_fn(batch_size):
    """Contexts from [-10, 10]^4."""

    def _context_sampling_fn():
        return np.random.randint(-10, 10, [batch_size, 4]).astype(np.float32)

    return _context_sampling_fn


class StationaryStochastic(sspe.StationaryStochasticPyEnvironment):
    def __init__(self, context_sampling_fn, reward_fns, batch_size):
        super().__init__(
            context_sampling_fn=context_sampling_fn(batch_size),
            reward_fns=reward_fns,
            batch_size=batch_size,
        )


default_env = StationaryStochastic(context_sampling_fn, arms_reward, batch_size)
