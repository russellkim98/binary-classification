# Imports for example.
import numpy as np
from tf_agents.bandits.environments import \
    stationary_stochastic_py_environment as sspe
from tf_agents.environments import tf_py_environment

batch_size = 2  # @param
arm0_param = [-3, 0, 1, -2]  # @param
arm1_param = [1, -2, 3, 0]  # @param
arm2_param = [0, 0, 1, 1]  # @param


def context_sampling_fn(batch_size):
    """Contexts from [-10, 10]^4."""

    def _context_sampling_fn():
        return np.random.randint(-10, 10, [batch_size, 4]).astype(np.float32)

    return _context_sampling_fn


class LinearNormalReward(object):
    """A class that acts as linear reward function when called."""

    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma

    def __call__(self, x):
        mu = np.dot(x, self.theta)
        return np.random.normal(mu, self.sigma)


arm0_reward_fn = LinearNormalReward(arm0_param, 1)
arm1_reward_fn = LinearNormalReward(arm1_param, 1)
arm2_reward_fn = LinearNormalReward(arm2_param, 1)

environment = tf_py_environment.TFPyEnvironment(
    sspe.StationaryStochasticPyEnvironment(
        context_sampling_fn(batch_size),
        [arm0_reward_fn, arm1_reward_fn, arm2_reward_fn],
        batch_size=batch_size,
    )
)
