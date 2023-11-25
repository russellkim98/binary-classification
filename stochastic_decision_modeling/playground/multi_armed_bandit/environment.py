import abc

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class BanditPyEnvironment(py_environment.PyEnvironment):
    def __init__(self, observation_spec, action_spec):
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        super(BanditPyEnvironment, self).__init__()

    # Helper functions.
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _empty_observation(self):
        return tf.nest.map_structure(
            lambda x: np.zeros(x.shape, x.dtype), self.observation_spec()
        )

    # These two functions below should not be overridden by subclasses.
    def _reset(self):
        """Returns a time step containing an observation."""
        return ts.restart(self._observe(), batch_size=self.batch_size)

    def _step(self, action):
        """Returns a time step containing the reward for the action taken."""
        reward = self._apply_action(action)
        return ts.termination(self._observe(), reward)

    # These two functions below are to be implemented in subclasses.
    @abc.abstractmethod
    def _observe(self):
        """Returns an observation."""

    @abc.abstractmethod
    def _apply_action(self, action):
        """Applies `action` to the Environment and returns the corresponding reward."""


class TwoWayPyEnvironment(BanditPyEnvironment):
    def __init__(self):
        action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name="action"
        )
        observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=-2, maximum=2, name="observation"
        )

        # Flipping the sign with probability 1/2.
        self._reward_sign = 2 * np.random.randint(2) - 1
        print("reward sign:")
        print(self._reward_sign)

        super(TwoWayPyEnvironment, self).__init__(observation_spec, action_spec)

    def _observe(self):
        self._observation = np.random.randint(-2, 3, (1,), dtype="int32")
        return self._observation

    def _apply_action(self, action):
        return self._reward_sign * action * self._observation[0]
