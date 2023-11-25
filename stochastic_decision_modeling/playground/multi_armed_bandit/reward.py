import abc

import numpy as np
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.drivers import driver
from tf_agents.environments import py_environment, tf_environment, tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

nest = tf.nest
# We need to add another dimension here because the agent expects the
# trajectory of shape [batch_size, time, ...], but in this tutorial we assume
# that both batch size and time are 1. Hence all the expand_dims.


def trajectory_for_bandit(initial_step, action_step, final_step):
    return trajectory.Trajectory(
        observation=tf.expand_dims(initial_step.observation, 0),
        action=tf.expand_dims(action_step.action, 0),
        policy_info=action_step.info,
        reward=tf.expand_dims(final_step.reward, 0),
        discount=tf.expand_dims(final_step.discount, 0),
        step_type=tf.expand_dims(initial_step.step_type, 0),
        next_step_type=tf.expand_dims(final_step.step_type, 0),
    )
