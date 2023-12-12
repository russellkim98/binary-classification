import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import numpy as np

import tensorflow as tf

from tf_agents.drivers import driver
from tf_agents.environments import tf_py_environment

# Imports for example.
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from stochastic_decision_modeling.model.multiarmed_bandit.environment import default_env
from stochastic_decision_modeling.model.multiarmed_bandit.agent import default_agent

# from stochastic_decision_modeling.model.multiarmed_bandit.reward import RegretMetric

nest = tf.nest
tf.compat.v1.enable_v2_behavior()


environment = tf_py_environment.TFPyEnvironment(default_env)
agent = default_agent


num_iterations = 90  # @param
steps_per_loop = 1  # @param

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.policy.trajectory_spec,
    batch_size=batch_size,
    max_length=steps_per_loop,
)

observers = [replay_buffer.add_batch, regret_metric]

driver = dynamic_step_driver.DynamicStepDriver(
    env=environment,
    policy=agent.collect_policy,
    num_steps=steps_per_loop * batch_size,
    observers=observers,
)

regret_values = []

for _ in range(num_iterations):
    driver.run()
    loss_info = agent.train(replay_buffer.gather_all())
    replay_buffer.clear()
    regret_values.append(regret_metric.result())
