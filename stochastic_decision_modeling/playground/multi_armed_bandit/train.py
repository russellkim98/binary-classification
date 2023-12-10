import numpy as np
import tensorflow as tf
import tf_agents.bandits as bandits
from tf_agents.bandits.replay_buffers.bandit_replay_buffer import \
    BanditReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics

means = [0.1, 0.2, 0.3, 0.45, 0.5]

######## Environment creation
# Bernoulli environment
bern_env = bandits.environments.bernoulli_py_environment.BernoulliPyEnvironment(
    means=means, batch_size=8
)

# Convert to tf_py_environment
env = tf_py_environment.TFPyEnvironment(bern_env)


######## Agent creation
agent = bandits.agents.bernoulli_thompson_sampling_agent.BernoulliThompsonSamplingAgent(
    time_step_spec=env.time_step_spec(),
    action_spec=env.action_spec(),
    dtype=tf.float64,
    batch_size=8,
)


def optimal_reward_fn(unused_observation):
    return np.max(means)


def optimal_action_fn(unused_observation):
    return np.int32(np.argmax(means))


######## Metrics
regret_metric = bandits.metrics.tf_metrics.RegretMetric(optimal_reward_fn)
suboptimal_arms_metric = bandits.metrics.tf_metrics.SuboptimalArmsMetric(
    optimal_action_fn
)

# `step_metric` records the number of individual rounds of bandit interaction;
# that is, (number of trajectories) * batch_size.
step_metric = tf_metrics.EnvironmentSteps()
metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.AverageEpisodeLengthMetric(batch_size=env.batch_size),
    regret_metric,
    suboptimal_arms_metric,
    tf_metrics.AverageReturnMultiMetric(
        batch_size=env.batch_size, reward_spec=env.reward_spec()
    ),
]

observers = [step_metric] + metrics

# Create Driver
driver = DynamicStepDriver(
    env=env,
    policy=agent.collect_policy,
    num_steps=2 * env.batch_size,
    observers=observers,
)

# Create replay buffer
replay_buffer = BanditReplayBuffer(
    data_spec=agent.policy.trajectory_spec, batch_size=env.batch_size
)

regret_values = []
for _ in range(1000):
    driver.run()
    loss_info = agent.train(replay_buffer.gather_all())
    replay_buffer.clear()
    regret_values.append(regret_metric.result())


from pprint import pprint

print("donnnne!")
