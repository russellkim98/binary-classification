import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import tensorflow as tf
from pprint import pprint
import plotly.express as px

from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metric

# Imports for example.
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from stochastic_decision_modeling.model.multiarmed_bandit.environment import (
    StationaryStochastic,
)
from stochastic_decision_modeling.model.multiarmed_bandit.agent import LinearUCB
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from stochastic_decision_modeling.model.multiarmed_bandit.reward import RegretMetric


nest = tf.nest
tf.compat.v1.enable_v2_behavior()


def train_step(
    driver: dynamic_step_driver.DynamicStepDriver,
    agent: LinearUCB,
    replay_buffer: tf_uniform_replay_buffer.TFUniformReplayBuffer,
    reward: tf_metric.TFStepMetric,
    step: int,
):
    driver.run()
    loss_info = agent.train(replay_buffer.gather_all())
    replay_buffer.clear()
    results = {
        "action": driver.policy.action(driver.env.current_time_step()).action.numpy()[
            0
        ],
        "reward": reward.result().numpy(),
        "loss": loss_info.loss.numpy(),
        "step": step,
    }
    return results


def run(params):
    environment = tf_py_environment.TFPyEnvironment(
        StationaryStochastic(
            reward_arms=dict.get(params, "reward_arms", []),
            batch_size=dict.get(params, "batch_size", 1),
        ),
    )
    agent = LinearUCB(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
    )
    reward = RegretMetric(
        rewards=dict.get(params, "reward_arms", []),
    )

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.policy.trajectory_spec,
        batch_size=dict.get(params, "batch_size", 1),
        max_length=dict.get(params, "steps_per_loop", 1),
    )

    observers = [replay_buffer.add_batch, reward]

    driver = dynamic_step_driver.DynamicStepDriver(
        env=environment,
        policy=agent.collect_policy,
        num_steps=(
            dict.get(params, "batch_size", 1) * dict.get(params, "steps_per_loop", 1)
        ),
        observers=observers,
    )

    metrics = []
    for step in range(dict.get(params, "num_iterations", 1)):
        result_step = train_step(
            driver=driver,
            agent=agent,
            replay_buffer=replay_buffer,
            reward=reward,
            step=step,
        )
        if step % 10 == 0:
            print(f"Step {step}")
        metrics.append(result_step)
    metric_df = pd.DataFrame.from_records(metrics)
    return metric_df


def main(params):
    results = run(params)
    # Make plots
    figs = {
        "action": px.scatter(results, x="step", y="action", title="action"),
        "loss": px.scatter(results, x="step", y="loss", title="loss"),
        "reward": px.scatter(results, x="step", y="reward", title="reward"),
        "histogram": px.histogram(
            results, x="reward", color="action", title="histogram"
        ),
    }
    figs["histogram"].show()
    fig = make_subplots(rows=2, cols=2)
    fig.add_trace(figs["action"].data[0], row=1, col=1)
    fig.add_trace(figs["loss"].data[0], row=1, col=2)
    fig.add_trace(figs["reward"].data[0], row=2, col=1)

    fig.update_yaxes(matches=None)
    fig.show()


def get_params():
    reward_arms = [float(x * 5) for x in range(5)]
    batch_size = 1
    observation_spec = tensor_spec.TensorSpec([1], tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2
    )
    num_iterations = 10
    steps_per_loop = 1

    # Dict of params
    params = {
        "reward_arms": reward_arms,
        "batch_size": batch_size,
        "observation_spec": observation_spec,
        "time_step_spec": time_step_spec,
        "action_spec": action_spec,
        "num_iterations": num_iterations,
        "steps_per_loop": steps_per_loop,
    }
    return params


if __name__ == "__main__":
    params = get_params()
    metrics = main(params)
