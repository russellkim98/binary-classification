from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.bandits.agents import lin_ucb_agent

import tensorflow as tf

observation_spec = tensor_spec.TensorSpec([4], tf.float32)
time_step_spec = ts.time_step_spec(observation_spec)
action_spec = tensor_spec.BoundedTensorSpec(
    dtype=tf.int32, shape=(), minimum=0, maximum=2
)

default_agent = lin_ucb_agent.LinearUCBAgent(
    time_step_spec=time_step_spec, action_spec=action_spec
)
