import tensorflow as tf
from tf_agents.bandits.metrics import tf_metrics


def compute_optimal_reward(observation):
    expected_reward_for_arms = [
        tf.linalg.matvec(observation, tf.cast(arm0_param, dtype=tf.float32)),
        tf.linalg.matvec(observation, tf.cast(arm1_param, dtype=tf.float32)),
        tf.linalg.matvec(observation, tf.cast(arm2_param, dtype=tf.float32)),
    ]
    optimal_action_reward = tf.reduce_max(expected_reward_for_arms, axis=0)
    return optimal_action_reward


regret_metric = tf_metrics.RegretMetric(compute_optimal_reward)
