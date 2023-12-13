import tensorflow as tf
from tf_agents.bandits.metrics import tf_metrics
import typing as T


def compute_optimal_reward(rewards):
    def _compute_optimal_reward(observation):
        expected_reward_for_arms = [
            tf.linalg.matvec(observation, tf.cast(reward, dtype=tf.float32))
            for reward in rewards
        ]
        optimal_action_reward = tf.reduce_max(expected_reward_for_arms, axis=0)
        return optimal_action_reward

    return _compute_optimal_reward


class RegretMetric(tf_metrics.RegretMetric):
    def __init__(self, rewards: T.List[T.List[float]]):
        super().__init__(compute_optimal_reward(rewards))
