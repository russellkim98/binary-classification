import tensorflow as tf
from tf_agents.trajectories import trajectory

# We need to add another dimension here because the agent expects the
# trajectory of shape [batch_size, time, ...], but in this tutorial we assume
# that both batch size and time are 1. Hence all the expand_dims.


class BanditTrajectory:
    @classmethod
    def trajectory_for_bandit(cls, initial_step, action_step, final_step):
        return trajectory.Trajectory(
            observation=tf.expand_dims(initial_step.observation, 0),
            action=tf.expand_dims(action_step.action, 0),
            policy_info=action_step.info,
            reward=tf.expand_dims(final_step.reward, 0),
            discount=tf.expand_dims(final_step.discount, 0),
            step_type=tf.expand_dims(initial_step.step_type, 0),
            next_step_type=tf.expand_dims(final_step.step_type, 0),
        )
