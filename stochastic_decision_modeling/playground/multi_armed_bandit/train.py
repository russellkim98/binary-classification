# from stochastic_decision_modeling.playground.multi_armed_bandit import (
#     agent, environment, trajectory)
from stochastic_decision_modeling.playground.multi_armed_bandit import (
    agent, environment, trajectory)

two_way_env = environment.TwoWayPyEnvironment()
sign_agent = agent.SignAgent()
trajectory = trajectory.BanditTrajectory()

step = two_way_env.reset()

for _ in range(10):
    action_step = sign_agent.collect_policy.action(step)
    next_step = two_way_env.step(action_step.action)
    experience = trajectory.trajectory_for_bandit(step, action_step, next_step)
    print(experience)
    sign_agent.train(experience)
    step = next_step
