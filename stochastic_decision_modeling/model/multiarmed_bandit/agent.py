from tf_agents.bandits.agents import lin_ucb_agent


class LinearUCB(lin_ucb_agent.LinearUCBAgent):
    def __init__(self, time_step_spec, action_spec):
        super().__init__(time_step_spec, action_spec)
