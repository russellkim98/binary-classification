import typing as T

import numpy as np
import pandas as pd

from stochastic_decision_modeling.data.data_generator.continuous.continuous_generators import (
    BetaSampleGenerator,
    ExponentialSampleGenerator,
    GammaSampleGenerator,
    NormalSampleGenerator,
    UniformSampleGenerator,
)
from stochastic_decision_modeling.data.data_generator.generator import DataGenerator

rng = np.random.default_rng(seed=0)


class MultiArmedBanditState:
    def __init__(self, params: T.Mapping[str, T.Any]):
        self.params = params
        self.n_bandits = len(params)
        self.cur_reward = 0
        self.cum_reward = 0
        self.belief = {
            v: {"mu": 0.0, "sigma": 0.0, "n_trials": 0} for v in params.keys()
        }

    def update(self, bandit_id: str, reward: float):
        """Update the state by recalculating the mu and sigma for the bandit"""
        bandit = self.params[bandit_id]
        n = bandit["n_trials"]
        old_mu = bandit["mu"]
        new_mu = (old_mu * n + reward) / (n + 1)
        bandit["mu"] = new_mu

        old_sigma = bandit["sigma"]
        new_sigma = np.sqrt(
            ((n - 1) * old_sigma**2 + (reward - new_mu) * (reward - old_mu)) / (n - 1)
        )
        bandit["sigma"] = new_sigma

        bandit["n_trials"] = n + 1


class MultiArmedBanditEnvironment:
    def __init__(self, params: T.Mapping[str, DataGenerator]):
        self.params = params
        self.n_bandits = len(params)

    def get_responses(self, bandit_id: str, n_samples: int = 1):
        """Get the response for the given vendor."""
        response = self.params[bandit_id].sample(n_samples)
        if not response:
            return 0.0
        return response[0]


class MultiArmedBanditPolicy:
    def __init__(self):
        pass


class MultiArmedBanditModel:
    def __init__(self, state: MultiArmedBanditState, env: MultiArmedBanditEnvironment):
        self.state = state
        self.env = env

    def step(self, action: str):
        reward = self.env.get_responses(action, n_samples=1)
        self.state.update(action, reward)

    pass


def main():
    import ipdb

    ipdb.set_trace()
    # list of 10 vendors
    vendors = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    # each vendor has a NormalSampleGenerator with a random loc between 1 and 100 and a random scale between 1 and 10

    vendor_params = {
        vendor: NormalSampleGenerator(loc=rng.uniform(1, 100), scale=rng.uniform(1, 10))
        for vendor in vendors
    }
    state = MultiArmedBanditState(vendor_params)
    env = MultiArmedBanditEnvironment(vendor_params)


if __name__ == "__main__":
    main()
