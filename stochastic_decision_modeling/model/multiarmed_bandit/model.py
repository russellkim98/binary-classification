import os

print(dir())
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


class MultiArmedBanditEnvironment:
    def __init__(self, params: T.Mapping[str, DataGenerator]):
        self.params = params
        self.logs = []
        self.n_bandits = len(params)
        self.n_trials = 0

    def get_response(self, bandit_id: str):
        """Get the response for the given vendor."""
        self.logs.append({"bandit_id": bandit_id, "n_trials": self.n_trials})
        self.n_trials += 1
        return self.params[bandit_id].sample(1)

    def return_logs_as_pd_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.logs)


class NewsvendorModel:
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
    env = MultiArmedBanditEnvironment(vendor_params)


if __name__ == "__main__":
    main()
