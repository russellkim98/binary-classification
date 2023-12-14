# Imports for example.
import numpy as np
from tf_agents.bandits.environments.stationary_stochastic_py_environment import (
    StationaryStochasticPyEnvironment,
)
from tf_agents.trajectories.time_step import TimeStep
import typing as T
import copy


def normal_reward_fn(mu, sigma) -> T.Callable[[np.ndarray], T.List[float]]:
    def reward_fn(observation: np.ndarray) -> T.List[float]:
        return [np.random.normal(mu, sigma)]

    return reward_fn


class Store(T.NamedTuple):
    name: str = "Test Reward"
    mu: float = 0.0
    sigma: float = 1.0


class StoreEnvironment(StationaryStochasticPyEnvironment):
    def __init__(
        self,
        stores: T.Sequence[Store],
        constraint_fns: T.Optional[
            T.Sequence[T.Callable[[np.ndarray], T.Sequence[float]]]
        ] = None,
        batch_size: T.Optional[int] = 1,
        name: T.Optional[T.Text] = "store_environment",
    ):
        self._stores = stores
        self._context = self.get_context()
        self._reward_fns = self.get_reward_fn()
        super().__init__(
            context_sampling_fn=self._context,
            reward_fns=self._reward_fns,
            constraint_fns=constraint_fns,
            batch_size=batch_size,
            name=name,
        )

    def get_state(self) -> TimeStep:
        return copy.deepcopy(self._current_time_step)

    def set_state(self, state: TimeStep):
        self._current_time_step = state
        self._states = state.observation

    def get_context(self):
        return lambda: np.ones((1, len(self._stores)), dtype=np.int32)

    def get_reward_fn(self):
        return [normal_reward_fn(store.mu, store.sigma) for store in self._stores]
