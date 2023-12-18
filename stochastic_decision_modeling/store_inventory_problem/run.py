import tf_agents
from tf_agents.environments import PyEnvironment
import tensorflow as tf
from tf_agents.typing import types
import typing as T
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.specs import array_spec
import numpy as np


class StoreInventoryEnvironment(PyEnvironment):
    def __init__(self):
        pass

    def observation_spec(self) -> T.Dict[str, array_spec.ArraySpec]:
        return {"observe": array_spec.ArraySpec(shape=(1,), dtype=np.float32)}

    def action_spec(self):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
