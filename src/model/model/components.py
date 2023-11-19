from abc import ABC, abstractmethod


class State(ABC):
    """
    Abstract base class for the state of the environment.
    """

    pass


class Action(ABC):
    """
    Abstract base class for actions.
    """

    pass


class Environment(ABC):
    """
    Abstract base class for environments.
    """

    @abstractmethod
    def update(self, state: State, action: Action) -> State:
        """
        Updates the environment based on the current state and action.
        """
        raise NotImplementedError


class Policy(ABC):
    """
    Abstract base class for policies.
    """

    pass


class Objective(ABC):
    """
    Abstract base class for objectives.
    """

    pass
