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

    pass


class Transition(ABC):
    """
    Abstract base class for transitions.
    """

    pass


class Objective(ABC):
    """
    Abstract base class for objectives.
    """

    pass
