import typing as T
from abc import ABC, abstractmethod

from src.model.model.components import Environment, Objective, Policy, State


class StochasticDecisionModel(ABC):
    """
    Abstract base class for a stochastic decision model.
    """

    def __init__(
        self,
        state_class: State,
        objective_class: Objective,
        environment_class: Environment,
        policy_class: Policy,
    ):
        """
        Constructor for the StochasticDecisionModel class.

        Args:
            state_class (Type[State]): The class representing the state of the environment.
            objective_class (Type[Objective]): The class representing the objective function.
            environment_class (Type[Environment]): The class representing the environment.
            policy_class (Type[Policy]): The class representing the policy.

        """
        self.state_class = state_class
        self.environment_class = environment_class
        self.policy_class = policy_class
        self.objective_class = objective_class

    @abstractmethod
    def step(self, action: T.Any) -> T.Dict[str, T.Any]:
        """
        Performs a single step of the decision process. This should update the environment based on the current state and action taken, update
        the state based on the environment, then update the class of possible actions.

        """

        # Update environment
        # Update state
        # Update class of possible actions

        raise NotImplementedError

    @abstractmethod
    def transition(self):
        """
        Updates the state given the

        """

    @abstractmethod
    def update_objective_function(self, state: State, reward: float) -> float:
        """
        Updates the objective function based on the current state and reward.

        Args:
            state (State): The current state of the environment.
            reward (float): The reward received in the current step.

        Returns:
            float: The updated objective function value.
        """
        raise NotImplementedError

    @abstractmethod
    def get_possible_actions(self) -> List[Action]:
        """
        Returns the list of possible actions that can be taken in the current state.

        Returns:
            List[Action]: The list of possible actions.
        """
        raise NotImplementedError
