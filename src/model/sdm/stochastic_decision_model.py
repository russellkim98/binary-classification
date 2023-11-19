from abc import ABC, abstractmethod
from src.models.




class StochasticDecisionModel(ABC):
    """
    Abstract base class for a stochastic decision model.
    """

    def __init__(self, state_class, policy_class, environment_class):
        """
        Constructor for the StochasticDecisionModel class.

        Args:
            state_class (Type[State]): The class representing the state of the environment.
            policy_class (Type[Policy]): The class representing the policy for selecting actions.
            environment_class (Type[Environment]): The class representing the environment.
        """
        self.state_class = state_class
        self.policy_class = policy_class
        self.environment_class = environment_class

    @abstractmethod
    def step(self, state: State, policy: Policy) -> Environment:
        """
        Performs a single step of the decision process.

        Args:
            state (State): The current state of the environment.
            policy (Policy): The policy for selecting actions.

        Returns:
            Environment: The new environment resulting from the action taken.
        """
        raise NotImplementedError

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
