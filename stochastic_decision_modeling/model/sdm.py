import typing as T


class State(T.NamedTuple):
    """
    The state of the stochastic decision model. Consists of 3 components: physical, information, and belief.
    Theoretically, physical ⊆ information ⊆ belief. However, it's just easier to have them be disjoint.

    Physical - P:
        There are many problems which involve a physical state which is typically some sort of resource being managed.
        For example, this could be
        - a vector P_t
        - a low-dimensional vector P_t = (P_t1, P_t2, P_t3, ..., P_tn) where P_ti is a low-dimensional vector (i might be a blood type, or a type of piece of equipment),
        - a high-dimensional vector P_t = {(P_ta) : a ∈ A}, where a is multidimensional attribute vector

    Information - I:
        In most cases, this evolves exogenously - although there are exceptions. This could be
        - Memoryless:
            I_{t} ⫫ I_{t-1}
        - First order markov:
            I_{t} depends on I_{t-1}
        - Higher order to Full History markov:
            I_{t} depends on I_{t-n}, 0 < n < t

        Realistically, we can create a variable that is a tuple of the history of information and then convert it to a first order markov, so
        we only have to deal with the first two cases. There are methods to deal with full history markovs, but they are not yet implemented.

    Belief - B:
        This captures our beliefs we have about uncertain quantities/parameters that can evolve over time - usually as the result of a decision.
        These could be:
        - Uncertainty about a static parameter:
            Ex. Sales of a product with certain features => Lookup table based on features
            Ex. Impact of price on demand => parameter tuning of some function (eg. derivative is 9% instead of 8%)

        - Uncertainty about dynamic and uncontrollable parameters:
            Ex. Sales of product with certain features may change over time due to consumer preference shifts (think moving average)

        - Uncertainty about dynamic and controllable parameters:
            Ex. We want to optimize inventory of a product. We know manufacturing which flows into the inventory, but we can't track outgoing perfectly.
            Hence, we have an imprecise estimate of inventory at any given time. These are called Partially Observable Markov Decision Processes (POMDP)

    """

    physical: T.Mapping[str, T.Any] = {}
    information: T.Mapping[str, T.Any] = {}
    belief: T.Mapping[str, T.Any] = {}


class Action(T.NamedTuple):
    """
    Actions can come in many forms.
    - Binary (Sell or not to sell)
    - Discrete choice (Choose 1 from n choices)
    - Continuous scalar (Choose value from [a,b])
    - Continuous vector (Real-valued vector)
    - Integer vector (Integer-valued vector)
    - Subset selection (Binary vector indicating inclusion of elements)
    - Multidimensional categorical (Making a choice based on attributes. Ex. Choosing a prescription drug for a patient based on its attributes)
    """

    value: T.Any = None


class Observation(T.NamedTuple):
    """
    This is all the exogenous information that arrives during the interval [t, t+1).
    Therefore, we know this completely when we need to make a decision at t+1. We think
    of information always arriving in continuous time.
    """

    state: State
    action: Action
    observation_fn: T.Callable[[State, Action], T.Any]


class Transition(T.NamedTuple):
    """
    This is how a State transitions from t to t+1.
    """

    transition: T.Callable[[State, Action, Observation], State]


class Objective(T.NamedTuple):
    """
    The objective of the stochastic decision model. Just for convention, we maximize the contribution function.
    If you need minimize something, you can just multiply by -1.
    """

    reward: T.Callable[[State, Action, Observation], float]
