import numpy as np
from stochastic_decision_modeling.model.sdm import (
    State,
    Action,
    Observation,
    Transition,
    Objective,
)
from pprint import pprint
from tqdm import tqdm


def generate_observation_fn(stores):
    def store_fn(state: State, action: Action) -> float:
        del state
        params = stores[action.value]
        mu, sigma = params["mu"], params["sigma"]
        return rng.normal(mu, sigma)

    return store_fn


def transition_fn(state: State, action: Action, observation: Observation):
    reward = observation.observation_fn(state, action)

    # selected store
    store = action.value

    # get existing state
    physical, information, belief = state.physical, state.information, state.belief

    # Update physical (skip for now)

    # Update information and belief
    size = information[store]["size"]
    mu, sigma = belief[store]["mu"], belief[store]["sigma"]
    var = sigma**2

    # use welford's online algorithm to update mu and sigma
    delta_mu = reward - mu
    new_size = size + 1
    new_mu = mu + (delta_mu) / new_size
    new_var = var + ((delta_mu) ** 2 - var) / new_size

    information[store]["size"] = new_size
    belief[store]["mu"] = new_mu
    belief[store]["sigma"] = np.sqrt(new_var)

    return State(physical=physical, information=information, belief=belief)


def main(rng: np.random.Generator):
    # True values
    store_names = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    store_mu = rng.integers(50, 100, size=len(store_names))
    store_sigma = rng.integers(5, 10, size=len(store_names))
    stores = {
        name: {"mu": mu, "sigma": sigma}
        for name, mu, sigma in zip(store_names, store_mu, store_sigma)
    }
    pprint("True values")
    pprint(stores)

    # Belief and state variable
    physical = {store: {"inventory": 0} for store in stores}
    information = {store: {"size": 0} for store in stores}
    belief = {store: {"mu": 0.0, "sigma": 1.0} for store in stores}

    # Transition
    transition = Transition(transition=transition_fn)

    ################
    # START
    ################

    # Time = 0
    state = State(physical=physical, information=information, belief=belief)
    action = Action(value="New York")

    # Time = 1
    for i in range(1000):
        observation = Observation(
            state=state,
            action=action,
            observation_fn=generate_observation_fn(stores),
        )
        # TODO: this should be the result of a policy
        action = Action(value="Chicago")
        state = transition.transition(state, action, observation)

        if i % 10 == 0:
            pprint(f"Time = {i+1}")
    test = "weraweraerf"
    pprint(state.information)
    pprint(state.belief)
    pprint(action)

    test = "hi"


if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    main(rng)
