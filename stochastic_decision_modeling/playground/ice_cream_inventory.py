import typing as T
import numpy as np


class AttributeMetaclass(type):
    def __new__(mcs, name, bases, attrs):
        init_method = attrs.pop("__init__", None)

        def __init__(self, *args, **kwargs):
            attributes = kwargs.pop("attributes", {})
            for attr_name, attr_value in attributes.items():
                setattr(self, attr_name, attr_value)

                def transition_wrapper(self):
                    if hasattr(self, f"_transition_{attr_name}"):
                        getattr(self, f"_transition_{attr_name}")()

            if init_method:
                init_method(self, *args, **kwargs)

        attrs["__init__"] = __init__
        return super().__new__(mcs, name, bases, attrs)


class IceCreamTest(metaclass=AttributeMetaclass):
    attributes = {"person": "John Doe", "age": 30, "active": True}

    def _transition_age(self):
        # Update logic for "age"
        self.age = 40

    def transition(self):
        # Call generated update methods based on attributes
        for attr_name in self.attributes:
            getattr(self, f"transition_{attr_name}")()


class IceCreamState:
    def __init__(self, price, inventory, cost, forecast, demand_std, forecast_std):
        self.price = price
        self.inventory = inventory
        self.cost = cost
        self.forecast = forecast
        self.demand_std = demand_std
        self.forecast_std = forecast_std
        self.alpha = 0.5

    # Takes in state, action, and observation and updates every state variable
    def transition(self, purchase, cost, forecast_error, forecast_change):
        self.inventory = max(
            0, self.inventory + purchase - (self.forecast + forecast_error)
        )
        self.demand_std = np.sqrt(
            (1 - self.alpha) * self.demand_std**2 + self.alpha * (forecast_error) ** 2
        )
        self.forecast_std = np.sqrt(
            (1 - self.alpha) * self.forecast_std**2
            + self.alpha * (forecast_change) ** 2
        )
        self.forecast = self.forecast + forecast_change
        self.cost = cost
