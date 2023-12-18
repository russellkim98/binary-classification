"""
This class is a metaclass that adds functionality for automatic attribute initialization
and validation to classes defined with it.

Key features:

* Extracts and stores custom attributes passed through a "attributes" keyword
  argument in the constructor.
* Applies transition methods named `_transition_<attribute_name>` (optional)
  for each custom attribute to perform any additional processing.
* Raises an error if a transition method is missing for an attribute.
* Integrates with the existing constructor (__init__) method if defined.

Classes using this metaclass should:

* Define any custom attributes as keyword arguments when instantiating the class.
* Optionally define `_transition_<attribute_name>` methods to handle specific
  processing for each attribute.

Remember: all attributes passed through the "attributes" keyword argument will be
automatically set on the instance object.
"""


class AttributeMetaclass(type):

    """
    Overrides the class creation process to provide attribute initialization and validation.

    Args:
        mcs (metaclass): The metaclass itself.
        name (str): The name of the new class.
        bases (tuple): The base classes of the new class.
        attrs (dict): The attributes dictionary of the new class.

    Returns:
        class: The newly created class object.

    Raises:
        TypeError: If any required transition method is missing.
    """

    def __new__(mcs, name, bases, attrs):
        # Pop the existing __init__ method, if present, for later integration.
        init_method = attrs.pop("__init__", None)

        @property
        def __init__(self, *args, **kwargs):
            # Extract custom attributes passed through the "attributes" keyword.
            attributes = kwargs.pop("attributes", {})

            # Set each attribute and apply its transition method (if available).
            for attr_name, attr_value in attributes.items():
                setattr(self, attr_name, attr_value)

                transition_method = getattr(self, f"_transition_{attr_name}", None)
                if transition_method:
                    transition_method()
                else:
                    raise NotImplementedError(
                        f"Update method not implemented for '{attr_name}'"
                    )

            # Call the original __init__ method if it exists.
            if init_method:
                init_method(self, *args, **kwargs)

        # Put the modified __init__ back into the attributes dictionary.
        attrs["__init__"] = __init__

        # Create the new class using the superclass __new__ method.
        return super().__new__(mcs, name, bases, attrs)

    @classmethod
    def __subclasshook__(cls, subclass):
        # Check for presence of all required transition methods in the subclass.
        for attr_name in dir(subclass):
            if attr_name.startswith("_transition_"):
                if not hasattr(subclass, attr_name):
                    raise TypeError(
                        f"Missing abstract transition method for '{attr_name.replace('_transition_', '')}'"
                    )

        # Perform the default subclass check.
        return super().__subclasshook__(subclass)
