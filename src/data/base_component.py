"""Base abstract class."""


from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """Base Abstract component Class."""

    def __init__(self, name="Base"):
        """init."""
        self.name = name

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run."""
        pass
