from abc import ABC, abstractmethod


class BaseVlnAgent(ABC):
    """Common interface for VLN methods used by the UnrealZoo navigation loop."""

    method_name = None

    @classmethod
    def add_cli_args(cls, parser):
        """Register method-specific CLI flags."""

    @abstractmethod
    def reset(self):
        """Reset episode-level model state."""

    @abstractmethod
    def predict(self, observation, instruction, info=None):
        """Return one or more text actions: forward, left, right, or stop."""
