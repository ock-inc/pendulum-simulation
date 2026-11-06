from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlChannelSpec:
    """One scalar input to the system."""

    name: str
    minimum: float = -1.0
    maximum: float = 1.0
    default: float = 0.0


@dataclass(frozen=True)
class ControlSpec:
    """Ordered description of the control vector u.

    The position of a channel in `channels` is its index in u.
    A System returns one of these from `control_spec()`; an uncontrolled
    system returns `ControlSpec()` (size 0).
    """

    channels: tuple[ControlChannelSpec, ...] = ()

    @property
    def size(self) -> int:
        return len(self.channels)

    def index(self, name: str) -> int:
        for i, c in enumerate(self.channels):
            if c.name == name:
                return i

        known = [c.name for c in self.channels]
        raise KeyError(f"Unknown control channel {name!r}; known channels: {known}")

    def default_vector(self) -> np.ndarray:
        return np.array([c.default for c in self.channels], dtype=float)


class ControlChannel:
    """Holds the latest control vector."""

    def __init__(self, spec: ControlSpec):
        self.spec = spec
        self._u = spec.default_vector()

    def set(self, name: str, value: float) -> None:
        """Set one channel, clipped to its [minimum, maximum]."""
        i = self.spec.index(name)
        c = self.spec.channels[i]
        self._u[i] = min(max(float(value), c.minimum), c.maximum)

    def release(self, name: str) -> None:
        """Return one channel to its default value."""
        i = self.spec.index(name)
        self._u[i] = self.spec.channels[i].default

    def read(self) -> np.ndarray:
        """Current control vector (a copy, so callers can't mutate it)."""
        return self._u.copy()
