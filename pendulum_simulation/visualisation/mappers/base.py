from abc import ABC, abstractmethod

from pendulum_simulation.core.state import AbsState

from ..scene.scene import Scene


class SceneMapper[S: AbsState](ABC):
    """Map a `State` type onto a `Scene` of primitives."""

    @abstractmethod
    def build_scene(self) -> Scene:
        """Create every primitive once, with sensible initial geometry."""

    @abstractmethod
    def update(self, scene: Scene, state: S, t: float):
        """Assign new geometry to the existing primitives. Must not add any."""
