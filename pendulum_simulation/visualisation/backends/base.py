from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from ..scene.primitives import Primitive
from ..scene.scene import Scene, Viewport
from ..ui_events import InputEvent, InputSink


class SceneGui(ABC):
    """A window/plot that can display a Scene and report user input."""

    def __init__(self):
        self._sink: InputSink | None = None

    def set_event_sink(self, sink: InputSink | None):
        self._sink = sink

    def _emit(self, event: InputEvent) -> None:
        """Backends call this to forward an input event."""
        if self._sink is not None:
            self._sink.handle(event)

    @abstractmethod
    def attach(self, scene: Scene):
        """Create one native object per primitive. Call once."""

    @abstractmethod
    def refresh(self):
        """Copy every primitive's current geometry/style to its native object
        and request a redraw."""

    @abstractmethod
    def run(self):
        """Show the window and block until it is closed."""


class PrimitiveAdapter(ABC):
    """Draws one primitive type in one toolkit."""

    @abstractmethod
    def create(self, prim: Primitive, surface: Any) -> Any:
        """Make the native object (and add it to the `surface`)."""

    @abstractmethod
    def update(self, gui_obj: Any, prim: Primitive, surface: Any):
        """Copy geometry, style and visibility from `prim` to `gui_obj`."""


class RetainedSceneGui(SceneGui):
    """Shared attach/refresh loops for toolkits with retained-mode objects.
    A subclass supplies an adapter per primitive type, a drawing surface
    and a redraw request."""

    def __init__(self, adapters: Mapping[type[Primitive], PrimitiveAdapter]):
        super().__init__()
        self._adapters = dict(adapters)
        self._surface: Any = None
        self._attached = False
        self._entries: list[tuple[PrimitiveAdapter, Any, Primitive]] = []

    def attach(self, scene: Scene) -> None:
        if self._attached:
            raise RuntimeError("View is already attached to a scene")

        self._surface = self._create_surface(scene.viewport)

        for name, prim in scene:
            adapter = self._adapters.get(type(prim))
            if adapter is None:
                raise TypeError(
                    f"{type(self).__name__} cannot draw {type(prim).__name__} ({name!r})"
                )

            self._entries.append((adapter, adapter.create(prim, self._surface), prim))

        self._attached = True

    def refresh(self) -> None:
        for adapter, gui_obj, prim in self._entries:
            adapter.update(gui_obj, prim, self._surface)
        self._request_redraw()

    @abstractmethod
    def _create_surface(self, viewport: Viewport) -> Any:
        """Configure the drawing area for `viewport`; return the surface given to adapters."""

    @abstractmethod
    def _request_redraw(self) -> None: ...
