from collections.abc import Iterator
from dataclasses import dataclass

from .primitives import Primitive


@dataclass(frozen=True)
class Viewport:
    """World coordinate rectangle the view should show."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class Scene:
    """Named primitives with a fixed set of members."""

    def __init__(self, viewport: Viewport):
        self.viewport = viewport
        self._primitives: dict[str, Primitive] = {}

    def add(self, name: str, prim: Primitive) -> Primitive:
        if name in self._primitives:
            raise ValueError(f"Scene already has a primitive named {name!r}")
        self._primitives[name] = prim
        return prim

    def get[P: Primitive](self, name: str, typ: type[P]) -> P:
        prim = self._primitives[name]
        if not isinstance(prim, typ):
            raise TypeError(
                f"{name!r} is a {type(prim).__name__}, not a {typ.__name__}"
            )
        return prim

    def __iter__(self) -> Iterator[tuple[str, Primitive]]:
        # TODO: order by z layer
        return iter(self._primitives.items())

    def __len__(self) -> int:
        return len(self._primitives)
