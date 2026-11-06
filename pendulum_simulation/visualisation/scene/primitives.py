from __future__ import annotations

import math
from abc import ABC
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np

from .style import Style

Point = tuple[float, float]
Vector = tuple[float, float]


@dataclass(kw_only=True)
class Primitive(ABC):
    """A minimal visible element."""

    kind: ClassVar[str]

    layer: int = 0  # z-order, higher drawn on top
    visible: bool = True
    style: Style = field(default_factory=Style)


@dataclass(kw_only=True)
class Line(Primitive):
    """A straight line between 2 points."""

    kind = "line"

    p0: Point
    p1: Point


@dataclass(kw_only=True)
class Polyline(Primitive):
    """A line made up of straight lines between a sequence of points."""

    kind = "polyline"

    capacity: int | None = None

    _points: deque = field(init=False)

    def __post_init__(self):
        self._points = deque(maxlen=self.capacity)

    def push(self, point: Point):
        self._points.append(point)

    def pop(self) -> Point:
        return self._points.popleft()

    def set_points(self, points: np.ndarray):
        self._points.clear()
        for point in points:
            self._points.append(tuple(point))

    @property
    def array_points(self) -> np.ndarray:
        return np.array(self._points)


@dataclass(kw_only=True)
class Circle(Primitive):
    """A circle with a centre and a radius."""

    kind = "circle"

    center: Point
    radius: float


@dataclass(kw_only=True)
class Rect(Primitive):
    """A rectangle with a centre, side lengths, and an angle.

    Angle is anticlockwise positive, and size is (width, height)"""

    kind = "rect"

    center: Point
    size: Vector
    angle: float = 0.0

    @property
    def corners(self) -> tuple[Point, ...]:
        """Four corners (anticlockwise from bottom-left before rotation)."""
        sin = math.sin(self.angle)
        cos = math.cos(self.angle)

        half_width = self.size[0] / 2
        half_height = self.size[1] / 2

        return tuple(
            (
                self.center[0] + dx * cos - dy * sin,
                self.center[1] + dx * sin + dy * cos,
            )
            for dx, dy in (
                (-half_width, -half_height),
                (half_width, -half_height),
                (half_width, half_height),
                (-half_width, half_height),
            )
        )


@dataclass(kw_only=True)
class Arrow(Primitive):
    """An arrow that points in a given direction."""

    kind = "arrow"

    origin: Point
    vector: Vector
    head_fraction: float = 0.25
    head_angle: float = math.radians(25)

    @property
    def points(self) -> tuple[Point, ...]:
        """One polyline drawing shaft + arrowhead: origin -> tip -> left -> tip -> right."""
        length = math.hypot(*self.vector)
        tip = (
            self.origin[0] + self.vector[0],
            self.origin[1] + self.vector[1],
        )

        if length < 1e-12:
            return (self.origin, self.origin)

        ux = self.vector[0] / length
        uy = self.vector[1] / length

        head_len = self.head_fraction * length
        pts = [self.origin, tip]

        for a in (self.head_angle, -self.head_angle):
            sin = math.sin(a)
            cos = math.cos(a)

            pts += [
                (
                    tip[0] - head_len * (ux * cos - uy * sin),
                    tip[1] - head_len * (ux * sin + uy * cos),
                ),
                tip,
            ]
        return tuple(pts[:-1])


@dataclass(kw_only=True)
class Text(Primitive):
    """A text object."""

    kind = "text"

    position: Point
    text: str
    align_h: Literal["left", "centre", "right"] = "centre"
    align_v: Literal["bottom", "centre", "top"] = "centre"
