"""Toolkit-neutral input events."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ValueChangedEvent:
    """A continuous input moved (e.g. a slider)."""

    name: str
    value: float


@dataclass(frozen=True)
class KeyEvent:
    """A key went down or up."""

    key: str
    pressed: bool


InputEvent = ValueChangedEvent | KeyEvent


class InputSink(Protocol):
    def handle(self, event: InputEvent) -> None: ...
