from __future__ import annotations

from .state import AbsState


class InvalidStateError(Exception):
    def __init__(self, message: str, state: AbsState, constraint: str):
        super().__init__(message)
        self.state = state
        self.constraint = constraint
