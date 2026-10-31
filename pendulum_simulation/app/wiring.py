from collections.abc import Mapping

from ..core.control import ControlChannel
from ..runtime.input_router import InputRouter, KeyBinding
from ..visualisation.backends.base import SceneGui


def wire_input(
    view: SceneGui,
    channel: ControlChannel,
    key_bindings: Mapping[str, KeyBinding] | None = None,
) -> InputRouter:
    """Connect a view's input to a control channel."""
    router = InputRouter(channel, key_bindings)
    view.set_event_sink(router)
    return router
