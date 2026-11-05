from collections.abc import Mapping

from pendulum_simulation.runtime.input_router import InputRouter, KeyBinding
from pendulum_simulation.system.core.control import ControlChannel
from pendulum_simulation.visualisation.backends.base import SceneGui


def wire_input(
    view: SceneGui,
    channel: ControlChannel,
    key_bindings: Mapping[str, KeyBinding] | None = None,
) -> InputRouter:
    """Connect a view's input to a control channel."""
    router = InputRouter(channel, key_bindings)
    view.set_event_sink(router)
    return router
