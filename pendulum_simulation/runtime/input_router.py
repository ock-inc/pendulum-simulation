from collections.abc import Mapping
from dataclasses import dataclass

from ..core.control import ControlChannel
from ..visualisation.ui_events import InputEvent, KeyEvent, ValueChangedEvent


@dataclass(frozen=True)
class KeyBinding:
    channel: str
    value: float  # value the channel takes while the key is held


class InputRouter:
    """Implements InputSink: turns generic UI events into ControlChannel writes."""

    def __init__(
        self,
        channel: ControlChannel,
        key_bindings: Mapping[str, KeyBinding] | None = None,
    ):
        self.channel = channel
        self.key_bindings = dict(key_bindings or {})
        self._held: dict[str, list[str]] = {}  # channel name -> held keys, oldest first

    def handle(self, event: InputEvent) -> None:
        if isinstance(event, ValueChangedEvent):
            self.channel.set(event.name, event.value)
        elif isinstance(event, KeyEvent):
            self._handle_key(event)

    def _handle_key(self, event: KeyEvent) -> None:
        binding = self.key_bindings.get(event.key)
        if binding is None:
            return

        held = self._held.setdefault(binding.channel, [])

        if event.pressed:
            if event.key not in held:
                held.append(event.key)
        elif event.key in held:
            held.remove(event.key)

        if held:  # most recently pressed key still held wins (e.g. left+right)
            self.channel.set(binding.channel, self.key_bindings[held[-1]].value)
        else:
            self.channel.release(binding.channel)
