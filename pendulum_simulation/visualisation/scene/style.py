from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    stroke: str | None = "#000000"
    fill: str | None = None
    line_width: float = 1.5
    alpha: float = 1.0
    font_size: float = 10.0
