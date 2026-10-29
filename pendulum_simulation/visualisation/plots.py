from __future__ import annotations

from collections.abc import Sequence

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "Plotting needs matplotlib. Install it with: "
        "pip install 'pendulum-simulation[viz]'"
    ) from e

from ..core.runner import Trajectory


def _prepare(
    trajectories: Trajectory | Sequence[Trajectory],
    labels: Sequence[str] | None,
) -> tuple[list[Trajectory], list[str | None]]:
    """Normalise the inputs to a list of trajectories and a matching list of labels."""
    trajs = (
        [trajectories] if isinstance(trajectories, Trajectory) else list(trajectories)
    )
    if not trajs:
        raise ValueError("No trajectories to plot")

    names = trajs[0].field_names
    if any(t.field_names != names for t in trajs):
        raise ValueError("All trajectories must have the same state fields")

    out: list[str | None]
    if labels is None:
        out: list[str | None] = (
            [None] if len(trajs) == 1 else [f"run {i + 1}" for i in range(len(trajs))]
        )
    else:
        out = list(labels)

    if len(out) != len(trajs):
        raise ValueError(f"Got {len(out)} labels for {len(trajs)} trajectories")

    return trajs, out


def plot_states(
    trajectories: Trajectory | Sequence[Trajectory],
    labels: Sequence[str] | None = None,
) -> Figure:
    """One subplot per state field against time, with all runs overlaid.

    Args:
        trajectories: A single trajectory, or several to compare.
        labels: A legend label per trajectory. Defaults to "run 1", "run 2", ...
            when there are several, and no legend for a single run.
    """
    trajs, labels_ = _prepare(trajectories, labels)
    names = trajs[0].field_names

    fig, axes = plt.subplots(
        len(names), 1, sharex=True, squeeze=False, figsize=(8, 2.2 * len(names) + 0.8)
    )

    for ax, name in zip(axes[:, 0], names, strict=True):
        for traj, label in zip(trajs, labels_, strict=True):
            ax.plot(traj.times, traj.field(name), label=label)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("time")

    if any(label is not None for label in labels_):
        axes[0, 0].legend()

    fig.tight_layout()
    return fig


def plot_phase(
    trajectories: Trajectory | Sequence[Trajectory],
    x_field: str,
    y_field: str,
    labels: Sequence[str] | None = None,
) -> Figure:
    """Phase portrait: one state field against another, with all runs overlaid.

    The start of each run is marked with a dot. For an undamped pendulum with
    an accurate solver this traces a closed loop; a drifting solver spirals.
    """
    trajs, labels_ = _prepare(trajectories, labels)

    fig, ax = plt.subplots(figsize=(6, 5))

    for traj, label in zip(trajs, labels_, strict=True):
        x, y = traj.field(x_field), traj.field(y_field)
        (line,) = ax.plot(x, y, label=label)
        ax.plot(x[0], y[0], "o", color=line.get_color())

    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.grid(True, alpha=0.3)

    if any(label is not None for label in labels_):
        ax.legend()

    fig.tight_layout()
    return fig
