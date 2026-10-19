"""AERE 5460, Homework 4.

Plots the deliverables for Exercise 1. Writes plots to the `figures`
directory relative to this source file.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

DATA = Path(__file__).parent / "data"
FIGURES = Path(__file__).parent / "figures"

# number of contour lines drawn in each panel
N_LEVELS = 9

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Run:
    """Holds one CSV run."""

    meta: dict[str, float | str]
    cols: dict[str, Array]


def _convert(value: str) -> float | str:
    """Converts to float, if possible."""
    try:
        return float(value)
    except ValueError:
        return value


def load(path: Path) -> Run:
    """Loads a data CSV.

    Expects the following CSV format:
    ```
    # key1=value1 key2=value2 ...
    col_header1, col_header2, ...
    data1, data2, ...
    ...
    ```
    Returns the metadata and a dict of columns keyed by header.
    """
    with open(path) as f:
        pairs = (p.split("=", 1) for p in f.readline().lstrip("#").split())
        meta = {k: _convert(v) for k, v in pairs}
        names = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=2, ndmin=2)
    return Run(meta=meta, cols=dict(zip(names, data.T, strict=True)))


def profiles_by(run: Run, prefix: str) -> list[tuple[float, Array]]:
    """Returns the (key, T) profiles of a run, in column order.

    Profile columns are named `<prefix><key>`, e.g. `T_t0.05`.
    """
    return [
        (float(name.removeprefix(prefix)), col)
        for name, col in run.cols.items()
        if name.startswith(prefix)
    ]


def to_grid(run: Run, col: Array) -> Array:
    """Reshapes a flat node column to a (Ny, Nx) array.

    The driver writes nodes with x varying fastest (k = j*Nx + i), so
    row j of the result is the line y = y_j.
    """
    return col.reshape(int(run.meta["Ny"]), int(run.meta["Nx"]))


def sci(value: float, digits: int = 4) -> str:
    r"""Formats `value` as mathtext `m \times 10^{e}`.

    Returns the inner math only (no `$`), e.g. `4.883 \times 10^{-6}`.
    """
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def sequential_colors(count: int) -> list[tuple[float, float, float, float]]:
    """Returns `count` colors from a light-to-dark single-hue ramp."""
    cmap = plt.get_cmap("Blues")
    return [cmap(v) for v in np.linspace(0.35, 1.0, count)]


def save(fig: Figure, name: str) -> None:
    """Writes `fig` to the figures directory and closes it."""
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_ex1_contours() -> None:
    """Plots contour lines of T(x, y) at every output time.

    One panel per snapshot. The peak drops by more than an order of
    magnitude between panels, so each panel gets its own N_LEVELS
    evenly spaced levels between 0 and its own max T (both endpoints
    excluded: 0 is the wall, max T is a single point). The panel title
    gives the max so the panels can still be compared.
    """
    run = load(DATA / "ex1_profiles.csv")
    X = to_grid(run, run.cols["x"])  # noqa: N806
    Y = to_grid(run, run.cols["y"])  # noqa: N806
    snaps = profiles_by(run, "T_t")

    fig, axes = plt.subplots(
        1, len(snaps), figsize=(5 * len(snaps), 5.4), squeeze=False
    )
    for ax, (t, T_flat) in zip(axes.flat, snaps, strict=True):  # noqa: N806
        T = to_grid(run, T_flat)  # noqa: N806
        T_max = float(T.max())  # noqa: N806
        levels = np.linspace(0.0, T_max, N_LEVELS + 2)[1:-1]
        cs = ax.contour(
            X, Y, T, levels=levels, colors=sequential_colors(N_LEVELS)
        )
        ax.clabel(cs, fmt="%.2g", fontsize=7, colors="#333333")
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(rf"$t = {t:g}$, $\max T = {T_max:.3g}$")
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        rf"2-D diffusion, AB2 + central differences, "
        rf"$N_x = N_y = {int(run.meta['Nx'])}$, "
        rf"$\Delta t = {sci(float(run.meta['dt']))}$"
    )
    save(fig, "ex1_contours")


def plot_ex1() -> None:
    """Plots the deliverables for Exercise 1."""
    plot_ex1_contours()


def main() -> None:
    plot_ex1()


if __name__ == "__main__":
    main()
