"""AERE 5460, Homework 4.

Plots the deliverables for Exercises 1, 2 and 3. Writes plots to the
`figures` directory relative to this source file.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DATA = Path(__file__).parent / "data"
FIGURES = Path(__file__).parent / "figures"

# number of contour lines drawn in each panel
N_LEVELS = 9

# at most this many panels side by side; more snapshots wrap to new rows
MAX_COLS = 3

# Exercise 3 contour levels: T falls off roughly exponentially away from
# the lower wall, so geometric levels are spread evenly in space
EX3_LEVELS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.7)

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


def panel_grid(count: int) -> tuple[Figure, list[Axes]]:
    """Makes a figure with `count` equal panels, at most MAX_COLS wide.

    Panels fill rows left to right. A partly filled last row is
    centered: the grid has two half-columns per panel, so the last row
    can be shifted by half a panel width per missing panel.
    """
    ncols = min(count, MAX_COLS)
    nrows = -(-count // ncols)  # ceiling division
    fig = plt.figure(figsize=(5 * ncols, 5.4 * nrows))
    grid = fig.add_gridspec(nrows, 2 * ncols)
    axes: list[Axes] = []
    for k in range(count):
        row, col = divmod(k, ncols)
        in_row = min(ncols, count - row * ncols)
        start = 2 * col + (ncols - in_row)
        axes.append(fig.add_subplot(grid[row, start : start + 2]))
    return fig, axes


def contour_panels(
    run: Run,
    velocity: tuple[float, float] | None = None,
    from_min: bool = False,
    levels: tuple[float, ...] | None = None,
) -> Figure:
    """Plots contour lines of T(x, y) at every output time of a run.

    One panel per snapshot. The peak drops by more than an order of
    magnitude between panels, so each panel gets its own N_LEVELS
    evenly spaced levels between 0 and its own max T (both endpoints
    excluded: 0 is the far field, max T is a single point). The panel
    title gives the max so the panels can still be compared.

    If `from_min` is set, the levels start at the panel's own min T
    instead of 0, and the title gives the min too. Use this when the
    field never gets near 0, otherwise most levels fall below the data.

    If `levels` is given, every panel uses those levels instead.

    If `velocity` = (U, V) is given, each panel also marks where a
    point starting at the hot spot center (0.25, 0.25) would be carried
    by the flow, as long as it is still inside the domain.
    """
    X = to_grid(run, run.cols["x"])  # noqa: N806
    Y = to_grid(run, run.cols["y"])  # noqa: N806
    snaps = profiles_by(run, "T_t")

    fig, axes = panel_grid(len(snaps))
    for ax, (t, T_flat) in zip(axes, snaps, strict=True):  # noqa: N806
        T = to_grid(run, T_flat)  # noqa: N806
        T_max = float(T.max())  # noqa: N806
        T_min = float(T.min()) if from_min else 0.0  # noqa: N806
        panel_levels = (
            np.linspace(T_min, T_max, N_LEVELS + 2)[1:-1]
            if levels is None
            else np.asarray(levels)
        )
        cs = ax.contour(
            X,
            Y,
            T,
            levels=panel_levels,
            colors=sequential_colors(len(panel_levels)),
        )
        ax.clabel(cs, fmt="%.2g", fontsize=7, colors="#333333")
        if velocity is not None:
            xc = 0.25 + velocity[0] * t
            yc = 0.25 + velocity[1] * t
            if xc <= 1.0 and yc <= 1.0:
                ax.plot(xc, yc, "x", color="#c0392b", markersize=8)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        title = rf"$t = {t:g}$, $\max T = {T_max:.3g}$"
        if from_min:
            title += rf", $\min T = {T_min:.3g}$"
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5)
    return fig


def plot_ex1_contours() -> None:
    """Plots contour lines of T(x, y) for Exercise 1."""
    run = load(DATA / "ex1_profiles.csv")
    fig = contour_panels(run)
    fig.suptitle(
        rf"2-D diffusion, AB2 + central differences, "
        rf"$N_x = N_y = {int(run.meta['Nx'])}$, "
        rf"$\Delta t = {sci(float(run.meta['dt']))}$"
    )
    save(fig, "ex1_contours")


def plot_ex1() -> None:
    """Plots the deliverables for Exercise 1."""
    plot_ex1_contours()


def plot_ex2_contours() -> None:
    """Plots contour lines of T(x, y) for Exercise 2.

    The red cross marks the hot spot center carried at (U, V); the
    contours should be centered on it until the spot reaches the
    outflow walls. Warm fluid keeps entering through the insulated
    inflow walls, so T never gets near 0; the levels span each panel's
    own min to max.
    """
    run = load(DATA / "ex2_profiles.csv")
    U, V = float(run.meta["U"]), float(run.meta["V"])  # noqa: N806
    fig = contour_panels(run, velocity=(U, V), from_min=True)
    fig.suptitle(
        rf"2-D convection-diffusion, FV + RK2 (midpoint), "
        rf"$U = V = {U:g}$, "
        rf"$N_x = N_y = {int(run.meta['Nx'])}$, "
        rf"$\Delta t = {sci(float(run.meta['dt']))}$"
    )
    save(fig, "ex2_contours")


def plot_ex2() -> None:
    """Plots the deliverables for Exercise 2."""
    plot_ex2_contours()


def plot_ex3_contours() -> None:
    """Plots contour lines of T(x, y) for Exercise 3.

    Uses the fixed geometric EX3_LEVELS in every panel, so the panels
    compare directly.
    """
    run = load(DATA / "ex3_profiles.csv")
    fig = contour_panels(run, levels=EX3_LEVELS)
    fig.suptitle(
        rf"2-D diffusion, ADI (Crank–Nicolson), "
        rf"$N_x = {int(run.meta['Nx'])}$, $N_y = {int(run.meta['Ny'])}$, "
        rf"$\Delta t = {sci(float(run.meta['dt']))}$, "
        rf"$r_y = {float(run.meta['ry']):.3g}$"
    )
    save(fig, "ex3_contours")


def plot_ex3_center() -> None:
    """Plots T(1/2, 1/2) vs t for Exercise 3.

    The top panel shows T itself; the bottom panel shows |dT/dt| on a
    log axis, where an exponential approach to steady state is a
    straight line.
    """
    run = load(DATA / "ex3_history.csv")
    t = run.cols["t"]
    T = run.cols["T_center"]  # noqa: N806
    rate = np.abs(np.gradient(T, t))

    fig, (ax_t, ax_r) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_t.plot(t, T, "-", color="#2a78d6")
    ax_t.set_ylabel(r"$T(1/2,\,1/2)$")
    ax_t.set_title(
        rf"$T(1/2,\,1/2,\,t = {t[-1]:g}) = {T[-1]:.6f}$, "
        rf"$\Delta t = {sci(float(run.meta['dt']))}$"
    )
    ax_r.semilogy(t, rate, "-", color="#2a78d6")
    ax_r.set_ylim(rate.max() * 1e-4, rate.max() * 2.0)
    ax_r.set_xlabel(r"$t$")
    ax_r.set_ylabel(r"$|\partial T / \partial t|$ at $(1/2,\,1/2)$")
    for ax in (ax_t, ax_r):
        ax.set_xlim(0.0, float(t[-1]))
        ax.grid(True, linestyle=":", alpha=0.5)
    save(fig, "ex3_center")


def plot_ex3() -> None:
    """Plots the deliverables for Exercise 3."""
    plot_ex3_contours()
    plot_ex3_center()


def main() -> None:
    plot_ex1()
    plot_ex2()
    plot_ex3()


if __name__ == "__main__":
    main()
