"""AERE 5460, Homework 3.

Plots the deliverables for Exercise 1. Writes plots to the `figures`
directory relative to this source file.
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

# the alpha shown in the profile and hT figures
ALPHA_EX1 = 0.25
# the alphas compared in the stability figure, one panel each
STABILITY_ALPHAS = (0.6, 0.51, 0.5, 0.49)
# number of early snapshots drawn in each stability panel
N_SHOWN = 6

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


def load_ex1(alpha: float, kind: str) -> Run:
    """Loads one Exercise 1 CSV.

    `kind` is `profiles`, `early`, or `history`, matching the file
    names written by the driver.
    """
    return load(DATA / f"ex1_alpha{alpha:.3g}_{kind}.csv")


def profiles_by(run: Run, prefix: str) -> list[tuple[float, Array]]:
    """Returns the (key, T) profiles of a run, in column order.

    Profile columns are named `<prefix><key>`: `T_t<t>` in the
    profiles CSV and `T_n<step>` in the early CSV.
    """
    return [
        (float(name.removeprefix(prefix)), col)
        for name, col in run.cols.items()
        if name.startswith(prefix)
    ]


def sequential_colors(count: int) -> list[tuple[float, float, float, float]]:
    """Returns `count` colors from a light-to-dark single-hue ramp."""
    cmap = plt.get_cmap("Blues")
    return [cmap(v) for v in np.linspace(0.35, 1.0, count)]


def style(
    ax: Axes, xlabel: str, ylabel: str, title: str, legend: bool = True
) -> None:
    """Applies the common labels, legend, and grid."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)


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


def plot_ex1_profiles() -> None:
    """Plots T(x) at every output time for alpha = ALPHA_EX1."""
    run = load_ex1(ALPHA_EX1, "profiles")
    x = run.cols["x"]
    snaps = profiles_by(run, "T_t")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sequential_colors(len(snaps))
    for (t, T), color in zip(snaps, colors, strict=True):  # noqa: N806
        ax.plot(x, T, "-", color=color, label=rf"$t = {t:.2f}$")
    style(
        ax,
        r"$x$",
        r"$T$",
        rf"FTCS, $\alpha = {ALPHA_EX1:g}$, $N_x = {int(run.meta['Nx'])}$",
    )
    save(fig, "ex1_profiles")


def plot_ex1_stability() -> None:
    """Plots the early profiles near x = 1 for each STABILITY_ALPHAS.

    One panel per alpha, each showing N_SHOWN evenly spaced early
    snapshots. Runs that blew up end at the step where max |T| first
    exceeded the cutoff. Each panel is zoomed to the region the last
    snapshot has disturbed, since the step in the initial condition at
    x = 1 is where any odd-even sawtooth starts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, alpha in zip(axes.flat, STABILITY_ALPHAS, strict=True):
        run = load_ex1(alpha, "early")
        x = run.cols["x"]
        snaps = profiles_by(run, "T_n")
        keep = np.linspace(0, len(snaps) - 1, N_SHOWN).round().astype(int)
        shown = [snaps[k] for k in np.unique(keep)]
        colors = sequential_colors(len(shown))
        for (n, T), color in zip(shown, colors, strict=True):  # noqa: N806
            ax.plot(x, T, ".-", color=color, label=rf"step {n:.0f}")
        disturbed = x[np.abs(shown[-1][1] - shown[0][1]) > 1e-3]
        ax.set_xlim(max(0.0, float(disturbed.min()) - 0.02), 1.0)

        title = rf"$\alpha = {alpha:g}$"
        if int(run.meta["diverged"]) == 1:
            title += rf", $|T| > 10$ at step {int(run.meta['steps'])}"
        style(ax, r"$x$", r"$T$", title)
    save(fig, "ex1_stability")


def plot_ex1_hT() -> None:  # noqa: N802
    """Plots hT vs t for t > 0.01 at alpha = ALPHA_EX1."""
    run = load_ex1(ALPHA_EX1, "history")
    t, hT = run.cols["t"], run.cols["hT"]  # noqa: N806
    mask = t > 0.01

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t[mask], hT[mask], "-", color="#2a78d6")
    style(
        ax,
        r"$t$",
        r"$h_T$",
        rf"bulk heat transfer coefficient, $\alpha = {ALPHA_EX1:g}$",
        legend=False,
    )
    save(fig, "ex1_hT")


def plot_ex1() -> None:
    """Plots the deliverables for Exercise 1."""
    plot_ex1_profiles()
    plot_ex1_stability()
    plot_ex1_hT()


def main():
    plot_ex1()


if __name__ == "__main__":
    main()
