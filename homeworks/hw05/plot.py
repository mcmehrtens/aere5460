"""AERE 5460, Homework 5.

Plots the deliverables for Exercises 1, 2 and 3. Writes plots to the
`figures` directory relative to this source file.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

DATA = Path(__file__).parent / "data"
FIGURES = Path(__file__).parent / "figures"

# shared by every panel so the panels compare directly
EX1_LEVELS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# contour lines in the Exercise 2 and 3 plots
N_LEVELS = 20

EX2_N_ARROWS = 15

# categorical colors, assigned to lambda in ascending order
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

MUTED = "#888888"

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


def to_grid(run: Run, col: Array) -> Array:
    """Reshapes a flat node column to a (Ny, Nx) array.

    The driver writes nodes with x varying fastest (k = j*Nx + i), so
    row j of the result is the line y = y_j.
    """
    return col.reshape(int(run.meta["Ny"]), int(run.meta["Nx"]))


def sequential_colors(count: int) -> list[tuple[float, float, float, float]]:
    """Returns `count` colors from a light-to-dark single-hue ramp."""
    cmap = plt.get_cmap("Blues")
    return [cmap(v) for v in np.linspace(0.35, 1.0, count)]


def save(fig: Figure, name: str) -> None:
    """Writes `fig` to the figures directory and closes it."""
    FIGURES.mkdir(exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(
        FIGURES / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_ex1_contours() -> None:
    """Plots contour lines of psi(x, y), one panel per n and grid.

    Rows are n, columns are grids (coarse to fine).
    """
    runs = sorted(
        (load(p) for p in DATA.glob("ex1_n*_psi.csv")),
        key=lambda r: (float(r.meta["n"]), int(r.meta["Nx"])),
    )
    nrows = len({float(r.meta["n"]) for r in runs})
    ncols = len(runs) // nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols + 1, 5 * nrows),
        squeeze=False,
        layout="constrained",
    )
    cmap = LinearSegmentedColormap.from_list("blues", sequential_colors(256))
    norm = Normalize(EX1_LEVELS[0], EX1_LEVELS[-1])

    for ax, run in zip(axes.flat, runs, strict=True):
        X = to_grid(run, run.cols["x"])  # noqa: N806
        Y = to_grid(run, run.cols["y"])  # noqa: N806
        psi = to_grid(run, run.cols["psi"])
        ax.contour(X, Y, psi, levels=EX1_LEVELS, cmap=cmap, norm=norm)

        title = (
            rf"$n = {float(run.meta['n']):g}$, "
            rf"${int(run.meta['Nx'])} \times {int(run.meta['Ny'])}$, "
            f"{int(run.meta['iterations'])} iterations"
        )
        if run.meta["converged"] != "true":
            title += " (not converged)"
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        ticks=EX1_LEVELS,
        label=r"$\psi$",
        shrink=0.6,
    )
    fig.suptitle(
        rf"Laplace equation, Gauss–Seidel + SOR, "
        rf"$\lambda = {float(runs[0].meta['lambda']):g}$"
    )
    save(fig, "ex1_contours")


def plot_ex1_residual() -> None:
    """Plots log10(eps/eps0) vs iteration for every lambda.

    Also prints the iteration count and final residual drop per lambda.
    """
    runs = sorted(
        (load(p) for p in DATA.glob("ex1_residual_lambda*.csv")),
        key=lambda r: float(r.meta["lambda"]),
    )
    tol = float(runs[0].meta["tol"])
    max_iter = float(runs[0].meta["max_iter"])

    fig, ax = plt.subplots(figsize=(8, 5))
    print(f"{'lambda':>8} {'iterations':>11} {'log10(eps/eps0)':>16}")
    for run, color in zip(runs, SERIES_COLORS, strict=True):
        it = run.cols["iteration"]
        eps = run.cols["eps"]
        log_drop = np.log10(eps / eps[0])
        lam = float(run.meta["lambda"])
        ax.plot(
            it,
            log_drop,
            "-",
            color=color,
            linewidth=2.0,
            label=rf"$\lambda = {lam:g}$",
        )
        print(f"{lam:>8g} {len(it):>11d} {log_drop[-1]:>16.3f}")

    ax.axhline(
        np.log10(tol),
        color=MUTED,
        linestyle="--",
        linewidth=1.0,
        label="tolerance",
    )
    ax.set_xlim(0.0, max_iter)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\log_{10}(\epsilon / \epsilon_0)$")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title(
        rf"SOR residual history, "
        rf"${int(runs[0].meta['Nx'])} \times {int(runs[0].meta['Ny'])}$, "
        rf"$n = {float(runs[0].meta['n']):g}$"
    )
    save(fig, "ex1_residual")


def plot_ex1() -> None:
    """Plots the deliverables for Exercise 1."""
    plot_ex1_contours()
    plot_ex1_residual()


def plot_ex2_grid() -> None:
    """Plots the stretched grid for Exercise 2."""
    run = load(DATA / "ex2_psi.csv")
    x = to_grid(run, run.cols["x"])[0, :]
    y = to_grid(run, run.cols["y"])[:, 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.vlines(x, 0.0, 1.0, color="#2a78d6", linewidth=0.3)
    ax.hlines(y, 0.0, 1.0, color="#2a78d6", linewidth=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(
        rf"Stretched grid, ${int(run.meta['Nx'])} \times "
        rf"{int(run.meta['Ny'])}$"
    )
    save(fig, "ex2_grid")


def psi_contours(run: Run, title: str) -> tuple[Figure, Axes]:
    """Plots N_LEVELS contour lines of psi(x, y) with a colorbar.

    Appends "(not converged)" to `title` if the run did not converge.
    """
    X = to_grid(run, run.cols["x"])  # noqa: N806
    Y = to_grid(run, run.cols["y"])  # noqa: N806
    psi = to_grid(run, run.cols["psi"])
    levels = np.linspace(psi.min(), psi.max(), N_LEVELS + 2)[1:-1]
    cmap = LinearSegmentedColormap.from_list("blues", sequential_colors(256))
    norm = Normalize(float(levels[0]), float(levels[-1]))

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.contour(X, Y, psi, levels=levels, cmap=cmap, norm=norm)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r"$\psi$")

    if run.meta["converged"] != "true":
        title += " (not converged)"
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    return fig, ax


def nearest_nodes(coords: Array) -> npt.NDArray[np.intp]:
    """Returns the indices of the nodes nearest evenly spaced points.

    The points are EX2_N_ARROWS evenly spaced interior points of [0, 1].
    """
    targets = np.linspace(0.0, 1.0, EX2_N_ARROWS + 2)[1:-1]
    return np.abs(coords[:, None] - targets).argmin(axis=0)


def plot_ex2_contours() -> None:
    """Plots contour lines of psi(x, y) for Exercise 2.

    Velocity vectors u = -dpsi/dy, v = dpsi/dx are superimposed on an
    EX2_N_ARROWS x EX2_N_ARROWS set of nodes, each the node nearest an
    evenly spaced point, so the arrows are evenly spaced on the
    stretched grid.
    """
    run = load(DATA / "ex2_psi.csv")
    fig, ax = psi_contours(
        run,
        rf"Poisson equation, Gauss–Seidel + SOR, "
        rf"$\lambda = {float(run.meta['lambda']):g}$",
    )

    X = to_grid(run, run.cols["x"])  # noqa: N806
    Y = to_grid(run, run.cols["y"])  # noqa: N806
    psi = to_grid(run, run.cols["psi"])
    dpsi_dy, dpsi_dx = np.gradient(psi, Y[:, 0], X[0, :])
    u: Array = -np.asarray(dpsi_dy)
    v: Array = np.asarray(dpsi_dx)
    nodes = np.ix_(nearest_nodes(Y[:, 0]), nearest_nodes(X[0, :]))
    ax.quiver(X[nodes], Y[nodes], u[nodes], v[nodes], color=MUTED)
    save(fig, "ex2_contours")


def plot_ex2() -> None:
    """Plots the deliverables for Exercise 2."""
    plot_ex2_grid()
    plot_ex2_contours()


def plot_ex3_contours() -> None:
    """Plots contour lines of psi(x, y) for Exercise 3."""
    run = load(DATA / "ex3_psi.csv")
    fig, _ = psi_contours(
        run,
        rf"Poisson equation, CG + SGS preconditioning, "
        rf"${int(run.meta['Nx'])} \times {int(run.meta['Ny'])}$",
    )
    save(fig, "ex3_contours")


def plot_ex3_residual() -> None:
    """Plots the CG residual history for Exercise 3.

    Also prints the iteration count and final residual.
    """
    run = load(DATA / "ex3_residual.csv")
    it = run.cols["iteration"]
    log_residual = np.log10(run.cols["residual"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        it,
        log_residual,
        "-",
        color=SERIES_COLORS[0],
        linewidth=2.0,
        label="PCG",
    )
    ax.axhline(
        np.log10(float(run.meta["tol"])),
        color=MUTED,
        linestyle="--",
        linewidth=1.0,
        label="tolerance",
    )
    ax.set_xlim(0.0, float(it[-1]))
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\log_{10}(\|\Delta\psi\|_{L2} / \|\Delta\psi\|_{L2}(0))$")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title(
        rf"CG residual history, {len(it)} iterations, "
        rf"${int(run.meta['Nx'])} \times {int(run.meta['Ny'])}$"
    )
    save(fig, "ex3_residual")
    print(f"ex3: {len(it)} iterations, final residual {log_residual[-1]:.3f}")


def plot_ex3() -> None:
    """Plots the deliverables for Exercise 3."""
    plot_ex3_contours()
    plot_ex3_residual()


def main() -> None:
    plot_ex1()
    plot_ex2()
    plot_ex3()


if __name__ == "__main__":
    main()
