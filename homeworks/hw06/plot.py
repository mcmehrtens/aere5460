"""AERE 5460, Homework 6.

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
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

DATA = Path(__file__).parent / "data"
FIGURES = Path(__file__).parent / "figures"

# contour lines in the psi plots
N_LEVELS = 20

# draw a velocity vector at every ARROW_STRIDE-th node
ARROW_STRIDE = 10

# Exercise 3 vortex patches as (x0, y0, width, height)
EX3_VORTICES = ((0.0, -0.13, 0.3, 0.13), (0.0, 0.0, 0.3, 0.13))

# categorical colors, assigned in this order
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

MUTED = "#888888"

# fill for i-blanked (solid) regions
SOLID = "#d9d9d9"

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

    The driver writes nodes with x varying fastest (index k*Nx + j), so
    row k of the result is the line y = y_k.
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


def velocity(run: Run) -> tuple[Array, Array]:
    """Returns u = -dpsi/dy and v = dpsi/dx on the (Ny, Nx) grid.

    Central differences in the interior, one-sided at the edges. Fluid
    nodes next to a blanked region difference against the wall value
    held there, which is the boundary condition.
    """
    X = to_grid(run, run.cols["x"])  # noqa: N806
    Y = to_grid(run, run.cols["y"])  # noqa: N806
    psi = to_grid(run, run.cols["psi"])
    dpsi_dy, dpsi_dx = np.gradient(psi, Y[:, 0], X[0, :])
    return -np.asarray(dpsi_dy), np.asarray(dpsi_dx)


def draw_solid(ax: Axes, run: Run) -> None:
    """Fills the bounding box of the i-blanked nodes in gray.

    Exact for a rectangular blanked region, such as the Exercise 1
    step. Sides on the domain edge are pushed past it (the axes clip
    the overhang), so the region still fills its corner when a plot
    pads the axes beyond the domain.
    """
    blanked = run.cols["iblank"] == 0.0
    x = run.cols["x"][blanked]
    y = run.cols["y"][blanked]
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    overhang = 1.0
    if x0 == run.cols["x"].min():
        x0 -= overhang
    if x1 == run.cols["x"].max():
        x1 += overhang
    if y0 == run.cols["y"].min():
        y0 -= overhang
    if y1 == run.cols["y"].max():
        y1 += overhang
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=SOLID,
            edgecolor=MUTED,
            linewidth=1.0,
            zorder=3,
        )
    )


def draw_blanked(ax: Axes, run: Run) -> None:
    """Fills one grid cell per i-blanked node in gray.

    Shows the stair-step shape the solver actually sees.
    """
    X = to_grid(run, run.cols["x"])  # noqa: N806
    Y = to_grid(run, run.cols["y"])  # noqa: N806
    fluid = to_grid(run, run.cols["iblank"]) != 0.0
    ax.pcolormesh(
        X,
        Y,
        np.ma.masked_where(fluid, np.ones_like(X)),
        cmap=ListedColormap([SOLID]),
        shading="nearest",
        zorder=3,
    )


def style_domain(ax: Axes, run: Run, title: str) -> None:
    """Sets equal axes over the grid extent, labels and `title`.

    Appends "(not converged)" to `title` if the run did not converge.
    """
    if run.meta["converged"] != "true":
        title += " (not converged)"
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(float(run.cols["x"].min()), float(run.cols["x"].max()))
    ax.set_ylim(float(run.cols["y"].min()), float(run.cols["y"].max()))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")


def psi_contours(run: Run, title: str) -> tuple[Figure, Axes]:
    """Plots N_LEVELS contour lines of psi(x, y) with a colorbar.

    Blanked nodes are masked out; the caller draws the solid region.
    """
    X = to_grid(run, run.cols["x"])  # noqa: N806
    Y = to_grid(run, run.cols["y"])  # noqa: N806
    psi = to_grid(run, run.cols["psi"])
    fluid = to_grid(run, run.cols["iblank"]) != 0.0
    # interior levels only: the extremes are the walls themselves
    levels = np.linspace(psi.min(), psi.max(), N_LEVELS + 2)[1:-1]
    cmap = LinearSegmentedColormap.from_list("blues", sequential_colors(256))
    norm = Normalize(float(levels[0]), float(levels[-1]))

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.contour(
        X,
        Y,
        np.ma.masked_where(~fluid, psi),
        levels=levels,
        cmap=cmap,
        norm=norm,
    )
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label=r"$\Psi$",
        shrink=0.8,
    )
    style_domain(ax, run, title)
    return fig, ax


def plot_ex1_contours() -> None:
    """Plots contour lines of psi(x, y) for Exercise 1."""
    run = load(DATA / "ex1_psi.csv")
    fig, ax = psi_contours(
        run,
        rf"Contraction, Gauss–Seidel + SOR, "
        rf"${int(run.meta['Nx'])} \times {int(run.meta['Ny'])}$, "
        rf"$\lambda = {float(run.meta['lambda']):g}$, "
        f"{int(run.meta['iterations'])} iterations",
    )
    draw_solid(ax, run)
    save(fig, "ex1_contours")


def velocity_vectors(run: Run, title: str) -> tuple[Figure, Axes]:
    """Plots velocity vectors (u, v) at every ARROW_STRIDE-th node.

    Vectors at blanked nodes are omitted; the caller draws the solid
    region.
    """
    X = to_grid(run, run.cols["x"])  # noqa: N806
    Y = to_grid(run, run.cols["y"])  # noqa: N806
    fluid = to_grid(run, run.cols["iblank"]) != 0.0
    u, v = velocity(run)

    s = np.s_[::ARROW_STRIDE, ::ARROW_STRIDE]
    keep = fluid[s]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.quiver(
        X[s][keep],
        Y[s][keep],
        u[s][keep],
        v[s][keep],
        color=SERIES_COLORS[0],
        zorder=2,
    )
    style_domain(ax, run, title)
    # pad so the arrows on the domain edges are not cut off
    ax.set_xlim(float(X.min()) - 0.05, float(X.max()) + 0.1)
    ax.set_ylim(float(Y.min()) - 0.05, float(Y.max()) + 0.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    return fig, ax


def plot_ex1_velocity() -> None:
    """Plots velocity vectors (u, v) for Exercise 1."""
    run = load(DATA / "ex1_psi.csv")
    fig, ax = velocity_vectors(
        run,
        rf"Contraction, velocity $(u, v)$, every {ARROW_STRIDE}th node",
    )
    draw_solid(ax, run)
    save(fig, "ex1_velocity")

    # mass check: u at the inlet and outlet, away from the corners
    fluid = to_grid(run, run.cols["iblank"]) != 0.0
    u, _ = velocity(run)
    print(
        f"ex1: {int(run.meta['iterations'])} iterations, "
        f"u(inlet) mean = {u[1:-1, 0].mean():.4f} (expect 1), "
        f"u(outlet, fluid) mean = "
        f"{u[1:-1, -1][fluid[1:-1, -1]].mean():.4f} (expect 4/3)"
    )


def plot_ex1() -> None:
    """Plots the deliverables for Exercise 1."""
    plot_ex1_contours()
    plot_ex1_velocity()


def plot_ex2() -> None:
    """Plots contour lines of psi(x, y) for every Exercise 2 run.

    Also prints the iterations and CPU time of each run.
    """
    solver_names = {"sor": "SOR", "cg": "PCG"}
    print(
        f"{'part':>4} {'solver':>6} {'alpha':>5} {'psi_cyl':>7} "
        f"{'tol':>7} {'iterations':>10} {'cpu_time [s]':>12}"
    )
    for path in sorted(DATA.glob("ex2_*_psi.csv")):
        run = load(path)
        # from the file name: meta would turn part "2" into 2.0
        _, part, solver, _ = path.stem.split("_")
        alpha = float(np.degrees(float(run.meta["alpha"])))
        psi_cyl = float(run.meta["psi_cyl"])
        iterations = int(run.meta["iterations"])

        fig, ax = psi_contours(
            run,
            rf"Part {part}, {solver_names[solver]}, "
            rf"$\alpha = {alpha:g}^\circ$, "
            rf"$\Psi_{{\mathrm{{cyl}}}} = {psi_cyl:g}$, "
            f"{iterations} iterations",
        )
        draw_blanked(ax, run)
        save(fig, f"ex2_{part}_{solver}_contours")

        print(
            f"{part:>4} {solver:>6} {alpha:>5g} {psi_cyl:>7g} "
            f"{float(run.meta['tol']):>7g} {iterations:>10d} "
            f"{float(run.meta['cpu_time']):>12.3f}"
        )


def draw_vortices(ax: Axes) -> None:
    """Outlines the Exercise 3 vortex patches with dashed lines."""
    for x0, y0, width, height in EX3_VORTICES:
        ax.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=MUTED,
                linestyle="--",
                linewidth=1.0,
                zorder=3,
            )
        )


def plot_ex3() -> None:
    """Plots contour lines of psi(x, y) and velocity vectors (u, v).

    Also prints how far psi is from being odd in y. The exact solution
    is, since the boundary values and the vorticity are; what remains is
    the SOR error at the stopping tolerance.
    """
    run = load(DATA / "ex3_psi.csv")
    fig, ax = psi_contours(
        run,
        rf"Square with vortices, Gauss–Seidel + SOR, "
        rf"${int(run.meta['Nx'])} \times {int(run.meta['Ny'])}$, "
        rf"$\lambda = {float(run.meta['lambda']):g}$, "
        f"{int(run.meta['iterations'])} iterations",
    )
    draw_blanked(ax, run)
    draw_vortices(ax)
    save(fig, "ex3_contours")

    fig, ax = velocity_vectors(
        run,
        rf"Square with vortices, velocity $(u, v)$, "
        f"every {ARROW_STRIDE}th node",
    )
    draw_blanked(ax, run)
    draw_vortices(ax)
    save(fig, "ex3_velocity")

    psi = to_grid(run, run.cols["psi"])
    print(
        f"ex3: {int(run.meta['iterations'])} iterations, "
        f"max |psi(x, y) + psi(x, -y)| = "
        f"{np.abs(psi + psi[::-1, :]).max():.1e} (0 when fully converged)"
    )


def main() -> None:
    plot_ex1()
    plot_ex2()
    plot_ex3()


if __name__ == "__main__":
    main()
