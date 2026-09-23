"""AERE 5460, Homework 2.

Plots the deliverables for Exercise 1. Writes
plots to the `figures` directory relative to this source file.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

DATA = Path(__file__).parent / "data"
FIGURES = Path(__file__).parent / "figures"
N = 81
N_EX1E = 161


@dataclass(frozen=True)
class Run:
    """Holds one CSV run."""

    meta: dict[str, float | str]
    x: npt.NDArray[np.float64]
    T: npt.NDArray[np.float64]


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
    # col_header1,col_header2,...
    # data1,data2,...
    # ...
    ```
    """
    with open(path) as f:
        pairs = (p.split("=", 1) for p in f.readline().lstrip("#").split())
        meta = {k: _convert(v) for k, v in pairs}
    x, T = np.loadtxt(path, delimiter=",", skiprows=2).T  # noqa: N806
    return Run(meta=meta, x=x, T=T)


def exact_1b(
    x: npt.NDArray[np.float64],
    M: float,  # noqa: N803
) -> npt.NDArray[np.float64]:
    r"""Compute the exact solution of the heat fin equation.

    The heat fin equation is given as

    .. math::

        \frac{\mathrm{d}^2T}{\mathrm{d}x^2}=MT

    where :math:`T` is temperature and :math:`M` is a sort of thermal
    mass. For initial conditions :math:`T(0)=1` and :math:`T(1)=0`, the
    exact solution is

    .. math::

        T(x)=\frac{\sinh(m(1-x))}{\sinh(m)}

    where :math:`m=\sqrt{M}`.
    """
    m = math.sqrt(M)
    return np.sinh(m * (1 - x)) / np.sinh(m)


def exact_flux_1b(M: float) -> float:  # noqa: N803
    r"""Compute the exact flux at :math:`x=1`.

    Heat flux is given by

    .. math::

        -\frac{\mathrm{d}T}{\mathrm{d}x}.

    For the initial conditions :math:`T(0)=1` and :math:`T(1)=0`, the
    exact heat flux is

    .. math::

        -\frac{\mathrm{d}T}{\mathrm{d}x}=\frac{m\cosh(m(1-x))}{\sinh(m)}

    where :math:`m=\sqrt{M}`. At :math:`x=1`, this simplifies to

    .. math::

        \mathrm{flux}|_{x=1}=\frac{m}{\sinh(m)}.
    """
    m = math.sqrt(M)
    return m / math.sinh(m)


def exact_1c(
    x: npt.NDArray[np.float64],
    M: float,  # noqa: N803
) -> npt.NDArray[np.float64]:
    r"""Compute the exact solution of the heat fin equation.

    The heat fin equation is given as

    .. math::

        \frac{\mathrm{d}^2T}{\mathrm{d}x^2}=MT

    where :math:`T` is temperature and :math:`M` is a sort of thermal
    mass. For initial conditions :math:`T(0)=1` and :math:`T'(1)=0`, the
    exact solution is

    .. math::

        T(x)=\frac{\cosh(m(1-x))}{\cosh(m)}

    where :math:`m=\sqrt{M}`.
    """
    m = math.sqrt(M)
    return np.cosh(m * (1 - x)) / np.cosh(m)


def exact_Tright_1c(M: float) -> float:  # noqa: N802, N803
    r"""Compute the exact value of T at :math:`x=1`.

    For the initial conditions :math:`T(0)=1` and :math:`T'(1)=0`, the
    exact temperature at T(1) is

    .. math::

        T(1)=\frac{1}{\cosh(m)}

    where :math:`m=\sqrt{M}`.
    """
    return 1 / math.cosh(math.sqrt(M))


def load_groups(pattern: str) -> dict[float, list[Run]]:
    """Loads every CSV matching `pattern`, grouped by M.

    Groups are ordered by increasing M, and the runs within each group
    are ordered by increasing N.
    """
    groups: dict[float, list[Run]] = {}
    for run in (load(p) for p in sorted(DATA.glob(pattern))):
        groups.setdefault(float(run.meta["M"]), []).append(run)
    for group in groups.values():
        group.sort(key=lambda r: float(r.meta["N"]))
    return dict(sorted(groups.items()))


def pick(group: list[Run], n_points: int) -> Run:
    """Returns the run in `group` discretized with `n_points` points."""
    return next(r for r in group if int(r.meta["N"]) == n_points)


def print_table(
    groups: dict[float, list[Run]],
    key: str,
    exact_fn: Callable[[float], float],
    title: str,
) -> None:
    """Tabulates a scalar from each run against its closed-form value.

    `key` names the scalar in the run metadata, and `exact_fn` returns
    its exact value for a given M. For each M, prints the scalar over
    the grid refinement along with the relative error and the ratio of
    consecutive errors. A second-order method drives that ratio toward
    4 as N doubles.
    """
    for M, group in groups.items():  # noqa: N806
        exact = exact_fn(M)
        print(f"=== {title}, M = {M:g} ===")
        print(f"exact = {exact:.9f}")
        print(f"{'N':>5}  {key:>13}  {'rel err':>10}  {'ratio':>6}")
        prev: float | None = None
        for run in group:
            value = float(run.meta[key])
            err = abs(value - exact)
            ratio = prev / err if prev is not None and err > 0.0 else math.nan
            print(
                f"{int(run.meta['N']):5d}  {value:13.9f}  "
                f"{err / exact * 100:9.4f}%  {ratio:6.2f}"
            )
            prev = err
        print()


def plot_ex1b() -> None:
    """Plots the deliverables for Exercise 1b.

    Tabulates the heat flux at x = 1 for M = 1, 5, 9, then writes a
    single figure holding one T(x) curve per M at N grid points, with
    the closed-form solutions overlaid.
    """
    FIGURES.mkdir(exist_ok=True)
    groups = load_groups("ex1b_*.csv")
    print_table(groups, "flux", exact_flux_1b, "EXERCISE 1b FLUX")

    x_fine = np.linspace(0.0, 1.0, 1000)
    fig, ax = plt.subplots(figsize=(8, 5))
    for M, group in groups.items():  # noqa: N806
        run = pick(group, N)
        ax.plot(run.x, run.T, "-", label=rf"$M = {M:g}$")
    for i, M in enumerate(groups):  # noqa: N806
        ax.plot(
            x_fine,
            exact_1b(x_fine, M),
            "k--",
            linewidth=1,
            label="exact" if i == 0 else "_nolegend_",
        )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$T$")
    ax.set_title(rf"heat fin, $T(0)=1$, $T(1)=0$, $N={N}$")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "ex1b.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_ex1c() -> None:
    """Prints the deliverables for Exercise 1c.

    Tabulates the temperature at the insulated end, x = 1, against the
    closed-form values for M = 1, 5, 9. Exercise 1c asks for no figure
    of its own; its curves appear in the Exercise 1d comparison.
    """
    groups = load_groups("ex1c_*.csv")
    print_table(groups, "Tright", exact_Tright_1c, "EXERCISE 1c T(1)")


def plot_ex1d() -> None:
    """Plots the deliverables for Exercise 1d.

    Writes a single figure comparing T(x) at M = 9 under the two
    boundary conditions at x = 1: the cold reservoir, T(1) = 0, and the
    insulated end, dT(1)/dx = 0.
    """
    FIGURES.mkdir(exist_ok=True)
    M = 9.0  # noqa: N806
    cold = pick(load_groups("ex1b_*.csv")[M], N)
    insulated = pick(load_groups("ex1c_*.csv")[M], N)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cold.x, cold.T, "-", label=r"$T(1)=0$")
    ax.plot(
        insulated.x,
        insulated.T,
        "-",
        label=r"$\mathrm{d}T(1)/\mathrm{d}x=0$",
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$T$")
    ax.set_title(rf"heat fin, $M={M:g}$, $N={N}$")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "ex1d.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def print_convergence_table(
    groups: dict[float, list[Run]],
    key: str,
    title: str,
) -> None:
    """Tabulates a scalar that has no closed form to compare against.

    Accuracy is judged by how much the scalar still moves as the grid
    refines. Prints the change between consecutive grids and the ratio
    of consecutive changes, which a second-order method drives toward
    4. The reference value is a Richardson extrapolation of the two
    finest grids, which cancels the leading error term; the last row
    is therefore not an independent check of itself.
    """
    for M, group in groups.items():  # noqa: N806
        values = [(int(run.meta["N"]), float(run.meta[key])) for run in group]
        best = (4.0 * values[-1][1] - values[-2][1]) / 3.0
        print(f"=== {title}, M = {M:g} ===")
        print(f"richardson estimate = {best:.9f}")
        print(
            f"{'N':>5}  {key:>13}  {'change':>10}  "
            f"{'ratio':>6}  {'rel err':>10}"
        )
        prev_value: float | None = None
        prev_change: float | None = None
        for n_points, value in values:
            rel = abs(value - best) / abs(best) * 100.0
            if prev_value is None:
                print(
                    f"{n_points:5d}  {value:13.9f}  {'':>10}  "
                    f"{'':>6}  {rel:9.4f}%"
                )
            else:
                change = abs(value - prev_value)
                if prev_change is not None and change > 0.0:
                    ratio = f"{prev_change / change:6.2f}"
                else:
                    ratio = f"{'':>6}"
                print(
                    f"{n_points:5d}  {value:13.9f}  {change:10.2e}  "
                    f"{ratio}  {rel:9.4f}%"
                )
                prev_change = change
            prev_value = value
        print()


def plot_ex1e() -> None:
    """Plots the deliverables for Exercise 1e.

    The thermal mass now varies with position, so the equation has no
    elementary closed form. Accuracy is established by self-convergence
    instead. Writes a single figure holding one T(x) curve per M.
    """
    FIGURES.mkdir(exist_ok=True)
    groups = load_groups("ex1e_*.csv")
    print_convergence_table(groups, "Tright", "EXERCISE 1e T(1)")

    fig, ax = plt.subplots(figsize=(8, 5))
    for M, group in groups.items():  # noqa: N806
        run = pick(group, N_EX1E)
        ax.plot(run.x, run.T, "-", label=rf"$M = {M:g}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$T$")
    ax.set_title(rf"heat fin, $M\cdot(1+20x^2)$, $N={N_EX1E}$")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "ex1e.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_ex1f() -> None:
    """Plots the deliverables for Exercise 1f.

    A distributed heat source makes the equation non-homogeneous. The
    source enters the right-hand side and the matrix is unchanged from
    Exercise 1c, so the solver is untouched. The exact solution is not
    required, so accuracy is again established by self-convergence.
    Writes a single figure holding one T(x) curve per M.
    """
    FIGURES.mkdir(exist_ok=True)
    groups = load_groups("ex1f_*.csv")
    print_convergence_table(groups, "Tright", "EXERCISE 1f T(1)")

    fig, ax = plt.subplots(figsize=(8, 5))
    for M, group in groups.items():  # noqa: N806
        run = pick(group, N)
        ax.plot(run.x, run.T, "-", label=rf"$M = {M:g}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$T$")
    ax.set_title(rf"heat fin, source $-100x^2(1-x)^2$, $N={N}$")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "ex1f.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main():
    plot_ex1b()
    plot_ex1c()
    plot_ex1d()
    plot_ex1e()
    plot_ex1f()


if __name__ == "__main__":
    main()
