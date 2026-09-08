"""AERE 5460, Homework 1.

Plots the deliverables for Exercises 2–3. Writes
plots to the `figures` directory relative to this source file.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

DATA = Path(__file__).parent / "data"
FIGURES = Path(__file__).parent / "figures"


@dataclass(frozen=True)
class Run:
    """Holds one CSV run."""

    meta: dict[str, float | str]
    t: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    ydot: npt.NDArray[np.float64]


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
    t, y, ydot = np.loadtxt(path, delimiter=",", skiprows=2).T
    return Run(meta=meta, t=t, y=y, ydot=ydot)


def exact(
    t: npt.NDArray[np.float64], sigma: float, omega: float
) -> npt.NDArray[np.float64]:
    r"""Compute the exact solution for an underdamped oscillator.

    The 2nd-order ODE for an underdamped oscillator is given as Y'' +
    𝜎Y' + 𝜔²Y = 0. This function, given values for 𝜎 and 𝜔, returns the
    exact solution

    .. math::

        Y(t)=e^{-\sigma t/2}
        \left(
            \cos(\omega_dt)
            + \frac{\sigma}{2\omega_d}\sin(\omega_dt)
        \right)

    where :math:`\omega_d=\sqrt{\omega^2-\frac{\sigma^2}{4}}` given
    initial conditions Y(0) = 1, Ẏ(0) = 0.
    """
    omega_d = np.sqrt(omega**2 - sigma**2 / 4)
    return np.exp(-sigma * t / 2) * (
        np.cos(omega_d * t) + sigma / (2 * omega_d) * np.sin(omega_d * t)
    )


def plot_ex2a():
    FIGURES.mkdir(exist_ok=True)
    runs = [load(p) for p in sorted(DATA.glob("ex2a_*.csv"))]
    groups: dict[float, list[Run]] = {}
    for run in runs:
        groups.setdefault(float(run.meta["sigma"]), []).append(run)
    for group in groups.values():
        group.sort(key=lambda r: r.meta["N"])
    for sigma, group in groups.items():
        omega = float(group[0].meta["omega"])
        fig, (ax_y, ax_err) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        t_fine = np.linspace(
            float(group[0].meta["t0"]), float(group[0].meta["t1"]), 1000
        )
        for run in group:
            ax_y.plot(run.t, run.y, "-", label=f"N = {int(run.meta['N'])}")
            ax_err.semilogy(
                run.t,
                np.abs(run.y - exact(run.t, sigma, omega)),
                "-",
                label=f"N = {int(run.meta['N'])}",
            )
        ax_y.plot(
            t_fine,
            exact(t_fine, sigma, omega),
            "k--",
            label="exact",
        )
        ax_err.set_yscale("log", nonpositive="mask")
        ax_y.set_ylim(-2, 2)
        ax_y.set_ylabel(r"$Y$")
        ax_err.set_ylabel(r"$|Y-Y_\mathrm{exact}|$")
        ax_err.set_xlabel(r"$t$")
        ax_y.set_title(
            rf"damped oscillator, $\sigma={sigma}$, $\omega={omega}$"
        )
        ax_y.legend(loc="lower right")
        ax_err.legend()
        ax_y.grid(True, linestyle=":", alpha=0.5)
        ax_err.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"ex2a_rk2_sigma{sigma}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def plot_ex2b():
    FIGURES.mkdir(exist_ok=True)
    runs = [load(p) for p in sorted(DATA.glob("ex2b_*.csv"))]
    groups: dict[float, list[Run]] = {}
    for run in runs:
        groups.setdefault(float(run.meta["B"]), []).append(run)
    for group in groups.values():
        group.sort(key=lambda r: r.meta["N"])
    for B, group in groups.items():  # noqa: N806
        ref = group[-1]
        fig, (ax_y, ax_err) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        for run in group[:-1]:
            ax_y.plot(run.t, run.y, "-", label=f"N = {int(run.meta['N'])}")
            err = np.abs(run.y - np.interp(run.t, ref.t, ref.y))
            ok = np.isfinite(err) & (err < 10)
            ax_err.semilogy(
                run.t[ok],
                err[ok],
                "-",
                label=f"N = {int(run.meta['N'])}",
            )
        ax_y.plot(ref.t, ref.y, "k--", label=f"N = {int(ref.meta['N'])}")
        ax_err.set_yscale("log", nonpositive="mask")
        ax_y.set_ylim(-2, 2)
        ax_y.set_ylabel(r"$Y$")
        ax_err.set_ylabel(rf"$|Y-Y_{{N={int(ref.meta['N'])}}}|$")
        ax_err.set_xlabel(r"$t$")
        ax_y.set_title(rf"nonlinear spring (without damping), $B={B}$")
        ax_y.legend()
        ax_err.legend()
        ax_y.grid(True, linestyle=":", alpha=0.5)
        ax_err.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"ex2b_rk2_B{B}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def plot_ex3():
    FIGURES.mkdir(exist_ok=True)
    runs = [load(p) for p in sorted(DATA.glob("ex3_*.csv"))]
    groups: dict[float, list[Run]] = {}
    for run in runs:
        groups.setdefault(float(run.meta["sigma"]), []).append(run)
    for group in groups.values():
        group.sort(key=lambda r: r.meta["N"])
    for sigma, group in groups.items():
        omega = float(group[0].meta["omega"])
        fig, (ax_y, ax_err) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        t_fine = np.linspace(
            float(group[0].meta["t0"]), float(group[0].meta["t1"]), 1000
        )
        for run in group:
            ax_y.plot(run.t, run.y, "-", label=f"N = {int(run.meta['N'])}")
            ax_err.semilogy(
                run.t,
                np.abs(run.y - exact(run.t, sigma, omega)),
                "-",
                label=f"N = {int(run.meta['N'])}",
            )
        ax_y.plot(
            t_fine,
            exact(t_fine, sigma, omega),
            "k--",
            label="exact",
        )
        ax_err.set_yscale("log", nonpositive="mask")
        ax_y.set_ylim(-2, 2)
        ax_y.set_ylabel(r"$Y$")
        ax_err.set_ylabel(r"$|Y-Y_\mathrm{exact}|$")
        ax_err.set_xlabel(r"$t$")
        ax_y.set_title(
            rf"damped oscillator, $\sigma={sigma}$, $\omega={omega}$"
        )
        ax_y.legend(loc="lower right")
        ax_err.legend()
        ax_y.grid(True, linestyle=":", alpha=0.5)
        ax_err.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"ex3_ab2_sigma{sigma}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def main():
    plot_ex2a()
    plot_ex2b()
    plot_ex3()


if __name__ == "__main__":
    main()
