"""AERE 5460, Homework 1.

Plots the deliverables for Exercises 2-3. Writes
plots to the `figures` directory relative to this source file.
"""

from dataclasses import dataclass
from pathlib import Path

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
    return


def main():
    plot_ex2a()


if __name__ == "__main__":
    main()
