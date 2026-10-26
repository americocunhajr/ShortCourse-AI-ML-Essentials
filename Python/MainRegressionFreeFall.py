"""Least-squares identification of the free-fall kinematic model."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main(show=True):
    time = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    displacement = 1e-2 * np.array([5.49, -9.45, -31.39, -75.59, -113.69])

    # s(t) = (g/2)*t**2 + v0*t + s0. Coefficients: [g, v0, s0].
    A = np.column_stack((0.5 * time**2, time, np.ones_like(time)))
    beta, *_ = np.linalg.lstsq(A, displacement, rcond=None)
    print(f"Estimated [g, v0, s0] = {beta}")

    t_fit = np.linspace(0, 0.5, 501)
    s_fit = 0.5 * beta[0] * t_fit**2 + beta[1] * t_fit + beta[2]
    fig, ax = plt.subplots()
    ax.plot(time, displacement, "om", linewidth=2, label="measurements")
    ax.plot(t_fit, s_fit, "-b", linewidth=2, label="quadratic regression")
    ax.set(xlabel="time (s)", ylabel="displacement (m)", title="Free-fall regression")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return beta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(show=not args.no_show)
