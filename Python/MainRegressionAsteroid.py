"""Fit an implicit conic curve to five planar measurements."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def conic(x, y, beta):
    a, b, c, d, e = beta
    return a * x**2 + b * x * y + c * y**2 + d * x + e * y + 1


def main(show=True):
    xdata = np.array([8.025, 10.170, 11.202, 10.736, 9.092])
    ydata = np.array([8.310, 6.355, 3.212, 0.375, -2.267])
    A = np.column_stack((xdata**2, xdata * ydata, ydata**2, xdata, ydata))
    beta = np.linalg.solve(A, -np.ones_like(xdata))
    print(f"Conic coefficients [a, b, c, d, e] = {beta}")

    xx, yy = np.meshgrid(np.linspace(-5, 12, 400), np.linspace(-10, 10, 400))
    fig, ax = plt.subplots()
    contour = ax.contour(xx, yy, conic(xx, yy, beta), levels=[0], colors="b", linewidths=2)
    ax.plot(xdata, ydata, "om", label="measurements")
    ax.set(xlim=(-5, 12), ylim=(-10, 10), xlabel="x", ylabel="y", title="Implicit conic fit")
    ax.set_aspect("equal", adjustable="box")
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
