"""Compare least-squares (L2) and least-absolute-deviations (L1) regression."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


def least_absolute_deviations(x, y):
    """Solve min_{a,b} sum_i |a*x_i+b-y_i| by linear programming."""
    n = len(x)
    # Unknowns [slope, intercept, t_1, ..., t_n], t_i >= |residual_i|.
    obj = np.r_[np.zeros(2), np.ones(n)]
    #  a*x_i + b - t_i <= y_i; -a*x_i - b - t_i <= -y_i
    C1 = np.column_stack((x, np.ones(n), -np.eye(n)))
    C2 = np.column_stack((-x, -np.ones(n), -np.eye(n)))
    result = linprog(obj, A_ub=np.vstack((C1, C2)),
                     b_ub=np.r_[y, -y], bounds=[(None, None)] * 2 + [(0, None)] * n,
                     method="highs")
    if not result.success:
        raise RuntimeError(f"L1 regression failed: {result.message}")
    return result.x[:2]


def main(show=True):
    x = np.arange(1., 11.)
    y = np.array([0.2, 0.5, 0.3, 3.5, 1.0, 1.5, 1.8, 2.0, 2.3, 2.2])

    A = np.column_stack((x, np.ones_like(x)))
    p2, *_ = np.linalg.lstsq(A, y, rcond=None)
    p1 = least_absolute_deviations(x, y)
    print(f"L2 coefficients [slope, intercept] = {p2}")
    print(f"L1 coefficients [slope, intercept] = {p1}")

    xx = np.linspace(1, 10, 300)
    fig, ax = plt.subplots()
    ax.plot(x, y, "om", linewidth=2, label="data")
    ax.plot(x[3], y[3], "or", label="outlier")
    ax.plot(xx, np.polyval(p2, xx), "-b", linewidth=2, label=r"$L^2$")
    ax.plot(xx, np.polyval(p1, xx), "--k", linewidth=2, label=r"$L^1$")
    ax.set(xlabel="x", ylabel="y", ylim=(0, 4), title="Robust linear regression")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return p2, p1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(show=not args.no_show)
