"""Fit daily case notifications in Rio de Janeiro to a logistic-derivative curve."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.special import expit


DATA_FILE = Path(__file__).resolve().parent / "COVID19RJ.mat"


def logistic_incidence(time, beta):
    """K*r*exp(-r*(t-t0)) / (1+exp(-r*(t-t0)))**2, evaluated stably."""
    rate, carrying_scale, midpoint = beta
    sigmoid = expit(rate * (np.asarray(time) - midpoint))
    return rate * carrying_scale * sigmoid * (1 - sigmoid)


def main(show=True):
    samples = loadmat(DATA_FILE)
    time = np.asarray(samples["xdata"], dtype=float).reshape(-1)
    notifications = np.asarray(samples["ydata"], dtype=float).reshape(-1)

    def objective(beta):
        return np.linalg.norm(logistic_incidence(time, beta) - notifications, ord=2)

    result = minimize(objective, x0=np.array([0.1, 8e4, 55]),
                      method="Nelder-Mead", options={"maxiter": 10_000,
                                                     "xatol": 1e-9, "fatol": 1e-6})
    if not result.success:
        print(f"Optimizer message: {result.message}")
    beta = result.x
    print(f"Estimated [rate, scale, midpoint] = {beta}")
    print(f"Residual L2 norm = {objective(beta):.6f}")

    fig, ax = plt.subplots()
    ax.stem(time, notifications, linefmt="b-", markerfmt="bo", basefmt=" ", label="daily observations")
    t_fit = np.linspace(time.min(), time.max(), 400)
    ax.plot(t_fit, logistic_incidence(t_fit, beta), "-r", linewidth=3, label="nonlinear fit")
    ax.set(xlabel="days", ylabel="new notifications per day", title="COVID-19: Rio de Janeiro")
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
