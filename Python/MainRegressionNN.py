"""Learn a one-step surrogate for the Lorenz flow using an MLP neural network.

The original MATLAB example uses a feedforward network with heterogeneous hidden
activations (logsig, radbas, purelin) and Levenberg-Marquardt training. scikit-learn
supports a common hidden activation, so this is a pedagogically equivalent, not
weight-for-weight identical, implementation with three logistic hidden layers.
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def lorenz(t, state, sigma=10.0, rho=28.0, beta=8 / 3):
    x, y, z = state
    return [sigma * (y - x), rho * x - x * z - y, x * y - beta * z]


def main(seed=7, trajectories=100, duration=8.0, dt=0.01, epochs=200, show=True):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration, round(duration / dt) + 1)
    input_pairs, output_pairs = [], []
    last_initial = None

    fig1 = plt.figure(figsize=(9, 7))
    ax1 = fig1.add_subplot(111, projection="3d")
    for _ in range(trajectories):
        x0 = 30 * (rng.random(3) - 0.5)
        sol = solve_ivp(lorenz, (t[0], t[-1]), x0, t_eval=t,
                        rtol=1e-9, atol=1e-11)
        if not sol.success:
            raise RuntimeError(sol.message)
        states = sol.y.T
        input_pairs.append(states[:-1])
        output_pairs.append(states[1:])
        ax1.plot(*states.T, linewidth=0.6, alpha=0.45)
        ax1.plot(*x0, "ro", markersize=3)
        last_initial = x0

    X, Y = np.vstack(input_pairs), np.vstack(output_pairs)
    ax1.set(xlabel="x", ylabel="y", zlabel="z", title=f"{trajectories} Lorenz training trajectories")
    fig1.tight_layout()

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    Y_scaled = y_scaler.fit_transform(Y)

    net = MLPRegressor(
        hidden_layer_sizes=(10, 10, 10), activation="logistic", solver="adam",
        random_state=seed, max_iter=epochs, batch_size=512,
        early_stopping=True, n_iter_no_change=15,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("once", ConvergenceWarning)
        net.fit(X_scaled, Y_scaled)
    print(f"Training pairs: {len(X):d}; MLP epochs: {net.n_iter_:d}")
    print(f"Training one-step RMSE: {np.sqrt(np.mean((net.predict(X_scaled) - Y_scaled)**2)):.5f} (standardized)")

    # Closed-loop rollout: future true observations are *not* passed to the NN.
    predicted = np.empty((len(t), 3))
    predicted[0] = last_initial
    for j in range(1, len(t)):
        next_scaled = net.predict(x_scaler.transform(predicted[j - 1:j]))
        predicted[j] = y_scaler.inverse_transform(next_scaled)[0]
        if not np.all(np.isfinite(predicted[j])):
            raise FloatingPointError("Non-finite neural-network prediction.")

    # A reference from the identical last initial condition, for comparison.
    true_solution = solve_ivp(lorenz, (t[0], t[-1]), last_initial, t_eval=t,
                              rtol=1e-9, atol=1e-11)
    if not true_solution.success:
        raise RuntimeError(true_solution.message)
    fig2 = plt.figure(figsize=(9, 7))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot(*true_solution.y, color="k", alpha=0.5, label="Lorenz reference")
    ax2.plot(*predicted.T, ":", linewidth=2, label="MLP closed-loop prediction")
    ax2.set(xlabel="x", ylabel="y", zlabel="z", title="Lorenz: direct neural one-step model")
    ax2.legend()
    fig2.tight_layout()
    if show:
        plt.show()
    return net, predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trajectories", type=int, default=100)
    parser.add_argument("--duration", type=float, default=8)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--quick", action="store_true", help="Fast, lower-fidelity classroom smoke test")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.trajectories, args.duration, args.epochs = 8, 2.0, 80
    main(seed=args.seed, trajectories=args.trajectories, duration=args.duration,
         dt=args.dt, epochs=args.epochs, show=not args.no_show)
