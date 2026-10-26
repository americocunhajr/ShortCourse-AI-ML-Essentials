"""Forecast chaotic Rössler dynamics with a leaky Echo State Network (ESN).

Faithful to the MATLAB demonstration: teacher-forced reservoir training,
regularized linear readout of *increments*, and autonomous closed-loop forecasting.
Default integration and reservoir settings match the source. Use --quick for a
shorter demonstration and --no-show in headless settings.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from scipy.sparse.linalg import eigs


@dataclass(frozen=True)
class Parameters:
    a: float = 0.2
    b: float = 0.2
    c: float = 5.7
    dt: float = 0.01
    transient_time: float = 200.0
    training_time: float = 400.0
    forecast_time: float = 200.0
    reservoir_size: int = 500
    reservoir_density: float = 0.02
    spectral_radius: float = 0.95
    input_scaling: float = 0.30
    bias_scaling: float = 0.20
    leak_rate: float = 0.35
    washout: int = 300
    ridge_beta: float = 1e-4
    prediction_threshold: float = 0.40
    largest_lyapunov_approx: float = 0.0714
    seed: int = 7


def rossler_rhs(t, state, a, b, c):
    x, y, z = state
    return [-y - z, x + a * y, b + z * (x - c)]


def normalize_states(data, mean, std):
    return (data - mean[:, None]) / std[:, None]


def inverse_normalize_states(data, mean, std):
    return data * std[:, None] + mean[:, None]


def compute_nrmse(prediction, truth):
    denominator = np.std(truth, ddof=1)
    if denominator < np.sqrt(np.finfo(float).eps):
        denominator = 1.0
    return float(np.sqrt(np.mean((prediction - truth) ** 2)) / denominator)


def estimate_spectral_radius(W):
    """Largest eigenvalue magnitude (not the spectral norm or singular value)."""
    try:
        eigenvalue = eigs(W, k=1, which="LM", tol=1e-8,
                          maxiter=2000, return_eigenvectors=False)[0]
        return float(abs(eigenvalue))
    except Exception as exc:
        print(f"ARPACK failed ({exc}); falling back to dense eigenvalues.")
        return float(np.max(np.abs(np.linalg.eigvals(W.toarray()))))


def make_reservoir(p, rng):
    Win = np.empty((p.reservoir_size, 4))
    Win[:, 0] = p.bias_scaling * rng.uniform(-1, 1, size=p.reservoir_size)
    Win[:, 1:] = p.input_scaling * rng.uniform(-1, 1, size=(p.reservoir_size, 3))
    Wres = sparse.random(p.reservoir_size, p.reservoir_size,
                         density=p.reservoir_density, format="csr",
                         random_state=rng, data_rvs=lambda n: rng.uniform(-1, 1, n))
    measured = estimate_spectral_radius(Wres)
    if measured <= np.finfo(float).eps:
        raise ValueError("Reservoir has negligible spectral radius.")
    Wres = Wres * (p.spectral_radius / measured)
    verified = estimate_spectral_radius(Wres)
    print(f"Requested spectral radius: {p.spectral_radius:.4f}")
    print(f"Verified spectral radius:  {verified:.4f}")
    return Win, Wres


def reservoir_step(reservoir_state, input_state, Win, Wres, leak_rate):
    # r_(k+1) = (1-alpha)*r_k + alpha*tanh(Win*[1;u_k] + Wres*r_k)
    preactivation = Win[:, 0] + Win[:, 1:] @ input_state + Wres @ reservoir_state
    return (1 - leak_rate) * reservoir_state + leak_rate * np.tanh(preactivation)


def generate_data(p):
    total_time = p.transient_time + p.training_time + p.forecast_time
    n_steps = round(total_time / p.dt)
    t_eval = np.linspace(0, total_time, n_steps + 1)
    print("Integrating the Rössler system...")
    solution = solve_ivp(rossler_rhs, (0, total_time), [1.0, 1.0, 1.0],
                         args=(p.a, p.b, p.c), t_eval=t_eval,
                         rtol=1e-10, atol=1e-12, max_step=p.dt)
    if not solution.success:
        raise RuntimeError(solution.message)

    first_data_index = np.searchsorted(t_eval, p.transient_time)
    n_training = round(p.training_time / p.dt) + 1
    n_forecast = round(p.forecast_time / p.dt)
    required = n_training + n_forecast
    data = solution.y[:, first_data_index:first_data_index + required]
    time = t_eval[first_data_index:first_data_index + required] - t_eval[first_data_index]
    if data.shape[1] != required:
        raise ValueError("Insufficient samples for training/forecast split.")
    print(f"Training snapshots: {n_training}")
    print(f"Forecast snapshots: {n_forecast}")
    return data, time, n_training, n_forecast


def style_dark_axes(ax):
    ax.set_facecolor((0.03, 0.03, 0.04))
    ax.tick_params(colors=(0.82, 0.84, 0.88))
    ax.xaxis.label.set_color((0.92, 0.92, 0.95))
    ax.yaxis.label.set_color((0.92, 0.92, 0.95))
    if hasattr(ax, "zaxis"):
        ax.zaxis.label.set_color((0.92, 0.92, 0.95))
    for spine in ax.spines.values():
        spine.set_color((0.82, 0.84, 0.88))
    if hasattr(ax, "zaxis"):
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.07, 0.07, 0.09, 1.0))
    ax.grid(True, color=(0.45, 0.47, 0.52), alpha=0.22)


def create_figures(data, time, n_training, forecast_truth, training_prediction,
                   forecast_prediction, nrmse, valid_time, output_dir):
    background = (0.03, 0.03, 0.04)
    truth_color = (0.90, 0.92, 0.96)
    teacher_color = (0.20, 0.78, 0.92)
    forecast_color = (1.00, 0.72, 0.15)
    training_end = time[n_training - 1]

    fig, axes = plt.subplots(3, 1, figsize=(14.5, 9.0), sharex=True,
                             facecolor=background, constrained_layout=True)
    for i, ax in enumerate(axes):
        low, high = np.min(data[i]), np.max(data[i])
        pad = 0.08 * max(high - low, 1)
        ax.axvspan(time[0], training_end, color=(0.15, 0.55, 0.75), alpha=0.10,
                   label="Training region" if i == 0 else None)
        ax.axvspan(training_end, time[-1], color=(0.38, 0.25, 0.06), alpha=0.30,
                   label="Forecast region" if i == 0 else None)
        ax.plot(time, data[i], color=truth_color, lw=1.35,
                label="True trajectory" if i == 0 else None)
        ax.plot(time[:n_training], training_prediction[i], color=teacher_color, lw=1.15,
                label="Teacher-forced ESN" if i == 0 else None)
        ax.plot(time[n_training:], forecast_prediction[i], color=forecast_color, lw=1.65,
                label="Autonomous ESN" if i == 0 else None)
        ax.axvline(training_end, ls="--", color=(0.95, 0.35, 0.30), lw=1.5,
                   label="End of training" if i == 0 else None)
        ax.set(ylabel=f"{('x', 'y', 'z')[i]}(t)", ylim=(low - pad, high + pad))
        style_dark_axes(ax)
        if i == 0:
            ax.text(0.02, 0.88, "TRAINING", color=teacher_color,
                    transform=ax.transAxes, weight="bold")
            ax.text(0.69, 0.88, "FORECAST", color=forecast_color,
                    transform=ax.transAxes, weight="bold")
            legend = ax.legend(ncol=3, loc="upper right", fontsize=8,
                               facecolor=(0.07, 0.07, 0.09), edgecolor=(0.30, 0.30, 0.35))
            for txt in legend.get_texts():
                txt.set_color(truth_color)
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Rössler Forecast with Reservoir Computing  |  NRMSE = {nrmse:.3f}"
                 f"  |  Valid time = {valid_time:.2f}", color=truth_color, fontsize=16)
    plot1 = output_dir / "Rossler_ESN_TimeSeries.png"
    fig.savefig(plot1, dpi=180, facecolor=fig.get_facecolor())

    training = data[:, :n_training]
    fig3d = plt.figure(figsize=(14, 6.5), facecolor=background, constrained_layout=True)
    axes3d = [fig3d.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    for ax, future, title, color, label in [
        (axes3d[0], forecast_truth, "True Rössler attractor", truth_color, "True forecast segment"),
        (axes3d[1], forecast_prediction, "ESN autonomous forecast", forecast_color, "ESN forecast segment"),
    ]:
        ax.plot(*training, color=(0.28, 0.55, 0.72), lw=0.9, label="Training segment")
        ax.plot(*future, color=color, lw=1.3, label=label)
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.set_title(title, color=truth_color)
        ax.view_init(elev=24, azim=38)
        style_dark_axes(ax)
        legend = ax.legend(loc="best", fontsize=8, facecolor=(0.07, 0.07, 0.09))
        for txt in legend.get_texts():
            txt.set_color(truth_color)
    # Both panels use exactly the same ranges, including both future trajectories.
    compared = np.hstack((training, forecast_truth, forecast_prediction))
    for ax in axes3d:
        ax.set(xlim=(compared[0].min(), compared[0].max()),
               ylim=(compared[1].min(), compared[1].max()),
               zlim=(compared[2].min(), compared[2].max()))
    fig3d.suptitle("Rössler Dynamics: True Evolution and Reservoir Forecast",
                   color=truth_color, fontsize=16)
    plot2 = output_dir / "Rossler_ESN_Attractor.png"
    fig3d.savefig(plot2, dpi=180, facecolor=fig3d.get_facecolor())
    print(f"Figures exported:\n  {plot1}\n  {plot2}")
    return fig, fig3d


def main(p=None, output_dir=None, show=True):
    p = p or Parameters()
    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(p.seed)
    data, time, n_training, n_forecast = generate_data(p)
    training_data = data[:, :n_training]
    forecast_truth = data[:, n_training:]

    # Training-only standardization prevents leakage from the forecast window.
    training_mean = np.mean(training_data, axis=1)
    training_std = np.std(training_data, axis=1, ddof=1)
    training_std[training_std < np.sqrt(np.finfo(float).eps)] = 1.0
    normalized = normalize_states(data, training_mean, training_std)
    train_normalized = normalized[:, :n_training]
    forecast_normalized_truth = normalized[:, n_training:]

    Win, Wres = make_reservoir(p, rng)
    n_pairs = n_training - 1
    n_reg = n_pairs - p.washout
    if n_reg <= 0:
        raise ValueError("Washout is too large for the training interval.")
    dimension = 1 + 3 + p.reservoir_size
    features = np.empty((n_reg, dimension))
    targets = np.empty((n_reg, 3))
    reservoir = np.zeros(p.reservoir_size)

    # Teacher forcing: u_k = true observation, target = u_(k+1) - u_k.
    for k in range(n_pairs):
        u = train_normalized[:, k]
        reservoir = reservoir_step(reservoir, u, Win, Wres, p.leak_rate)
        if k >= p.washout:
            j = k - p.washout
            features[j, :4] = np.r_[1.0, u]
            features[j, 4:] = reservoir
            targets[j] = train_normalized[:, k + 1] - u

    # Ridge least squares with SPD solve, without an explicit inverse.
    gram = features.T @ features
    gram.flat[::dimension + 1] += p.ridge_beta
    Wout_T = solve(gram, features.T @ targets, assume_a="pos", check_finite=True)
    Wout = Wout_T.T
    del features, targets, gram, Wout_T

    # Teacher-forced one-step reconstruction, starting from a fresh reservoir.
    training_prediction_normalized = np.full((3, n_training), np.nan)
    reservoir = np.zeros(p.reservoir_size)
    for k in range(n_pairs):
        u = train_normalized[:, k]
        reservoir = reservoir_step(reservoir, u, Win, Wres, p.leak_rate)
        extended = np.r_[1., u, reservoir]
        training_prediction_normalized[:, k + 1] = u + Wout @ extended

    # Autonomous prediction with NO access to true future observations.
    autonomous_input = train_normalized[:, -1].copy()
    prediction_normalized = np.empty((3, n_forecast))
    for j in range(n_forecast):
        reservoir = reservoir_step(reservoir, autonomous_input, Win, Wres, p.leak_rate)
        extended = np.r_[1., autonomous_input, reservoir]
        autonomous_input = autonomous_input + Wout @ extended
        prediction_normalized[:, j] = autonomous_input
    if not np.all(np.isfinite(prediction_normalized)):
        raise FloatingPointError("ESN autonomous predictions became nonfinite.")

    training_prediction = inverse_normalize_states(training_prediction_normalized,
                                                    training_mean, training_std)
    forecast_prediction = inverse_normalize_states(prediction_normalized,
                                                    training_mean, training_std)

    individual_nrmse = [compute_nrmse(forecast_prediction[i], forecast_truth[i])
                        for i in range(3)]
    overall_nrmse = float(np.sqrt(np.mean((prediction_normalized - forecast_normalized_truth) ** 2)))
    error = np.sqrt(np.mean((prediction_normalized - forecast_normalized_truth) ** 2, axis=0))
    exceeded = np.flatnonzero(error > p.prediction_threshold)
    # Match the source convention: first forecast snapshot has lead time zero.
    valid_time = float(exceeded[0] * p.dt) if len(exceeded) else p.forecast_time
    lyapunov_times = valid_time * p.largest_lyapunov_approx
    print("\nForecast diagnostics\n--------------------")
    for coord, val in zip("xyz", individual_nrmse):
        print(f"NRMSE {coord}: {val:.6f}")
    print(f"Overall normalized RMSE: {overall_nrmse:.6f}")
    if len(exceeded):
        print(f"Valid prediction time: {valid_time:.3f} time units")
    else:
        print(f"Error threshold not reached. Valid prediction time >= {valid_time:.3f}")
    print(f"Approximate prediction horizon: {lyapunov_times:.3f} Lyapunov times")
    figures = create_figures(data, time, n_training, forecast_truth, training_prediction,
                             forecast_prediction, overall_nrmse, valid_time, output_dir)
    if show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
    return {"nrmse": individual_nrmse, "overall_nrmse": overall_nrmse,
            "valid_time": valid_time, "forecast_prediction": forecast_prediction}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Smaller reservoir and shorter integration")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.quick:
        p = Parameters(transient_time=30., training_time=50., forecast_time=20.,
                       dt=0.02, reservoir_size=80, washout=100, seed=args.seed)
    else:
        p = Parameters(seed=args.seed)
    main(p=p, output_dir=args.output_dir, show=not args.no_show)
