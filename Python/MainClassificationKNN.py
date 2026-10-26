"""Classification of two Gaussian clouds using k nearest neighbors."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def main(seed=7, show=True):
    rng = np.random.default_rng(seed)
    n = 10_000
    X = np.vstack((
        0.8 * rng.standard_normal((n // 2, 2)) + [2, 2],
        0.8 * rng.standard_normal((n // 2, 2)) + [-2, -1],
    ))
    y = np.concatenate((np.ones(n // 2, dtype=int), 2 * np.ones(n // 2, dtype=int)))

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    x_new = np.array([[3.0, 3.0]])  # scikit-learn expects one row per observation
    prediction = int(model.predict(x_new)[0])
    print(f"Predicted class: {prediction}")

    fig, ax = plt.subplots()
    for label, color in [(1, "r"), (2, "b")]:
        ax.scatter(*X[y == label].T, s=12, c=color, marker="o", label=f"class {label}", alpha=0.55)
    ax.plot(x_new[0, 0], x_new[0, 1], "ks", markersize=10,
            markerfacecolor="none", markeredgewidth=2, label="query point")
    ax.set(xlabel="feature 1", ylabel="feature 2", title="k-nearest neighbors (k=3)")
    ax.legend(loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(seed=args.seed, show=not args.no_show)
