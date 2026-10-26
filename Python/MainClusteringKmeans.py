"""Discover three Gaussian clusters with k-means."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def main(seed=7, show=True):
    rng = np.random.default_rng(seed)
    X = np.vstack((
        0.3 * rng.standard_normal((50, 2)) + [2, 2],
        0.4 * rng.standard_normal((50, 2)) + [-2, -1],
        0.3 * rng.standard_normal((50, 2)) + [0, 4],
    ))

    model = KMeans(n_clusters=3, n_init=10, random_state=seed)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    fig, ax = plt.subplots()
    for j in range(3):
        ax.scatter(*X[labels == j].T, s=50, label=f"cluster {j + 1}")
    ax.plot(centers[:, 0], centers[:, 1], "kp", markersize=14,
            markerfacecolor="none", markeredgewidth=2, label="centroids")
    ax.set(xlabel="feature 1", ylabel="feature 2", title="k-means (k=3)")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return labels, centers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(seed=args.seed, show=not args.no_show)
