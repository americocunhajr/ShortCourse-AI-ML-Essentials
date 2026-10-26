"""Ward's hierarchical clustering and its dendrogram."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage


def main(seed=7, show=True):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((100, 2))

    fig1, ax1 = plt.subplots()
    ax1.plot(X[:, 0], X[:, 1], "r*")
    ax1.set(xlabel="x", ylabel="y", title="Observations")
    fig1.tight_layout()

    Z = linkage(X, method="ward", metric="euclidean")
    fig2, ax2 = plt.subplots()
    dendrogram(Z, ax=ax2)
    ax2.set(xlabel="Point index", ylabel="Linkage distance", title="Ward dendrogram")
    fig2.tight_layout()
    if show:
        plt.show()
    return Z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(seed=args.seed, show=not args.no_show)
