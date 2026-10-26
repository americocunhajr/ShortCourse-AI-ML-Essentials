"""Linear support-vector machine on two Gaussian classes."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


def main(seed=7, show=True):
    rng = np.random.default_rng(seed)
    X = np.vstack((rng.standard_normal((100, 2)) + 2,
                   rng.standard_normal((100, 2)) - 2))
    y = np.concatenate((np.ones(100, dtype=int), -np.ones(100, dtype=int)))

    svm = SVC(kernel="linear", C=1.0)
    svm.fit(X, y)
    x_new = np.array([[0.0, 0.0]])
    prediction = int(svm.predict(x_new)[0])
    print(f"Predicted class: {prediction}")

    fig, ax = plt.subplots()
    for label, color in [(-1, "b"), (1, "r")]:
        ax.scatter(*X[y == label].T, s=30, c=color, label=f"class {label}", alpha=0.7)
    ax.plot(*x_new[0], "ks", markersize=10, markerfacecolor="none", label="query point")

    # Decision boundary: w[0]*x + w[1]*y + b = 0.
    w, b = svm.coef_[0], svm.intercept_[0]
    x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    if abs(w[1]) > 1e-12:
        ax.plot(x_line, -(w[0] * x_line + b) / w[1], "k-", linewidth=1.5,
                label="decision boundary")
    else:
        ax.axvline(-b / w[0], color="k", linewidth=1.5, label="decision boundary")
    ax.set(xlabel="feature 1", ylabel="feature 2", title="Linear SVM")
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
