"""Low-rank approximation of an RGB image by truncated singular-value decomposition."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.linalg import svd

from Image2Matrix import image_to_matrix
from Matrix2Image import matrix_to_image


IMAGE_FILE = Path(__file__).resolve().parent / "LaTourEiffel.jpg"


def main(rank=35, show=True):
    image = np.asarray(Image.open(IMAGE_FILE).convert("RGB"))
    matrix, m, n = image_to_matrix(image)
    U, s, Vt = svd(matrix, full_matrices=False)
    if not 1 <= rank <= len(s):
        raise ValueError(f"rank must be between 1 and {len(s)}")
    U_r, s_r, Vt_r = U[:, :rank], s[:rank], Vt[:rank, :]
    reduced_matrix = (U_r * s_r) @ Vt_r
    reduced_image = matrix_to_image(reduced_matrix, m, n)

    original_entries = matrix.size
    compressed_entries = U_r.size + s_r.size + Vt_r.size
    saving = 100 * (1 - compressed_entries / original_entries)
    print(f"Original entries:  {original_entries:d}")
    print(f"Low-rank entries:  {compressed_entries:d}")
    print(f"Entry savings:     {saving:.2f}%")
    print("Note: savings count matrix entries, not JPEG file sizes or file overhead.")

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[1].imshow(reduced_image)
    axs[1].set_title(f"Rank-{rank} approximation")
    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    if show:
        plt.show()
    return saving


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=35)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(rank=args.rank, show=not args.no_show)
