"""Inverse channel stacking for the SVD image-compression example."""

import numpy as np


def matrix_to_image(matrix, m, n):
    """Map a (3*m) x n float array back to a clipped uint8 RGB image."""
    matrix = np.asarray(matrix)
    if matrix.shape != (3 * m, n):
        raise ValueError(f"Expected matrix shape {(3 * m, n)}, got {matrix.shape}.")
    channels = [matrix[i * m : (i + 1) * m, :] for i in range(3)]
    # Round to the nearest intensity and saturate to [0, 255].
    return np.clip(np.rint(np.stack(channels, axis=2)), 0, 255).astype(np.uint8)


Matrix2Image = matrix_to_image
