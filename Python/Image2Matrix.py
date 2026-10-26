"""Convert an RGB image to/from the channel-stacked matrix used in the SVD demo."""

import numpy as np


def image_to_matrix(image):
    """Stack R, G, B (each m x n) into a (3*m) x n float matrix."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB array with shape (height, width, 3).")
    m, n, _ = image.shape
    return np.vstack([image[:, :, channel] for channel in range(3)]).astype(float), m, n


# MATLAB-inspired alias for comparing the two languages.
Image2Matrix = image_to_matrix
