"""
Step 5: Apply the block-letter mask to the stippled image.
Where the mask is dark, stipples are removed (set to white), producing a biased estimate.
"""

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply a grayscale mask to a stippled image.

    Mask values are in [0, 1]: black (0) marks the masked region, white (1) keeps stipples.
    Pixels where ``mask_img < threshold`` are treated as mask (dark): stipples are removed
    by setting output to 1.0 (white). Elsewhere the stipple image is unchanged.

    Parameters
    ----------
    stipple_img : np.ndarray
        Stippled image, 2D, same shape as ``mask_img``.
    mask_img : np.ndarray
        Mask in [0, 1]; lower values are the letter / masked area.
    threshold : float
        Values strictly below this are considered part of the mask (dark).

    Returns
    -------
    np.ndarray
        Same shape as inputs; dtype follows ``stipple_img`` after masking.
    """
    if stipple_img.shape != mask_img.shape:
        raise ValueError(
            f"Shape mismatch: stipple_img {stipple_img.shape}, mask_img {mask_img.shape}"
        )
    if stipple_img.ndim != 2 or mask_img.ndim != 2:
        raise ValueError("stipple_img and mask_img must be 2D arrays")

    out = np.where(mask_img < threshold, 1.0, stipple_img)
    return np.asarray(out, dtype=stipple_img.dtype)
