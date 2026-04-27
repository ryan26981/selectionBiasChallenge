"""
Step 5: Apply a selection-bias mask to a stippled image.
"""

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray, mask_img: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """
    Apply a mask to a stippled image to remove points in masked regions.

    Parameters
    ----------
    stipple_img : np.ndarray
        2D stippled image with values in [0, 1] (0 = black dot, 1 = white background).
    mask_img : np.ndarray
        2D mask image with values in [0, 1] (dark = remove region, light = keep region).
    threshold : float
        Pixels in mask_img below this value are considered part of the mask.

    Returns
    -------
    np.ndarray
        Masked stippled image with same shape as inputs.
    """
    if stipple_img.shape != mask_img.shape:
        raise ValueError(
            "stipple_img and mask_img must have the same shape. "
            f"Got {stipple_img.shape} and {mask_img.shape}."
        )

    threshold = float(np.clip(threshold, 0.0, 1.0))
    stipple = np.asarray(stipple_img, dtype=np.float32)
    mask = np.asarray(mask_img, dtype=np.float32)

    # Dark mask pixels indicate regions where stipples are removed (forced white).
    masked_stipple = np.where(mask < threshold, 1.0, stipple)
    return np.clip(masked_stipple, 0.0, 1.0).astype(np.float32, copy=False)
