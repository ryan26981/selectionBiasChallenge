"""
Final step: assemble a four-panel statistics meme.
"""

import numpy as np
import matplotlib.pyplot as plt


def _to_grayscale_array(img: np.ndarray, name: str) -> np.ndarray:
    """Convert input to a 2D float array in [0, 1]."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array. Got shape {arr.shape}.")
    return np.clip(arr, 0.0, 1.0)


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white",
) -> None:
    """
    Create and save a 1x4 statistics meme.

    Panels are labeled: Reality, Your Model, Selection Bias, Estimate.
    """
    original = _to_grayscale_array(original_img, "original_img")
    stipple = _to_grayscale_array(stipple_img, "stipple_img")
    block_letter = _to_grayscale_array(block_letter_img, "block_letter_img")
    masked = _to_grayscale_array(masked_stipple_img, "masked_stipple_img")

    target_shape = original.shape
    for name, arr in (
        ("stipple_img", stipple),
        ("block_letter_img", block_letter),
        ("masked_stipple_img", masked),
    ):
        if arr.shape != target_shape:
            raise ValueError(
                f"All images must have the same shape. "
                f"Expected {target_shape}, got {arr.shape} for {name}."
            )

    panels = (original, stipple, block_letter, masked)
    labels = ("Reality", "Your Model", "Selection Bias", "Estimate")

    h, w = target_shape
    aspect = w / max(h, 1)
    panel_width = 3.0 * aspect
    fig_width = max(12.0, panel_width * 4 + 1.0)
    fig_height = max(3.6, panel_width * 0.9)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(fig_width, fig_height),
        facecolor=background_color,
    )

    for ax, panel, label in zip(axes, panels, labels):
        ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")

    fig.patch.set_facecolor(background_color)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.06, wspace=0.03)

    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
