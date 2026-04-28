"""
Assemble the four-panel statistics meme (Reality, Your Model, Selection Bias, Estimate).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _resize_grayscale(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize to (height, width); values assumed in [0, 1]."""
    th, tw = target_hw
    if img.ndim != 2:
        raise ValueError("Expected 2D grayscale array")
    if img.shape[0] == th and img.shape[1] == tw:
        return np.asarray(img, dtype=np.float32)
    u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(u8, mode="L")
    pil = pil.resize((tw, th), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0


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
    Build a 1×4 figure and save it as a PNG.

    Panel order: Reality | Your Model | Selection Bias | Estimate.
    All inputs are resized to match ``original_img`` spatial dimensions.
    """
    target_hw = (original_img.shape[0], original_img.shape[1])

    imgs = [
        _resize_grayscale(original_img, target_hw),
        _resize_grayscale(stipple_img, target_hw),
        _resize_grayscale(block_letter_img, target_hw),
        _resize_grayscale(masked_stipple_img, target_hw),
    ]
    titles = ["Reality", "Your Model", "Selection Bias", "Estimate"]

    ncols = len(imgs)
    fig_w = 3.6 * ncols
    fig_h = 4.0
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
    )
    fig.patch.set_facecolor(background_color)

    for ax, panel, title in zip(axes, imgs, titles):
        ax.imshow(panel, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=13, fontweight="bold", color="#1a1a1a", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#c8c8c8")

    plt.savefig(
        output_path,
        dpi=dpi,
        facecolor=background_color,
        bbox_inches="tight",
        pad_inches=0.25,
        format="png",
    )
    plt.close(fig)
