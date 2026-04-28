"""
Step 4: Render a block letter (default "S") as a grayscale mask matching image dimensions.
Used for the selection-bias pattern in the statistics meme.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common system bold fonts; fall back to PIL default."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arialblk.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9,
) -> np.ndarray:
    """
    Draw a single letter in black on a white background.

    Parameters
    ----------
    height, width : int
        Output array shape (rows, cols).
    letter : str
        Character to draw (default ``"S"``).
    font_size_ratio : float
        Initial font size as a fraction of ``min(height, width)``; reduced if needed
        so the glyph fits inside the canvas.

    Returns
    -------
    np.ndarray
        Shape ``(height, width)``, float32, values in ``[0, 1]`` (letter ≈ 0, background 1).
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not letter:
        raise ValueError("letter must be non-empty")

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    max_dim = min(height, width)
    font_size = max(8, int(max_dim * font_size_ratio))
    font = _load_bold_font(font_size)

    while font_size >= 8:
        font = _load_bold_font(font_size)
        bbox = draw.textbbox((0, 0), letter, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= width and th <= height:
            break
        font_size = max(8, font_size - max(1, font_size // 25))

    bbox = draw.textbbox((0, 0), letter, font=font)
    x = (width - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (height - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.text((x, y), letter, fill=0, font=font)

    out = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)
