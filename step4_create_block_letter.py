import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_block_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common system fonts (Windows/macOS/Linux); fall back to default bitmap font."""
    candidates = []
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates.extend(
        [
            os.path.join(windir, "Fonts", "arialbd.ttf"),
            os.path.join(windir, "Fonts", "arial.ttf"),
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    )
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def create_block_letter_s(
    height: int, width: int, letter: str = "S", font_size_ratio: float = 0.9
) -> np.ndarray:
    """
    Create a block letter on a white canvas matching image dimensions.
    Returns float32 array (height × width) in [0, 1], black letter on white background.
    """
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    font_size = max(8, int(font_size_ratio * min(width, height)))
    font = _load_block_font(font_size)

    bbox = draw.textbbox((0, 0), letter, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) / 2 - bbox[0]
    y = (height - th) / 2 - bbox[1]

    draw.text((x, y), letter, font=font, fill=0)

    return (np.asarray(img, dtype=np.float32) / 255.0)
