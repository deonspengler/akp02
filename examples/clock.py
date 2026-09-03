#!/usr/bin/env python3
"""Portrait background with a live clock. Ctrl-C to quit."""

import contextlib
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from akp02 import AKP02, Orientation

BACKGROUND = Path(__file__).resolve().parent / "background.jpg"
FONT = ImageFont.truetype("LiberationMono-Bold.ttf", 84)
BAND_Y, BAND_H = 170, 200
SCRIM = Image.new("RGBA", (462, BAND_H), (0, 0, 0, 110))


def band(plate: Image.Image, text: str) -> Image.Image:
    """Draw the centred time onto a copy of the pre-scrimmed plate."""
    img = plate.copy()
    draw = ImageDraw.Draw(img)
    _, _, w, h = draw.textbbox((0, 0), text, font=FONT)
    draw.text(((462 - w) / 2, (BAND_H - h) / 2), text, font=FONT, fill="white")
    return img.convert("RGB")


with AKP02() as panel, contextlib.suppress(KeyboardInterrupt):
    background = Image.open(BACKGROUND)
    plate = Image.alpha_composite(
        background.crop((0, BAND_Y, 462, BAND_Y + BAND_H)).convert("RGBA"), SCRIM
    )

    panel.orientation(Orientation.PORTRAIT)
    panel.screen_on()
    panel.show(background)

    while True:
        now = time.strftime("%H:%M:%S")
        panel.show(band(plate, now), at=(0, BAND_Y))
        time.sleep(1 - time.time() % 1)
