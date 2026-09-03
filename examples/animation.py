#!/usr/bin/env python3
"""Landscape panel with a looping animation centred on it. Ctrl-C to quit."""

import contextlib
import itertools
import time
from pathlib import Path

from PIL import Image, ImageSequence

from akp02 import AKP02, Orientation

ANIMATION = Path(__file__).resolve().parent / "samurai.webp"
DEFAULT_MS = 62.5


def frames_of(path: Path) -> list[tuple[Image.Image, float]]:
    """Decode every frame up front; seeking mid-loop costs more than RAM."""
    with Image.open(path) as src:
        return [
            (f.convert("RGB"), f.info.get("duration", DEFAULT_MS) / 1000)
            for f in ImageSequence.Iterator(src)
        ]


with AKP02() as panel, contextlib.suppress(KeyboardInterrupt):
    frames = frames_of(ANIMATION)

    panel.orientation(Orientation.LANDSCAPE)
    panel.screen_on()
    panel.clear()

    deadline = time.monotonic()
    for frame, delay in itertools.cycle(frames):
        panel.show(frame, at=(785, 68))
        now = time.monotonic()
        deadline = max(deadline + delay, now)
        time.sleep(deadline - now)
