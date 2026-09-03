# akp02

[![PyPI](https://img.shields.io/pypi/v/akp02)](https://pypi.org/project/akp02/)
[![Python](https://img.shields.io/pypi/pyversions/akp02)](https://pypi.org/project/akp02/)
[![License: MPL-2.0](https://img.shields.io/badge/license-MPL--2.0-blue)](LICENSE)

Linux library for the **Ajazz AKP02** 9.2" (1920x462) USB
secondary display. The AKP02 officially ships with Windows-only
software; this project is the result of reverse engineering its USB HID
protocol so the panel can be driven natively from Linux.

Confirmed on real hardware: full-screen frames, partial (region)
updates including correct color rendering, brightness, screen on/off,
clear, firmware version query, splash-screen orientation, and sustained
keepalive operation.

## Install

### Arch Linux

Available in the AUR as
[`python-akp02`](https://aur.archlinux.org/packages/python-akp02):

```bash
paru -S python-akp02      # or yay, or makepkg -si
```

The package installs the udev rule for you. Replug the panel afterwards
and you're done -- skip the rest of this section.

### Other distributions

```bash
pip install akp02
```

Then install the udev rule so the device is usable without root. The
rule ships inside the package, so there's no need to clone the repo:

```bash
sudo cp "$(python -c 'from importlib.resources import files; print(files("akp02") / "udev" / "99-akp02.rules")')" /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**Replug the device after installing the rule** -- udev only applies it
on the next connect.

### From a checkout

```bash
git clone https://github.com/deonspengler/akp02
cd akp02
pip install .
sudo cp udev/99-akp02.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Verify

With the panel connected:

```python
from akp02 import AKP02

with AKP02() as panel:
    panel.screen_on()
    panel.set_brightness(30)
```

The backlight should visibly dim. A permissions error here almost always
means the udev rule isn't active yet -- confirm the file is in
`/etc/udev/rules.d/` and that you replugged the device.

### Note on `hidapi`

Most Linux users get a prebuilt `hidapi` wheel and need nothing extra.
If pip falls back to building it from source -- unusual architecture, or
a Python version newer than the available wheels -- install the headers
first:

```bash
sudo apt install libhidapi-dev libudev-dev     # Debian/Ubuntu
sudo dnf install hidapi-devel systemd-devel    # Fedora
sudo pacman -S hidapi                          # Arch
```

## Library

```python
from akp02 import AKP02

with AKP02() as panel:
    panel.screen_on()
    panel.set_brightness(80)
    panel.show(pil_image)              # full screen, letterboxed if needed
    panel.show(widget, at=(1600, 16))  # partial update, rest preserved
```

`AKP02()` raises `DeviceNotFoundError` if the panel isn't connected; the
message lists what was on the bus instead.

The panel sleeps unless it receives periodic heartbeats, so entering the
context manager starts a keepalive thread for you (every 5s by default).
`AKP02(keepalive_interval=30)` changes the interval;
`AKP02(keepalive_interval=None)` starts nothing, for one-shot use or if
you'd rather drive it yourself:

```python
panel.start_keepalive(interval_sec=10)  # or resume after a disconnect
panel.stop_keepalive()                  # let the panel sleep
panel.heartbeat()                       # one beat, e.g. from your own timer
```

The thread starts in `__enter__` rather than `__init__`, so constructing
an `AKP02` never has a background side effect. Constructing one without
the `with` statement means no automatic keepalive -- call
`start_keepalive()` yourself in that case, and `close()` when done.

All public methods are thread-safe: a single lock is held across whole
multi-report image transfers so the keepalive thread can never
interleave with or corrupt a frame. Image composition and JPEG encoding
happen outside the lock, so the keepalive thread isn't blocked by them.

### Orientation and mounting

The panel is sold as a 1920x462 landscape strip, and that is the default:
`show()` takes landscape images and `at=(x, y)` is in landscape
coordinates. Hang it on its side and switch:

```python
from akp02 import AKP02, Orientation

panel.orientation(Orientation.PORTRAIT)  # now a 462x1920 surface
panel.orientation()                      # read it back, sends nothing
panel.size                               # (462, 1920) -- follows the mode
```

`show()` expects images in whichever mode is active, so build them at
`panel.size` and they stay right across a switch. The JPEG on the wire
is 462x1920 either way; only your view of it changes.

If the panel is mounted the other way up, set `panel.inverted = True`.
That turns the image a further 180 degrees, so what you draw at the top
appears at the top of the panel as you are looking at it. It is a
rotation, not a mirror -- text stays readable. It is software only, has
no hardware command behind it, and does not change `panel.size`.

`orientation()` also sends the device's `SET` command, which orients the
splash screen the panel draws for itself at power-on. That setting
persists across power cycles and is the command's *only* effect --
nothing on the device rotates frames the host sends, which is why the
rotation above happens here. Entering the context manager deliberately
does not send `SET`, so a `with` block can't overwrite a splash setting
you never mentioned.

A region update sent immediately after a full-frame draw needs a brief
settling delay first, or the full frame can silently fail to render at
all -- this is tracked and applied automatically; callers don't need to
do anything.

Region placement is also corrected automatically: the device renders a
region with the wrong color unless its position satisfies a specific,
confirmed alignment rule (see Protocol notes below). `show()` nudges
the position by a few pixels when needed and warns when it does, so
`at=(x, y)` always renders correctly without the caller having to know
about this.

## Examples

Runnable scripts in [`examples/`](examples) -- repo only, not shipped in the
sdist -- one file each, with the panel connected:

```bash
python examples/clock.py       # portrait background, one-second region updates
python examples/animation.py   # landscape, a looping webp drawn into a region
```

`clock.py` composites a scrim over the strip of background the time sits on,
then redraws only that band once a second; the rest of the image is never
resent. `animation.py` does the same thing at frame rate, decoding every webp
frame up front and cycling them into a fixed region. Between them they cover
both orientations, full-frame draws, region updates, and the keepalive the
context manager runs for you.

`clock.py` renders its text with Liberation Mono (`ttf-liberation` on Arch,
`fonts-liberation` on Debian/Ubuntu) and falls back to Pillow's built-in font
if that isn't installed. Both image assets were generated with ComfyUI for
this repository and are covered by its license -- reuse them freely.

For a full application built on the library rather than a demo of it, see
[akp02d](https://github.com/deonspengler/akp02d): a stats panel daemon that
renders live system information to the display -- CPU, memory, network and
temperatures -- updated in place with region draws.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The suite runs entirely against a fake HID device -- no physical
hardware or `hidapi` installation required -- and holds the library at
100% line and branch coverage. It covers protocol byte-exactness pinned
against real captures, the region color-alignment correction with real
hardware-confirmed data points, and concurrency behavior: dead-thread
recovery after a disconnect, bounded shutdown against a wedged device,
and lock-interleaving prevention verified under real contention.

The orientation geometry is checked against Pillow and against the full
frame rather than against the library's own arithmetic: for each of the
four mode/inverted combinations, the net transform is identified by name
from the whole dihedral group (and asserted to be a rotation, never a
reflection), and a region is required to reproduce exactly the bytes the
full frame put at that rect.

For linting and type checking as well:

```bash
pip install -e ".[dev]"
ruff check .
mypy
```

## Protocol notes (reverse engineered)

Plain USB HID, no encryption. VID:PID `0300:3017`. Output reports are
1024 bytes (EP1 OUT), input reports 512 bytes (EP2 IN).

**Commands** are one zero-padded report:
`"CRT" + 00 00 + <mnemonic> + 00 00 + <params>`

| Mnemonic  | Action                                                |
| --------- | ----------------------------------------------------- |
| `HAN`     | screen off                                            |
| `DIS`     | screen on                                             |
| `LIG`     | brightness (1 param byte, 0-100)                      |
| `CONNECT` | heartbeat (device sleeps without periodic heartbeats) |
| `STP`     | commit / render buffered image data                   |
| `SET`     | splash-screen orientation (see below)                 |

Layout exceptions: `CLE` (clear) uses a 3-byte gap plus a literal `0xFF`
trailer instead of the usual 2-byte gap; `VER` (firmware version) has a
leading `0x00` device-context byte before `CRT`, no gap after the
mnemonic, and returns its answer via a synchronous `GET_REPORT` on the
IN endpoint. Gap sizes are **not** universal across commands -- verify
per command when adding new ones.

**Brightness reset on screen-on** (observed on real hardware): the
device reverts its backlight to its factory default (80) when the
screen comes back on after `HAN`. The library tracks the last
host-requested brightness and re-applies it in `screen_on()` -- right
after `DIS`, in the same lock hold -- so an off/on cycle preserves the
brightness the caller set.

**Image transfers**: a 32-byte `CRT..DRA` header (big-endian length =
payload + 0x20, then width/height/x/y as big-endian uint16, all zero
for a full-panel draw) followed immediately by JPEG bytes, chunked into
1024-byte reports, then `STP`. The JPEG is **always** 462x1920
**portrait**; the library's orientation mode only decides which rotation
it applies to get there -- landscape is 90 degrees **clockwise**
(confirmed on hardware; the other way renders the panel upside down),
portrait passes through unrotated, and `inverted` turns either a further
180. Non-zero header coordinates draw a partial region in that buffer's
space; content outside it is preserved.

**Region color alignment** (confirmed empirically on real hardware): a
region update renders with the wrong color -- not a placement shift --
unless the value that actually lands in the header's x field satisfies
`x % 8 == 2`. Which of your coordinates that is depends on the mode: in
landscape it is `462 - y - height`, in portrait it is `x` directly, and
`inverted` reverses each. Root cause understood, not just observed: 462
(the axis this applies to) doesn't divide evenly into 8- or 16-pixel
JPEG blocks the way 1920 (the other axis, which shows no equivalent
sensitivity) does, so the device's firmware evidently pads its buffer
on that axis with a fixed internal offset. The library corrects this
automatically rather than requiring callers to pick special
coordinates.

**Settling delay**: a region update sent immediately after a
full-frame draw can fail to render the full frame at all unless a
brief delay (confirmed: 4ms is sufficient, 0ms fails; the library uses
20ms for margin) separates them. Region-after-region and
full-after-full both need no delay. Likely cause: a full-frame draw is
a clean buffer replace, but a region draw is a read-modify-write
against the current framebuffer; if that read starts before the
previous commit has actually finished settling internally, the
in-flight commit can apparently be corrupted or aborted.

**Splash-screen orientation** (confirmed on real hardware):
`"CRT" + 00,00 + "SET" + 00,00 + 0x00 + <orientation byte>`, where the
orientation byte is `0x00` for horizontal or `0x01` for vertical.
Found by comparing two real captures of the same action with different
outcomes -- the first capture only ever showed the default value
(indistinguishable from "no parameter"), a second capture with actual
mode switches revealed the real byte. The setting persists across a
power cycle, verified by physically unplugging and replugging after
setting each value.

It affects **only** the splash screen the device draws for itself at
power-on -- it does not rotate frames sent by the host, and nothing
visible happens when the command is sent. (An earlier reading of the
same evidence had it applying to the live display too; the persisted
splash coming back rotated after a replug is consistent with both, and
the narrower one is correct.) The device therefore offers no way to
rotate host frames, which is why the library rotates them itself.

hidapi note: `write()` needs a leading `0x00` report-ID placeholder per
report; the kernel strips it before the wire (captures show no
report-ID byte on the wire itself).

## License

MPL-2.0
