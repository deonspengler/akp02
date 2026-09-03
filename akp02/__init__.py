"""Linux library for the Ajazz AKP02 (9.2" 1920x462) USB HID display.

Usage:

    from akp02 import AKP02

    with AKP02() as panel:                          # keepalive starts here
        panel.set_brightness(80)
        while True:
            panel.show(render_dashboard())          # full screen
            panel.show(clock_img, at=(1600, 20))    # partial update
            time.sleep(1)

The device sleeps without heartbeats, so `with` starts a keepalive
thread by default; AKP02(keepalive=None) leaves it to the caller.

Protocol summary (reverse-engineered, verified against real usbmon captures):
  - Plain USB HID, no encryption. VID:PID = 0300:3017.
  - Commands: "CRT" + 00 00 + <mnemonic> + 00 00 + <params>, zero-padded to
    one HID report (see the AKP02.CMD_* attributes for known mnemonics).
    Layout exceptions: clear "CLE" (see AKP02.clear) and the firmware
    version query "VER" (see AKP02.firmware_version).
  - Image transfers: 32-byte CRTDRA header (see AKP02._crtdra_header) followed
    immediately by JPEG bytes, sent as one continuous stream chunked into
    1024-byte HID reports, then a commit (STP) command.
  - The JPEG payload is 462x1920 PORTRAIT (landscape content rotated 90
    degrees clockwise before encoding; confirmed on hardware -- rotating
    the other way renders the panel upside down).
  - hidapi's write() needs a leading 0x00 placeholder byte per report; the
    kernel strips it before the wire (captures show no report-ID byte).
  - Region updates: non-zero x/y/width/height in the header draw a partial
    region (portrait-space coordinates); content outside it is preserved.
    Verified on hardware. A region renders with the WRONG COLOR (not a
    placement shift) unless its position satisfies a specific alignment
    -- see AKP02.PORTRAIT_X_ALIGN_MODULUS/RESIDUE for the confirmed rule
    and why; show() corrects this automatically.
"""

from __future__ import annotations

import io
import struct
import threading
import time
import warnings
from collections.abc import Sequence
from typing import NamedTuple, Protocol, Self, cast

from PIL import Image

__all__ = ["AKP02"]

# Single source of truth: pyproject declares `dynamic = ["version"]` and
# hatchling reads this line at build time, so there is no second copy to
# forget to bump.
__version__ = "1.1.0"


class _Rect(NamedTuple):
    """An axis-aligned region in the portrait buffer's coordinate space.

    All-zero (FULL_SCREEN) means "whole panel" in the CRTDRA header.
    """

    x: int
    y: int
    width: int
    height: int


_FULL_SCREEN = _Rect(0, 0, 0, 0)


class _HidDevice(Protocol):
    """The subset of hidapi's hid.device interface this library uses.

    Structural (Protocol) rather than nominal because hidapi ships no
    type stubs; anything with these five methods works, which is also
    the contract a test double passed as AKP02(dev=...) must satisfy.
    """

    def write(self, data: bytes) -> int: ...

    def get_input_report(self, report_id: int, size: int) -> Sequence[int]: ...

    def get_serial_number_string(self) -> str: ...

    def error(self) -> str: ...

    def close(self) -> None: ...


class AKP02:
    """Handle to an AKP02 panel.

    All public methods are thread-safe: a single lock serializes every
    command and holds for the WHOLE of a multi-report image transfer, so
    the keepalive thread can never inject a heartbeat mid-frame.

    Usage:
        with AKP02() as panel:
            panel.show(pil_image)

    `with` starts the keepalive thread; constructing an AKP02 on its
    own never starts a thread.
    """

    class DeviceNotFoundError(Exception):
        """Raised when no AKP02 is found on the USB bus."""

    VENDOR_ID = 0x0300
    PRODUCT_ID = 0x3017

    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 462

    HID_REPORT_SIZE = 1024  # EP1 OUT wMaxPacketSize from the device descriptor
    INPUT_REPORT_SIZE = 512  # EP2 IN wMaxPacketSize
    JPEG_QUALITY = 85  # default; override per instance via __init__
    JPEG_QUALITY_MIN = 1
    JPEG_QUALITY_MAX = 95  # Pillow advises <= 95; above it, size grows for
    # almost no visual gain
    BRIGHTNESS_MIN = 0  # range of the LIG command's parameter byte
    BRIGHTNESS_MAX = 100

    # A region update sent immediately after a full-frame draw can fail
    # to render the full frame at all (confirmed on real hardware: 4ms
    # was found sufficient, 0ms fails). Region-after-region and
    # full-after-full both need no delay at all -- only this one specific
    # transition does. Likely cause: a full-frame draw is a clean buffer
    # replace, but a region draw is a read-modify-write against whatever's
    # currently in the framebuffer; if that read starts before the
    # previous full-frame commit has actually finished settling internally
    # (not just "USB bytes received"), the in-flight commit can apparently
    # get corrupted/aborted. 20ms is used here (5x the confirmed-working
    # 4ms) for margin against real-world timing jitter -- this delay only
    # ever fires once per full-frame draw, not per region update, so
    # there's no real cost to being generous with it.
    FULL_TO_REGION_SETTLE_SEC = 0.02

    # Confirmed on real hardware: a region update renders with the wrong
    # color unless the position value that actually lands in the CRTDRA
    # header's x field (== SCREEN_HEIGHT - y - height, see
    # _to_portrait_rect) satisfies this residue mod this modulus. Root
    # cause understood, not just observed: SCREEN_HEIGHT (462) does not
    # divide evenly into 8- or 16-pixel JPEG blocks the way SCREEN_WIDTH
    # (1920) does, so the device's firmware evidently pads/rounds its
    # buffer on that axis with a fixed internal offset. Landscape x
    # (which maps to the cleanly-tiling 1920-pixel portrait axis) shows
    # no equivalent sensitivity -- confirmed by testing it directly.
    # show() corrects this automatically (see _align_region_y); these
    # constants exist to make that correction's target explicit rather
    # than a magic number, and so a future finding can update it in one
    # place.
    PORTRAIT_X_ALIGN_MODULUS = 8
    PORTRAIT_X_ALIGN_RESIDUE = 2

    CMD_SCREEN_OFF = b"HAN"
    CMD_SCREEN_ON = b"DIS"
    CMD_BRIGHTNESS = b"LIG"  # + 1 param byte, 0-100
    CMD_HEARTBEAT = b"CONNECT"  # device sleeps without periodic heartbeats
    CMD_COMMIT = b"STP"  # render buffered image data
    CMD_BOOT_ORIENTATION = b"SET"  # + 0x00 + orientation byte, see below

    # Confirmed on real hardware: sets the panel orientation immediately
    # AND persists it as the boot-time default (survives a power cycle --
    # verified by physically unplugging/replugging after setting each
    # value). One command does both; there's no separate "apply now" vs
    # "save as default" step.
    ORIENTATION_HORIZONTAL = 0x00
    ORIENTATION_VERTICAL = 0x01

    # Well inside the device's sleep timeout, with room for a missed beat.
    KEEPALIVE_INTERVAL_SEC = 5.0

    def __init__(
        self,
        dev: _HidDevice | None = None,
        jpeg_quality: int | None = None,
        keepalive: float | None = KEEPALIVE_INTERVAL_SEC,
    ) -> None:
        """Open the panel.

        Pass an already-open hidapi device (or any _HidDevice-shaped
        object, e.g. a test double) via dev to skip discovery.
        jpeg_quality (1-95) overrides the default encoding quality --
        small text UIs may want it higher.

        keepalive is the interval in seconds __enter__ starts the
        keepalive thread with, or None to not start one. Only recorded
        here -- no thread is started by __init__ (see __enter__).

        Raises AKP02.DeviceNotFoundError if the panel isn't connected.
        """
        if (
            jpeg_quality is not None
            and not self.JPEG_QUALITY_MIN <= jpeg_quality <= self.JPEG_QUALITY_MAX
        ):
            raise ValueError(
                f"jpeg_quality must be {self.JPEG_QUALITY_MIN}-{self.JPEG_QUALITY_MAX}"
            )
        # Checked here too, so the traceback points at the construction
        # site rather than at __enter__.
        if keepalive is not None:
            self._check_keepalive_interval(keepalive)
        self.jpeg_quality: int = (
            self.JPEG_QUALITY if jpeg_quality is None else jpeg_quality
        )
        self._dev: _HidDevice | None = dev if dev is not None else self._open()
        self._lock = threading.Lock()
        # Guards _keepalive_thread/_keepalive_stop management only. Separate
        # from _lock so stop_keepalive() never holds a lock the keepalive
        # thread needs (heartbeat() takes _lock), which could deadlock under
        # specific timing: join() while the thread is blocked in heartbeat().
        self._keepalive_mgmt_lock = threading.Lock()
        self._keepalive_stop: threading.Event | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._keepalive_interval: float | None = keepalive
        # Tracks whether the last show() was full-screen, to know when the
        # settling delay is needed (see FULL_TO_REGION_SETTLE_SEC). Starts
        # True: if the very first show() call ever made is a region update
        # on a never-painted screen, there's no hardware evidence either
        # way, so default to the safe (delay-inserting) assumption rather
        # than risk the same failure mode with no data to justify skipping it.
        self._last_show_was_full_screen = True

    @classmethod
    def _open(cls) -> _HidDevice:
        import hid

        if not hasattr(hid, "device"):
            # Two incompatible PyPI packages import as `hid`. This library
            # needs `hidapi` (which provides hid.device); `pip install hid`
            # installs the other one.
            raise ImportError(
                "the installed 'hid' module is not the 'hidapi' package "
                "(hid.device is missing); run: pip uninstall hid && "
                "pip install hidapi"
            )

        devices = hid.enumerate()
        same_vendor = [d for d in devices if d["vendor_id"] == cls.VENDOR_ID]
        matches = [d for d in same_vendor if d["product_id"] == cls.PRODUCT_ID]
        if not matches:
            if same_vendor:
                detail = (
                    f"found vendor {cls.VENDOR_ID:04x} but wrong "
                    "product id(s): "
                    + ", ".join(
                        f"{d['product_id']:04x} {d.get('product_string')!r}"
                        for d in same_vendor
                    )
                )
            else:
                detail = "candidates: " + (
                    ", ".join(
                        f"{d['vendor_id']:04x}:{d['product_id']:04x} "
                        f"{d.get('product_string')!r}"
                        for d in devices
                    )
                    or "none"
                )
            raise cls.DeviceNotFoundError(
                f"no HID device {cls.VENDOR_ID:04x}:{cls.PRODUCT_ID:04x} "
                f"found; {detail}"
            )
        # The AKP02 currently exposes a single HID interface; if a firmware
        # revision ever adds more (as sibling Ajazz keypads do), prefer the
        # lowest interface number for a deterministic choice.
        chosen = min(matches, key=lambda d: d.get("interface_number", 0))
        dev = hid.device()
        dev.open_path(chosen["path"])
        # hidapi is untyped; this is the one point where its object enters
        # our typed world, asserted to match the _HidDevice surface.
        return cast(_HidDevice, dev)

    # -- context manager / lifecycle --

    def __enter__(self) -> Self:
        """Start the keepalive (unless disabled) and return self.

        The device sleeps without heartbeats, so this calls the public
        start_keepalive() for you; AKP02(keepalive=None) opts out.

        Here rather than in __init__ because the keepalive closure holds
        a strong reference to self (an auto-started panel never closed
        could never be collected, and would keep writing), a subclass
        would see heartbeats before its own __init__ finished, and a
        dev= test double would get a background writer just by being
        constructed. This is also where the panel is actually in use,
        and it pairs with __exit__ -> close() -> stop_keepalive().
        """
        if self._keepalive_interval is not None:
            self.start_keepalive(self._keepalive_interval)
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the device on context exit."""
        self.close()

    def close(self, timeout_sec: float = 5.0) -> None:
        """Stop the keepalive thread and close the device.

        The keepalive is stopped first, then _lock is taken before
        touching the handle, so the close can't land mid-transfer: any
        thread already inside show() finishes its header + chunks +
        commit first, and any call that starts afterwards sees a closed
        device and raises RuntimeError. Closing under an active writer
        would be worse than a logic error -- libhidapi frees the handle,
        so an in-flight write() in another thread is a use-after-free.

        The lock acquisition is bounded by timeout_sec for the same
        reason stop_keepalive's join is: a write wedged against a
        hung device must not hang the caller forever. On timeout the
        handle is deliberately LEAKED rather than closed, since the
        blocked writer may still be holding it; leaking a file
        descriptor is the cheaper of the two failures.

        Safe to call more than once.
        """
        self.stop_keepalive(timeout_sec)
        if not self._lock.acquire(timeout=timeout_sec):
            warnings.warn(
                f"akp02 close() could not acquire the device lock within "
                f"{timeout_sec}s (a transfer is likely wedged); leaking the "
                f"HID handle rather than closing it out from under the "
                f"in-flight writer",
                stacklevel=2,
            )
            return
        try:
            if self._dev is not None:
                self._dev.close()
                self._dev = None
        finally:
            self._lock.release()

    # -- low-level (callers must hold self._lock) --

    def _write_report(self, data: bytes) -> None:
        """Send one HID report, zero-padded to HID_REPORT_SIZE.

        A 0x00 report-ID placeholder byte is prepended for hidapi's
        write(); the kernel strips it before the wire.

        hidapi signals failure by returning -1 instead of raising, so the
        return value is checked here; otherwise a mid-frame failure (e.g.
        the device being unplugged) would silently drop reports.
        """
        if self._dev is None:
            raise RuntimeError("device is closed")
        written = self._dev.write(
            bytes([0x00]) + data + bytes(self.HID_REPORT_SIZE - len(data))
        )
        if written < 0:
            raise OSError("HID write failed: " + (self._dev.error() or "unknown error"))

    def _send_command(self, mnemonic: bytes, params: bytes = b"") -> None:
        self._write_report(b"CRT" + bytes(2) + mnemonic + bytes(2) + params)

    @staticmethod
    def _crtdra_header(payload_len: int, rect: _Rect, flag: int = 0xB1) -> bytes:
        """Build the 32-byte image transfer header.

        rect is in portrait buffer space; _FULL_SCREEN (all zeros)
        addresses the whole panel.

        bytes 0-2   : "CRT"
        bytes 3-4   : zero gap
        bytes 5-7   : "DRA"
        bytes 8-11  : total length, big-endian uint32 (payload size + 0x20)
        byte 12     : flag byte, 0xb1 in captures (meaning unknown)
        bytes 13-14 : width,  big-endian uint16 (0 = whole panel)
        bytes 15-16 : height, big-endian uint16
        bytes 17-18 : x offset, big-endian uint16
        bytes 19-20 : y offset, big-endian uint16
        bytes 21-31 : zero padding
        """
        header = bytearray(32)
        header[0:3] = b"CRT"
        header[5:8] = b"DRA"
        header[8:12] = struct.pack(">I", payload_len + 0x20)
        header[12] = flag & 0xFF
        header[13:15] = struct.pack(">H", rect.width & 0xFFFF)
        header[15:17] = struct.pack(">H", rect.height & 0xFFFF)
        header[17:19] = struct.pack(">H", rect.x & 0xFFFF)
        header[19:21] = struct.pack(">H", rect.y & 0xFFFF)
        return bytes(header)

    # -- commands --

    def screen_off(self) -> None:
        """Turn the display panel off ("HAN")."""
        with self._lock:
            self._send_command(self.CMD_SCREEN_OFF)

    def screen_on(self) -> None:
        """Turn the display panel on ("DIS")."""
        with self._lock:
            self._send_command(self.CMD_SCREEN_ON)

    def set_brightness(self, percent: int) -> None:
        """Set the backlight brightness ("LIG"), 0-100 percent."""
        if not self.BRIGHTNESS_MIN <= percent <= self.BRIGHTNESS_MAX:
            raise ValueError(
                f"brightness must be {self.BRIGHTNESS_MIN}-{self.BRIGHTNESS_MAX}"
            )
        with self._lock:
            self._send_command(self.CMD_BRIGHTNESS, bytes([percent]))

    def heartbeat(self) -> None:
        """Send one keepalive heartbeat ("CONNECT") manually."""
        with self._lock:
            self._send_command(self.CMD_HEARTBEAT)

    def set_boot_orientation(self, orientation: str) -> None:
        """Set the panel orientation ("horizontal" or "vertical").

        Takes effect on the live display immediately, and persists as
        the boot-time default surviving a power cycle. Confirmed on
        real hardware: byte layout is "CRT" + 00,00 + "SET" + 00,00 +
        0x00 + orientation byte, following the standard 2-byte-gap
        command pattern (unlike CLE/VER's exceptions).
        """
        values = {
            "horizontal": self.ORIENTATION_HORIZONTAL,
            "vertical": self.ORIENTATION_VERTICAL,
        }
        if orientation not in values:
            raise ValueError(
                f"orientation must be 'horizontal' or 'vertical', got {orientation!r}"
            )
        with self._lock:
            self._send_command(
                self.CMD_BOOT_ORIENTATION, bytes([0x00, values[orientation]])
            )

    def clear(self) -> None:
        """Clear the screen ("CLE").

        Layout exception: 3-byte gap plus a hardcoded 0xFF trailer (0xFF
        means "all" in the sibling multi-key products' clear command),
        unlike _send_command's 2-byte gap. Gap size is apparently not
        universal across CRT-tagged commands -- verify per command rather
        than assuming _send_command's layout.
        """
        with self._lock:
            self._write_report(b"CRT" + bytes(2) + b"CLE" + bytes(3) + bytes([0xFF]))

    def firmware_version(self) -> str:
        """Query the firmware version ("VER"). Confirmed on real hardware.

        Layout exception: a leading 0x00 device-context byte precedes
        "CRT" and there is no gap after the mnemonic. The response is
        read with get_input_report(), which is a GET_REPORT request on
        the control endpoint (not an interrupt read on the IN endpoint,
        though the 512-byte size matches EP2 IN's wMaxPacketSize); its
        first byte echoes the report ID.
        """
        with self._lock:
            dev = self._dev
            if dev is None:
                raise RuntimeError("device is closed")
            self._write_report(bytes(1) + b"CRT" + bytes(2) + b"VER")
            response = bytes(dev.get_input_report(0x00, self.INPUT_REPORT_SIZE + 1))
        return response[1:].split(b"\x00", 1)[0].decode("ascii", errors="replace")

    def serial_number(self) -> str:
        """Return the device's USB serial number (e.g. "C511D378553A").

        Unlike firmware_version(), this isn't part of the custom "CRT"
        command protocol at all -- it's the standard USB iSerial device
        descriptor string, read via hidapi's own
        get_serial_number_string(), the same value shown by `lsusb -v`.
        No custom protocol involved, nothing to reverse-engineer here.
        """
        with self._lock:
            dev = self._dev
            if dev is None:
                raise RuntimeError("device is closed")
            return dev.get_serial_number_string()

    # -- images --

    def _align_region_y(self, y: int, height: int) -> int:
        """Nudge y to satisfy the color-alignment rule.

        See PORTRAIT_X_ALIGN_MODULUS/RESIDUE for the rule itself. Tries
        both directions (shift the region up or down in landscape
        space) and prefers whichever is the smaller shift, falling back
        to the other direction if the preferred one would push the
        region outside the panel. Warns whenever a correction is applied
        -- callers who need exact, unshifted positioning can see this in
        their logs and adjust the region they ask for. If neither
        direction fits (the region already spans nearly the whole
        height), warns and returns y unchanged rather than raising --
        an occasional color glitch is a smaller failure than refusing to
        draw at all. Returns y unchanged, with no warning, if it's
        already aligned.

        Confirmed on real hardware: height == SCREEN_HEIGHT (a region
        spanning the full short axis, which can only ever sit at y=0)
        needs no correction and gets none, silently -- consistent with
        our understanding of the underlying bug, since a region with no
        partial remainder on this axis leaves nothing for the device's
        edge-padding logic to misalign in the first place. Without this,
        the "can't find a valid shift" fallback above would still avoid
        touching y (there's no room to shift when height fills the
        whole axis), but would incorrectly warn every time regardless.
        """
        if height == self.SCREEN_HEIGHT:
            return y

        portrait_x = self.SCREEN_HEIGHT - y - height
        residue = portrait_x % self.PORTRAIT_X_ALIGN_MODULUS
        if residue == self.PORTRAIT_X_ALIGN_RESIDUE:
            return y

        # Shifting y UP by `up_shift` decreases portrait_x by the same
        # amount (portrait_x = SCREEN_HEIGHT - y - height) -- solve for
        # the smallest non-negative up_shift reaching the target residue,
        # and its complementary down_shift reaching the same residue the
        # other way.
        up_shift = (
            residue - self.PORTRAIT_X_ALIGN_RESIDUE
        ) % self.PORTRAIT_X_ALIGN_MODULUS
        down_shift = self.PORTRAIT_X_ALIGN_MODULUS - up_shift

        def fits(candidate: int) -> bool:
            return candidate >= 0 and candidate + height <= self.SCREEN_HEIGHT

        for _shift, candidate_y in sorted(
            [(up_shift, y + up_shift), (down_shift, y - down_shift)]
        ):
            if fits(candidate_y):
                warnings.warn(
                    f"akp02: region y={y} shifted to y={candidate_y} "
                    f"(height={height}) for correct color rendering -- see "
                    f"AKP02.PORTRAIT_X_ALIGN_MODULUS/PORTRAIT_X_ALIGN_RESIDUE",
                    stacklevel=3,
                )
                return candidate_y

        warnings.warn(
            f"akp02: region y={y} (height={height}) cannot be shifted to a "
            f"color-safe position without leaving the panel -- drawing "
            f"uncorrected; this region's color may render incorrectly",
            stacklevel=3,
        )
        return y

    def _to_portrait_rect(self, x: int, y: int, width: int, height: int) -> _Rect:
        """Map a rect from landscape space to the portrait buffer space.

        Follows the clockwise rotation used for full frames:
        (x, y) -> (H-1-y, x), applied to the rect's bounding box; width
        and height swap.

        This must stay in step with the transpose in show(): the two are
        descriptions of the same mapping, and changing one alone leaves
        full frames looking correct while silently misplacing every
        region update.
        """
        return _Rect(x=self.SCREEN_HEIGHT - y - height, y=x, width=height, height=width)

    def _letterbox(self, img: Image.Image) -> Image.Image:
        """Scale an image to fit the panel, preserving aspect ratio.

        The result is centered on a black canvas of the panel's size.
        """
        w, h = self.SCREEN_WIDTH, self.SCREEN_HEIGHT
        scale = min(w / img.width, h / img.height)
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(img, ((w - new_size[0]) // 2, (h - new_size[1]) // 2))
        return canvas

    def _encode_jpeg(self, img: Image.Image) -> bytes:
        """JPEG-encode at the instance's quality (see show())."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()

    def show(
        self, image: Image.Image | bytes, at: tuple[int, int] | None = None
    ) -> None:
        """Display an image on the panel.

        Accepts a landscape PIL image, or ready portrait JPEG bytes
        (sent untouched; full-screen only).

        at=None: full-screen, letterboxed if not exactly the panel size.
        at=(x, y): partial update at that landscape position, sized by the
        image; the rest of the screen is preserved. The device renders a
        region's color wrong unless its position satisfies a confirmed
        alignment requirement (see PORTRAIT_X_ALIGN_MODULUS/RESIDUE) --
        this is corrected automatically, nudging y by a few pixels and
        warning when it does. Only y is ever adjusted; the position you
        pass for x, and the image's size, are never changed.

        PIL input gets the protocol's mandatory 90-degree clockwise
        rotation and is JPEG-encoded (before the lock, so the keepalive
        thread isn't blocked); the lock then holds for the whole header +
        chunks + commit sequence so no report can interleave.

        A region update sent immediately after a full-frame draw needs a
        brief settling delay first, or the full frame can fail to render
        at all -- see FULL_TO_REGION_SETTLE_SEC. This is tracked and
        applied automatically; callers don't need to do anything.
        """
        is_region = at is not None
        rect = _FULL_SCREEN
        if isinstance(image, Image.Image):
            img = image if image.mode == "RGB" else image.convert("RGB")
            if at is None:
                if img.size != (self.SCREEN_WIDTH, self.SCREEN_HEIGHT):
                    img = self._letterbox(img)
            else:
                x, y = at
                if (
                    x < 0
                    or y < 0
                    or x + img.width > self.SCREEN_WIDTH
                    or y + img.height > self.SCREEN_HEIGHT
                ):
                    raise ValueError(
                        f"region ({x},{y},{img.width}x{img.height}) does not "
                        f"fit the {self.SCREEN_WIDTH}x{self.SCREEN_HEIGHT} screen"
                    )
                y = self._align_region_y(y, img.height)
                rect = self._to_portrait_rect(x, y, img.width, img.height)
            # transpose() (exact permutation) rather than rotate()
            img = img.transpose(Image.Transpose.ROTATE_270)
            jpeg = self._encode_jpeg(img)
        else:
            if at is not None:
                raise ValueError("at= requires a PIL image, not raw bytes")
            jpeg = image
        payload = self._crtdra_header(len(jpeg), rect) + jpeg
        with self._lock:
            if is_region and self._last_show_was_full_screen:
                time.sleep(self.FULL_TO_REGION_SETTLE_SEC)
            for offset in range(0, len(payload), self.HID_REPORT_SIZE):
                self._write_report(payload[offset : offset + self.HID_REPORT_SIZE])
            self._send_command(self.CMD_COMMIT)
            self._last_show_was_full_screen = not is_region

    # -- keepalive --

    @staticmethod
    def _check_keepalive_interval(interval_sec: float) -> None:
        """Reject an interval that would make the keepalive loop spin.

        The loop is `while not stop.wait(interval_sec)`, and wait()
        returns immediately for 0 or less -- so a bad interval doesn't
        fail loudly, it becomes a tight loop writing to the device.
        """
        if interval_sec <= 0:
            raise ValueError(
                f"keepalive interval must be greater than 0 seconds, got "
                f"{interval_sec!r}; pass keepalive=None to AKP02() to not "
                f"start one automatically"
            )

    def start_keepalive(self, interval_sec: float = KEEPALIVE_INTERVAL_SEC) -> None:
        """Start a daemon thread sending a heartbeat every interval_sec.

        No-op if already running. Thread-safe: concurrent calls can't
        spawn two threads. __enter__ calls this for you unless
        AKP02(keepalive=None) was passed; calling it directly is still
        supported -- for a different interval, or to resume after a
        disconnect (see the dead-thread note below).

        Raises ValueError for a non-positive interval_sec, before the
        no-op check, so a bad value is reported either way.

        The guard tests is_alive() rather than "is not None" because a
        keepalive thread that lost the device exits on its own, leaving
        a dead Thread object behind; treating that as "already running"
        would silently no-op every restart for the life of the object,
        including a legitimate resume after the caller recovered the
        device. A dead thread is dropped and replaced here instead of
        clearing the state from inside loop() -- that would need the
        thread to take _keepalive_mgmt_lock, which stop_keepalive holds
        across its join(), and deadlock.
        """
        self._check_keepalive_interval(interval_sec)
        with self._keepalive_mgmt_lock:
            existing = self._keepalive_thread
            if existing is not None and existing.is_alive():
                return
            stop = self._keepalive_stop = threading.Event()

            def loop() -> None:
                while not stop.wait(interval_sec):
                    try:
                        self.heartbeat()
                    except Exception:
                        return  # device gone; let the main thread discover it

            self._keepalive_thread = threading.Thread(
                target=loop, name="akp02-keepalive", daemon=True
            )
            self._keepalive_thread.start()

    def stop_keepalive(self, timeout_sec: float = 5.0) -> None:
        """Stop the keepalive thread.

        The join is bounded by timeout_sec so a write blocked on a wedged
        device can't hang the caller forever; the thread is a daemon, so
        a leaked one can't block interpreter exit either.
        """
        with self._keepalive_mgmt_lock:
            if self._keepalive_thread is None or self._keepalive_stop is None:
                return
            self._keepalive_stop.set()
            self._keepalive_thread.join(timeout_sec)
            if self._keepalive_thread.is_alive():
                warnings.warn(
                    "akp02 keepalive thread did not stop within "
                    f"{timeout_sec}s; abandoning it (daemon thread)",
                    stacklevel=2,
                )
            self._keepalive_thread = None
            self._keepalive_stop = None
