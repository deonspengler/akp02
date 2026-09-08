"""Linux library for the Ajazz AKP02 (9.2" 1920x462) USB HID display.

The device sleeps without heartbeats, so `with` starts a keepalive
thread by default; AKP02(keepalive_interval=None) leaves it to the
caller.

The panel defaults to the landscape (1920x462) layout it is sold as:
show() takes landscape input and at=(x, y) as landscape coordinates,
rotating into the 462x1920 portrait buffer the device expects.
orientation(Orientation.PORTRAIT) switches to portrait (462x1920),
where input and coordinates are buffer space and unrotated. Setting
panel.inverted turns either mode a further 180 degrees, for a panel
mounted the other way up. The JPEG sent is 462x1920 in every case.

Protocol (reverse-engineered; details in the README, "Protocol notes
(reverse engineered)"): commands are "CRT" + 00 00 + <mnemonic> +
00 00 + <params> (AKP02.CMD_* for the mnemonics), and image transfers
are a 32-byte CRTDRA header (AKP02._crtdra_header) + JPEG, chunked
into 1024-byte reports, then a commit (STP).
"""

from __future__ import annotations

import io
import struct
import threading
import time
import warnings
from collections.abc import Sequence
from enum import IntEnum
from typing import NamedTuple, Protocol, Self, cast

from PIL import Image

__all__ = ["AKP02", "DeviceNotFoundError", "Orientation"]

# Single source of truth: pyproject declares `dynamic = ["version"]`, so
# hatchling reads this line at build time.
__version__ = "1.3.0"


class Orientation(IntEnum):
    """Panel orientation; the member value *is* the byte SET sends.

    LANDSCAPE is the 1920x462 layout the panel is sold as; PORTRAIT
    turns the glass 90 degrees, so the caller sees 462x1920. Neither is
    the transfer's own orientation -- that is always the 462x1920
    buffer.

    Confirmed on real hardware (verified across replugs): SET only sets
    which way up the device draws its own power-on splash, and persists
    that. Nothing on the device rotates host frames, which is why show()
    rotates them here.
    """

    LANDSCAPE = 0x00
    PORTRAIT = 0x01


# The single net rotation show() applies, per (orientation, inverted).
# One entry per case, so an inverted mount stays a rotation rather than
# the reflection flip-then-rotate composes to -- text would read
# backwards.
_TRANSPOSE: dict[tuple[Orientation, bool], Image.Transpose | None] = {
    (Orientation.LANDSCAPE, False): Image.Transpose.ROTATE_270,
    (Orientation.LANDSCAPE, True): Image.Transpose.ROTATE_90,
    (Orientation.PORTRAIT, False): None,
    (Orientation.PORTRAIT, True): Image.Transpose.ROTATE_180,
}


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
    type stubs. Also the contract a test double passed as
    AKP02(dev=...) must satisfy.
    """

    def write(self, data: bytes) -> int: ...

    def get_input_report(self, report_id: int, size: int) -> Sequence[int]: ...

    def get_serial_number_string(self) -> str: ...

    def error(self) -> str: ...

    def close(self) -> None: ...


class DeviceNotFoundError(Exception):
    """Raised when no AKP02 is found on the USB bus."""


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

    # Every attribute an instance may have. Mainly so a misspelled
    # `panel.inverted` raises instead of silently binding a dead
    # attribute and leaving a frame that just looks wrong; `inverted` is
    # the one settable knob with no method behind it. Subclasses without
    # their own __slots__ still get a __dict__ and stay open.
    __slots__ = (
        "_brightness",
        "_dev",
        "_keepalive_interval",
        "_keepalive_mgmt_lock",
        "_keepalive_stop",
        "_keepalive_thread",
        "_last_show_was_full_screen",
        "_lock",
        "_orientation",
        "inverted",
        "jpeg_quality",
        "jpeg_subsampling",
    )

    VENDOR_ID = 0x0300
    PRODUCT_ID = 0x3017

    # Named for the glass, not for an orientation: which one is "width"
    # depends on the mode, but the buffer sent is always SHORT x LONG.
    # _screen_size() gives the caller's width and height.
    PANEL_LONG_SIDE = 1920
    PANEL_SHORT_SIDE = 462

    HID_REPORT_SIZE = 1024  # EP1 OUT wMaxPacketSize from the device descriptor
    INPUT_REPORT_SIZE = 512  # EP2 IN wMaxPacketSize
    JPEG_QUALITY = 85  # default; override per instance via __init__
    JPEG_QUALITY_MIN = 1
    # Pillow advises <= 95; above it, size grows for almost no visual gain.
    JPEG_QUALITY_MAX = 95
    # Chroma subsampling. -1 leaves the choice to libjpeg, which picks
    # 4:2:0 at these qualities. 0 (4:4:4) is worth setting for small
    # colored text, which 4:2:0 visibly smears: measured on text-on-dark
    # content, roughly a third of the error for a third more bytes. Not
    # confirmed against the panel's own JPEG decoder, so try it on
    # hardware first. Pillow accepts out-of-range ints silently, hence
    # the explicit set.
    JPEG_SUBSAMPLING = -1
    JPEG_SUBSAMPLING_VALUES = (-1, 0, 1, 2)  # -1 auto, 4:4:4, 4:2:2, 4:2:0
    BRIGHTNESS_MIN = 0  # range of the LIG command's parameter byte
    BRIGHTNESS_MAX = 100
    # The device's factory default: observed on real hardware to revert
    # its backlight to this after an off->on cycle, which is why
    # screen_on() re-applies the caller's value.
    BRIGHTNESS_DEFAULT = 80

    # A region update sent immediately after a full-frame draw can stop
    # the full frame rendering at all (confirmed on real hardware: 4ms
    # suffices, 0ms fails). Only this transition needs it. Likely cause:
    # a full draw replaces the buffer, but a region is a
    # read-modify-write, and reading before the previous commit has
    # settled internally corrupts it. 20ms (5x the confirmed 4ms) for
    # jitter margin; it fires once per full-frame draw, not per region,
    # so being generous costs nothing.
    FULL_TO_REGION_SETTLE_SEC = 0.02

    # The residue to nudge a region onto, measured on hardware across
    # all eight. With e = (3 * header_x) mod 8, the device rotates the
    # channels by e mod 3 and slides the region's CONTENTS -- not its
    # rect -- by e // 3 pixels. Residues 0, 1 and 2 all give correct
    # color; only 0 also has no slide. 2 was the previous target, and
    # drew every corrected region 2px off with no outward sign.
    #
    # Callers choosing their own coordinates can skip the nudge
    # entirely: in landscape that is y == (PANEL_SHORT_SIDE - height)
    # mod 8, which depends on the height, so compute it, don't hardcode.
    #
    # A property of the buffer, not of the caller's coordinates: the rule
    # binds whichever coordinate _to_buffer_rect maps into the header's
    # x -- y in landscape, x in portrait. show() corrects this
    # automatically (see _align_axis).
    SHORT_AXIS_ALIGN_MODULUS = 8
    SHORT_AXIS_ALIGN_RESIDUE = 0

    CMD_SCREEN_OFF = b"HAN"
    CMD_SCREEN_ON = b"DIS"
    CMD_BRIGHTNESS = b"LIG"  # + 1 param byte, 0-100
    CMD_HEARTBEAT = b"CONNECT"  # device sleeps without periodic heartbeats
    CMD_COMMIT = b"STP"  # render buffered image data
    CMD_BOOT_ORIENTATION = b"SET"  # + 0x00 + Orientation value

    # Well inside the device's sleep timeout, with room for a missed beat.
    KEEPALIVE_INTERVAL_SEC = 5.0

    def __init__(
        self,
        dev: _HidDevice | None = None,
        jpeg_quality: int | None = None,
        keepalive_interval: float | None = KEEPALIVE_INTERVAL_SEC,
        *,
        jpeg_subsampling: int | None = None,
    ) -> None:
        """Open the panel.

        Pass an already-open hidapi device (or any _HidDevice-shaped
        object, e.g. a test double) via dev to skip discovery.
        jpeg_quality (1-95) overrides the default encoding quality --
        small text UIs may want it higher. jpeg_subsampling picks the
        chroma subsampling (see JPEG_SUBSAMPLING).

        keepalive_interval is the interval in seconds __enter__ starts
        the keepalive thread with, or None to not start one. Only
        recorded here; __init__ starts no thread and sends nothing to
        the device.

        The display mode starts at Orientation.LANDSCAPE; read or change
        it with orientation(). Set `inverted` on the instance for a panel
        mounted the other way up (software only; see show()).

        Raises DeviceNotFoundError if the panel isn't connected.
        """
        if (
            jpeg_quality is not None
            and not self.JPEG_QUALITY_MIN <= jpeg_quality <= self.JPEG_QUALITY_MAX
        ):
            raise ValueError(
                f"jpeg_quality must be {self.JPEG_QUALITY_MIN}-{self.JPEG_QUALITY_MAX}"
            )
        if (
            jpeg_subsampling is not None
            and jpeg_subsampling not in self.JPEG_SUBSAMPLING_VALUES
        ):
            raise ValueError(
                f"jpeg_subsampling must be one of "
                f"{self.JPEG_SUBSAMPLING_VALUES} (-1 libjpeg's own choice, "
                f"0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0), got {jpeg_subsampling!r}"
            )
        # Checked here too, so the traceback points at the construction
        # site rather than at __enter__.
        if keepalive_interval is not None:
            self._check_keepalive_interval(keepalive_interval)
        self.jpeg_quality: int = (
            self.JPEG_QUALITY if jpeg_quality is None else jpeg_quality
        )
        self.jpeg_subsampling: int = (
            self.JPEG_SUBSAMPLING if jpeg_subsampling is None else jpeg_subsampling
        )
        self._orientation: Orientation = Orientation.LANDSCAPE
        # Last host-requested brightness, for screen_on() to re-apply.
        self._brightness: int = self.BRIGHTNESS_DEFAULT
        # Extra 180-degree rotation for an inverted mount (see show()).
        self.inverted: bool = False
        self._dev: _HidDevice | None = dev if dev is not None else self._open()
        self._lock = threading.Lock()
        # Guards _keepalive_thread/_keepalive_stop only. Separate from
        # _lock so stop_keepalive() never holds a lock the keepalive
        # thread needs (heartbeat() takes _lock): join() while the thread
        # is blocked in heartbeat() would deadlock.
        self._keepalive_mgmt_lock = threading.Lock()
        self._keepalive_stop: threading.Event | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._keepalive_interval: float | None = keepalive_interval
        # Whether the last show() was full-screen, to know when the
        # settling delay is needed (see FULL_TO_REGION_SETTLE_SEC).
        # Starts True: a region update on a never-painted screen has no
        # evidence either way, so take the safe (delay-inserting) branch.
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
            raise DeviceNotFoundError(
                f"no HID device {cls.VENDOR_ID:04x}:{cls.PRODUCT_ID:04x} "
                f"found; {detail}"
            )
        # The AKP02 exposes a single HID interface today; if a firmware
        # revision ever adds more (as sibling Ajazz keypads do), the
        # lowest interface number keeps the choice deterministic.
        chosen = min(matches, key=lambda d: d.get("interface_number", 0))
        dev = hid.device()
        dev.open_path(chosen["path"])
        # hidapi is untyped: the one point where its object enters our
        # typed world, asserted to match the _HidDevice surface.
        return cast(_HidDevice, dev)

    # -- context manager / lifecycle --

    def __enter__(self) -> Self:
        """Start the keepalive (unless disabled) and return self.

        The device sleeps without heartbeats, so this calls
        start_keepalive() for you; AKP02(keepalive_interval=None) opts
        out.

        Here rather than in __init__ because the keepalive closure holds
        a strong reference to self (an auto-started panel never closed
        could never be collected, and would keep writing), a subclass
        would see heartbeats before its own __init__ finished, and a
        dev= test double would get a background writer just by being
        constructed. It also pairs with __exit__ -> close().

        No SET is sent here: the splash orientation is persisted device
        state, so pushing this instance's default at every `with` would
        overwrite a setting the caller never mentioned. That is
        orientation()'s to send, once.
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
        touching the handle, so the close can't land mid-transfer: a
        thread already inside show() finishes its header + chunks +
        commit, and any later call sees a closed device and raises
        RuntimeError. libhidapi frees the handle, so closing under an
        active writer would be a use-after-free, not just a logic error.

        The lock acquisition is bounded by timeout_sec so a write wedged
        against a hung device can't hang the caller forever; the two
        waits are sequential, so a fully wedged device can take up to
        twice timeout_sec. On timeout the handle is deliberately LEAKED,
        since the blocked writer may still hold it -- a leaked file
        descriptor is the cheaper failure.

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

        hidapi signals failure by returning -1 rather than raising, so
        the return value is checked here; otherwise a mid-frame unplug
        would silently drop reports.
        """
        if self._dev is None:
            raise RuntimeError("device is closed")
        if len(data) > self.HID_REPORT_SIZE:
            # Otherwise bytes() below gets a negative count and fails
            # with nothing pointing at the cause.
            raise ValueError(
                f"report of {len(data)} bytes exceeds HID_REPORT_SIZE "
                f"({self.HID_REPORT_SIZE}); callers must chunk"
            )
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
        """Turn the display panel on ("DIS").

        The device resets its backlight to BRIGHTNESS_DEFAULT when the
        screen returns, so the brightness last set via set_brightness()
        is re-applied. The LIG must follow DIS -- the reset is tied to
        the screen coming on -- and both are sent under one lock hold so
        nothing can interleave between them.
        """
        with self._lock:
            self._send_command(self.CMD_SCREEN_ON)
            self._send_command(self.CMD_BRIGHTNESS, bytes([self._brightness]))

    def set_brightness(self, percent: int | None = None) -> int:
        """Get, or set, the backlight brightness ("LIG"), 0-100 percent.

        With no argument, returns the current brightness and touches
        nothing. With one, LIG is sent under the lock and the value is
        remembered for screen_on() to re-apply (see screen_on).

        Returns the current brightness in every case.
        """
        if percent is None:
            return self._brightness
        if not self.BRIGHTNESS_MIN <= percent <= self.BRIGHTNESS_MAX:
            raise ValueError(
                f"brightness must be {self.BRIGHTNESS_MIN}-{self.BRIGHTNESS_MAX}"
            )
        with self._lock:
            self._brightness = percent
            self._send_command(self.CMD_BRIGHTNESS, bytes([percent]))
        return percent

    def heartbeat(self) -> None:
        """Send one keepalive heartbeat ("CONNECT") manually."""
        with self._lock:
            self._send_command(self.CMD_HEARTBEAT)

    def orientation(self, mode: Orientation | None = None) -> Orientation:
        """Get, or set, the display mode.

        With no argument, returns the current Orientation and touches
        nothing. With one, show() renders for it from the next call, and
        SET is sent under the lock so the device's splash matches -- that
        is the command's only effect (see Orientation). Wire layout is
        the standard 2-byte-gap pattern: "CRT" + 00,00 + "SET" + 00,00 +
        0x00 + value (unlike CLE/VER's exceptions).

        Set panel.inverted directly for an upside-down mount; it is
        software only, so it needs no method.

        Returns the current Orientation in every case.
        """
        if mode is None:
            return self._orientation
        if not isinstance(mode, Orientation):
            raise ValueError(
                f"mode must be an Orientation member (LANDSCAPE or "
                f"PORTRAIT), got {mode!r}"
            )
        with self._lock:
            # Sent first: a failed write must not leave the host rendering
            # for a mode the caller was told did not take.
            self._send_command(self.CMD_BOOT_ORIENTATION, bytes([0x00, mode.value]))
            self._orientation = mode
        return mode

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) the caller draws at, for the current mode.

        Handy as Image.new("RGB", panel.size), which stays right across
        an orientation() change. `inverted` does not affect it: a
        180-degree turn does not change the shape.
        """
        return self._screen_size()

    def clear(self) -> None:
        """Clear the screen ("CLE").

        Layout exception: a 3-byte gap plus a hardcoded 0xFF trailer
        (0xFF means "all" in the sibling multi-key products' clear
        command), not _send_command's 2-byte gap. Gap size is not
        universal across CRT commands -- verify each one.
        """
        with self._lock:
            self._write_report(b"CRT" + bytes(2) + b"CLE" + bytes(3) + bytes([0xFF]))

    def firmware_version(self) -> str:
        """Query the firmware version ("VER"). Confirmed on real hardware.

        Layout exception: a leading 0x00 device-context byte precedes
        "CRT" and there is no gap after the mnemonic. The response comes
        from get_input_report(), a GET_REPORT on the control endpoint
        (not an interrupt read, though the 512-byte size matches EP2
        IN's wMaxPacketSize); its first byte echoes the report ID.
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

        Unlike firmware_version(), no custom "CRT" protocol is involved:
        this is the standard USB iSerial descriptor string read via
        hidapi's get_serial_number_string(), the same value `lsusb -v`
        shows.
        """
        with self._lock:
            dev = self._dev
            if dev is None:
                raise RuntimeError("device is closed")
            return dev.get_serial_number_string()

    # -- images --

    def _screen_size(self, orientation: Orientation | None = None) -> tuple[int, int]:
        """(width, height) of the caller's space for an orientation.

        The JPEG sent is 462x1920 either way -- only the caller's view
        changes.
        """
        # Not `orientation or self._orientation`: Orientation.LANDSCAPE
        # is 0x00, i.e. falsy as an IntEnum.
        mode = orientation if orientation is not None else self._orientation
        if mode is Orientation.PORTRAIT:
            return self.PANEL_SHORT_SIDE, self.PANEL_LONG_SIDE
        return self.PANEL_LONG_SIDE, self.PANEL_SHORT_SIDE

    def _align_axis(self, value: int, extent: int, axis: str, reflected: bool) -> int:
        """Nudge `value` so the header's x field satisfies the color rule.

        `value` is the caller's coordinate on the 462-px axis;
        `reflected` says whether it reaches the header as
        (462 - value - extent) or unchanged. See SHORT_AXIS_ALIGN_* for
        the rule.

        Prefers the smaller shift, falls back to the other direction if
        that one leaves the panel, ties to the smaller coordinate. Warns
        on every correction, and warns and returns unchanged if neither
        fits: an occasional color glitch beats refusing to draw.
        stacklevel=4 reaches the user's show() call via _region_rect.

        A region spanning the whole axis can only sit at 0, which is the
        target residue, so it needs no special case.
        """
        header_x = (self.PANEL_SHORT_SIDE - value - extent) if reflected else value
        residue = header_x % self.SHORT_AXIS_ALIGN_MODULUS
        if residue == self.SHORT_AXIS_ALIGN_RESIDUE:
            return value

        # Solve in header space, then step `value` back through the sign
        # `reflected` implies.
        sign = -1 if reflected else 1
        plus = (self.SHORT_AXIS_ALIGN_RESIDUE - residue) % self.SHORT_AXIS_ALIGN_MODULUS
        candidate = next(
            c
            for _shift, c in sorted(
                (abs(s), value + sign * s)
                for s in (plus, plus - self.SHORT_AXIS_ALIGN_MODULUS)
            )
            if c >= 0 and c + extent <= self.PANEL_SHORT_SIDE
        )
        warnings.warn(
            f"akp02: region {axis}={value} shifted to "
            f"{axis}={candidate} (extent={extent}) for correct color "
            f"rendering -- see AKP02."
            f"SHORT_AXIS_ALIGN_MODULUS/SHORT_AXIS_ALIGN_RESIDUE",
            stacklevel=4,
        )
        return candidate

    def _to_buffer_rect(
        self,
        rect: _Rect,
        orientation: Orientation,
        inverted: bool,
    ) -> _Rect:
        """Map a rect from the caller's space into the portrait buffer.

        Puts the rect through exactly the net rotation _TRANSPOSE applies
        to the pixels, so a region lands where the full frame would put
        it. Change one without the other and full frames still look
        correct while every region is misplaced.
        """
        x, y, width, height = rect
        screen_w, screen_h = self._screen_size(orientation)
        if orientation is Orientation.LANDSCAPE:
            # ROTATE_270, or ROTATE_90 when inverted.
            if inverted:
                return _Rect(y, screen_w - x - width, height, width)
            return _Rect(screen_h - y - height, x, height, width)
        # Portrait is the buffer's own space: identity, or ROTATE_180.
        if inverted:
            return _Rect(screen_w - x - width, screen_h - y - height, width, height)
        return _Rect(x, y, width, height)

    def _region_rect(
        self,
        at: tuple[int, int],
        size: tuple[int, int],
        orientation: Orientation,
        inverted: bool,
    ) -> _Rect:
        """Bounds-check, align, and map a caller-space region rect.

        Shared by show()'s two paths so cached bytes land in exactly the
        rect their source image would have. `size` is caller space
        (pre-rotation); the rect comes back in buffer space, so landscape
        returns it with width and height swapped.
        """
        x, y = at
        width, height = size
        screen_w, screen_h = self._screen_size(orientation)
        if x < 0 or y < 0 or x + width > screen_w or y + height > screen_h:
            raise ValueError(
                f"region ({x},{y},{width}x{height}) does not "
                f"fit the {screen_w}x{screen_h} screen"
            )
        if orientation is Orientation.LANDSCAPE:
            y = self._align_axis(y, height, "y", reflected=not inverted)
        else:
            x = self._align_axis(x, width, "x", reflected=inverted)
        return self._to_buffer_rect(_Rect(x, y, width, height), orientation, inverted)

    @staticmethod
    def _region_size_from_jpeg(
        jpeg: bytes, orientation: Orientation
    ) -> tuple[int, int]:
        """Read a region's caller-space (width, height) from its own bytes.

        Region bytes arrive already rotated into buffer space, so this
        undoes _to_buffer_rect's axis mapping: landscape swaps the two,
        portrait is identity, and `inverted` is a 180 that leaves the
        mapping alone. Deriving rather than being told means the header
        cannot declare a size the bytes contradict. Image.open() stops
        at the JPEG header, so no pixel is decoded.
        """
        try:
            with Image.open(io.BytesIO(jpeg)) as probe:
                fmt, (width, height) = probe.format, probe.size
        except Exception as exc:
            raise ValueError(f"region bytes are not a readable image: {exc}") from exc
        if fmt != "JPEG":
            raise ValueError(f"region bytes must be JPEG, got {fmt}")
        if orientation is Orientation.LANDSCAPE:
            return height, width
        return width, height

    def _letterbox(
        self, img: Image.Image, orientation: Orientation | None = None
    ) -> Image.Image:
        """Scale to fit the mode's screen size, preserving aspect ratio.

        Centered on a black canvas of that size.
        """
        w, h = self._screen_size(orientation)
        scale = min(w / img.width, h / img.height)
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(img, ((w - new_size[0]) // 2, (h - new_size[1]) // 2))
        return canvas

    def _encode_jpeg(self, img: Image.Image) -> bytes:
        """JPEG-encode at the instance's quality and subsampling."""
        buf = io.BytesIO()
        img.save(
            buf,
            format="JPEG",
            quality=self.jpeg_quality,
            subsampling=self.jpeg_subsampling,
        )
        return buf.getvalue()

    def encode_region(self, image: Image.Image) -> bytes:
        """Encode a region exactly as show(image, at=...) would send it.

        Push the result with show(jpeg, at=...) as often as you like:
        the size comes from the bytes, so a cache entry is just
        (jpeg, at), and unchanged pixels cost no encode. Nothing here
        touches the handle or takes the lock, so encoding can run off
        the thread that owns the device. No full-screen counterpart:
        show() already letterboxes and encodes that case.

        The bytes carry the rotation the CURRENT orientation and
        `inverted` imply, so drop the cache when either changes. show()
        derives the size from whatever it is handed, so stale bytes are
        reinterpreted rather than rejected; pass size= to catch that.
        """
        img = image if image.mode == "RGB" else image.convert("RGB")
        transpose = _TRANSPOSE[(self._orientation, bool(self.inverted))]
        if transpose is not None:
            img = img.transpose(transpose)
        return self._encode_jpeg(img)

    def _prepare_image(
        self,
        image: Image.Image,
        at: tuple[int, int] | None,
        size: tuple[int, int] | None,
        orientation: Orientation,
        inverted: bool,
    ) -> tuple[bytes, _Rect]:
        """Fit or place, rotate, and encode a PIL image for show().

        Returns the JPEG and the buffer rect to send it with. Called
        outside the lock, so the encode never blocks the keepalive.
        """
        if size is not None:
            raise ValueError("size= is for raw JPEG bytes; an image has its own")
        img = image if image.mode == "RGB" else image.convert("RGB")
        rect = _FULL_SCREEN
        if at is None:
            if img.size != self._screen_size(orientation):
                img = self._letterbox(img, orientation)
        else:
            rect = self._region_rect(at, img.size, orientation, inverted)
        # transpose() (an exact permutation) rather than rotate().
        transpose = _TRANSPOSE[(orientation, inverted)]
        if transpose is not None:
            img = img.transpose(transpose)
        return self._encode_jpeg(img), rect

    def _prepare_bytes(
        self,
        jpeg: bytes,
        at: tuple[int, int] | None,
        size: tuple[int, int] | None,
        orientation: Orientation,
        inverted: bool,
    ) -> tuple[bytes, _Rect]:
        """Place ready JPEG bytes for show(), returning them untouched.

        A region is sized from the JPEG's own header, with size= checked
        against that if the caller supplied it.
        """
        if at is None:
            if size is not None:
                raise ValueError("size= requires at=; full-screen bytes are whole")
            return jpeg, _FULL_SCREEN
        derived = self._region_size_from_jpeg(jpeg, orientation)
        if size is not None and tuple(size) != derived:
            raise ValueError(
                f"size={size[0]}x{size[1]} disagrees with the region "
                f"JPEG, which is {derived[0]}x{derived[1]} in the "
                f"current mode's coordinates; the usual causes are "
                f"bytes never rotated into buffer space (use "
                f"encode_region()) and a cache reused after an "
                f"orientation() change"
            )
        return jpeg, self._region_rect(at, derived, orientation, inverted)

    def show(
        self,
        image: Image.Image | bytes,
        at: tuple[int, int] | None = None,
        size: tuple[int, int] | None = None,
    ) -> None:
        """Display an image on the panel.

        Accepts a PIL image in the panel's active mode (orientation() --
        LANDSCAPE: 1920x462 space, PORTRAIT: 462x1920 space), or ready
        JPEG bytes of the portrait buffer, sent untouched: the whole
        462x1920 for a full-screen draw, or one region's worth with at=.

        at=None: full-screen, letterboxed if not exactly that size.
        at=(x, y): partial update there, sized by the image; the rest of
        the screen is preserved. Regions are nudged a few pixels along
        the 462-px axis when the color-alignment rule requires it,
        warning when they are (see SHORT_AXIS_ALIGN_*). The other
        coordinate and the size never change.

        size=(width, height) is optional and only cross-checks, since a
        region's dimensions are read from the JPEG's own header. Pass it
        to assert the caller-space shape you expect -- the one thing
        deriving cannot do, as unrotated or stale bytes look just like
        correct bytes for a different shape. Produce region bytes with
        encode_region().

        PIL input is rotated (_TRANSPOSE) and JPEG-encoded before the
        lock, so the keepalive isn't blocked; a region's rect goes
        through the same rotation, so it lands where the full frame
        would put it. Raw JPEG bytes are never transformed, which is why
        region bytes must arrive already rotated. The lock then holds
        for the whole header + chunks + commit so no report can
        interleave, and the FULL_TO_REGION_SETTLE_SEC delay is applied
        automatically where it is needed.
        """
        is_region = at is not None
        # Snapshotted once: every geometry decision below must come from
        # the same state, or a concurrent orientation() could rotate the
        # pixels one way and place their rect the other.
        orientation, inverted = self._orientation, bool(self.inverted)
        args = (at, size, orientation, inverted)
        if isinstance(image, Image.Image):
            jpeg, rect = self._prepare_image(image, *args)
        else:
            jpeg, rect = self._prepare_bytes(image, *args)
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
        returns immediately for 0 or less, so a bad interval would
        become a tight loop writing to the device rather than an error.
        """
        if interval_sec <= 0:
            raise ValueError(
                f"keepalive interval must be greater than 0 seconds, got "
                f"{interval_sec!r}; pass keepalive_interval=None to AKP02() "
                f"to not start one automatically"
            )

    def start_keepalive(self, interval_sec: float = KEEPALIVE_INTERVAL_SEC) -> None:
        """Start a daemon thread sending a heartbeat every interval_sec.

        No-op if already running. Thread-safe: concurrent calls can't
        spawn two threads. __enter__ calls this unless
        AKP02(keepalive_interval=None) was passed; call it directly for
        a different interval, or to resume after a disconnect.

        Raises ValueError for a non-positive interval_sec, before the
        no-op check, so a bad value is reported either way.

        The guard tests is_alive() rather than "is not None" because a
        keepalive thread that lost the device exits on its own, leaving
        a dead Thread object behind; treating that as "already running"
        would no-op every restart for the life of the object, including
        a legitimate resume. The dead thread is dropped here rather than
        cleared from inside loop(), which would need the thread to take
        _keepalive_mgmt_lock -- held by stop_keepalive across its join().
        """
        self._check_keepalive_interval(interval_sec)
        with self._keepalive_mgmt_lock:
            existing = self._keepalive_thread
            if existing is not None and existing.is_alive():
                return
            stop = self._keepalive_stop = threading.Event()

            def loop() -> None:
                # The try wraps the whole loop rather than each beat: the
                # handler ends the thread either way, so the two are
                # equivalent, and this keeps the hot path shallower.
                try:
                    while not stop.wait(interval_sec):
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
        a leaked one can't block interpreter exit.
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
