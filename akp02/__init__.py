"""Linux library for the Ajazz AKP02 (9.2" 1920x462) USB HID display.

The device sleeps without heartbeats, so `with` starts a keepalive
thread by default; AKP02(keepalive_interval=None) leaves it to the
caller.

By default the panel runs in the landscape (1920x462) layout it is
sold as: show() takes landscape input, rotates it into the 462x1920
portrait buffer the device expects, and treats at=(x, y) as landscape
coordinates. panel.orientation(Orientation.PORTRAIT) switches it to
portrait (462x1920), after which show() takes portrait input unrotated
and at=(x, y) as buffer coordinates. Setting panel.inverted turns
either mode a further 180 degrees, for a panel mounted the other way
up. The JPEG sent is 462x1920 in every case (see show()).

Protocol (reverse-engineered; details in the README, "Protocol notes
(reverse engineered)"): commands are "CRT" + 00 00 + <mnemonic> +
00 00 + <params> (AKP02.CMD_* for the mnemonics), and image transfers
are a 32-byte CRTDRA header (AKP02._crtdra_header) + JPEG, chunked
into 1024-byte reports, then a commit (STP). The JPEG is always
462x1920 PORTRAIT; landscape's 90 degrees clockwise is the
hardware-confirmed rotation.
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

# Single source of truth: pyproject declares `dynamic = ["version"]` and
# hatchling reads this line at build time, so there is no second copy to
# forget to bump.
__version__ = "1.1.0"


class Orientation(IntEnum):
    """Panel orientation, expressed as the byte the SET command sends.

    LANDSCAPE is the 1920x462 layout the panel is sold as; PORTRAIT
    turns the glass 90 degrees, so the caller sees 462x1920. Neither is
    the transfer's own orientation -- that is always the 462x1920
    buffer, whichever way the glass is hung. The member value *is* the
    wire byte, so there is no second copy to keep in step.

    Confirmed on real hardware: SET only sets which way up the device
    draws its own power-on splash, and persists that (verified by
    unplugging and replugging after setting each value). It does not
    rotate frames the host sends -- nothing on the device does, which is
    why show() rotates them here.
    """

    LANDSCAPE = 0x00
    PORTRAIT = 0x01


# The single net rotation show() applies, per (orientation, inverted).
# One entry per case, so the "inverted" mount is a rotation and not the
# reflection a flip-then-rotate composes to: a reflection reverses
# chirality, and text on the panel would read backwards.
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
    type stubs; anything with these five methods works, which is also
    the contract a test double passed as AKP02(dev=...) must satisfy.
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

    # Every attribute an instance may have. Without this, the class
    # accepts any assignment, so a misspelled `panel.inverted` binds a
    # dead attribute in silence and the panel goes on doing the default
    # thing -- the failure being a frame that looks wrong, with nothing
    # pointing at the line that caused it. `inverted` is the one settable
    # knob with no method behind it, so that is exactly where the mistake
    # lands.
    #
    # Subclasses are unaffected: one without its own __slots__ still gets
    # a __dict__ and stays open.
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
    )

    VENDOR_ID = 0x0300
    PRODUCT_ID = 0x3017

    # Named for the glass, not for an orientation: which one is "width"
    # depends on the mode, but the strip is 1920 along its long edge
    # however you hang it. The buffer sent is always SHORT x LONG.
    # _screen_size() is what gives the caller's width and height.
    PANEL_LONG_SIDE = 1920
    PANEL_SHORT_SIDE = 462

    HID_REPORT_SIZE = 1024  # EP1 OUT wMaxPacketSize from the device descriptor
    INPUT_REPORT_SIZE = 512  # EP2 IN wMaxPacketSize
    JPEG_QUALITY = 85  # default; override per instance via __init__
    JPEG_QUALITY_MIN = 1
    # Pillow advises <= 95; above it, size grows for almost no visual gain.
    JPEG_QUALITY_MAX = 95
    BRIGHTNESS_MIN = 0  # range of the LIG command's parameter byte
    BRIGHTNESS_MAX = 100
    # The device's factory default: observed on real hardware to revert
    # its backlight to this value after an off->on cycle, which is what
    # screen_on() re-applies.
    BRIGHTNESS_DEFAULT = 80

    # A region update sent immediately after a full-frame draw can stop
    # the full frame rendering at all (confirmed on real hardware: 4ms
    # suffices, 0ms fails). Only this transition needs it --
    # region-after-region and full-after-full do not. Likely cause: a
    # full draw is a clean buffer replace, but a region is a
    # read-modify-write against the framebuffer; if that read starts
    # before the previous commit has finished settling internally (not
    # just "USB bytes received"), the in-flight commit can get
    # corrupted. 20ms here (5x the confirmed 4ms) for jitter margin; it
    # fires once per full-frame draw, not per region, so being generous
    # costs nothing.
    FULL_TO_REGION_SETTLE_SEC = 0.02

    # Confirmed on real hardware: a region update renders with the wrong
    # color unless whatever lands in the CRTDRA header's x field
    # satisfies this residue mod this modulus. Root cause understood,
    # not just observed: 462 does not divide evenly into 8- or 16-pixel
    # JPEG blocks the way 1920 does, so the firmware evidently
    # pads/rounds its buffer on that axis with a fixed internal offset.
    # The 1920-pixel axis shows no equivalent sensitivity -- confirmed by
    # testing it directly.
    #
    # A property of the buffer, not of the caller's coordinates: the rule
    # binds whichever coordinate _to_buffer_rect maps into the header's
    # x, which is y in landscape and x in portrait, running with or
    # against it depending on `inverted`. show() corrects this
    # automatically (see _align_axis); the constants exist so the target
    # is explicit and updatable in one place.
    SHORT_AXIS_ALIGN_MODULUS = 8
    SHORT_AXIS_ALIGN_RESIDUE = 2

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
    ) -> None:
        """Open the panel.

        Pass an already-open hidapi device (or any _HidDevice-shaped
        object, e.g. a test double) via dev to skip discovery.
        jpeg_quality (1-95) overrides the default encoding quality --
        small text UIs may want it higher.

        keepalive_interval is the interval in seconds __enter__ starts
        the keepalive thread with, or None to not start one. Only
        recorded here -- no thread is started by __init__ (see
        __enter__).

        The display mode starts at Orientation.LANDSCAPE; read or change
        it with orientation(). Nothing is sent to the device here.

        inverted (default False, set on the instance after construction)
        turns the image a further 180 degrees, for a panel mounted the
        other way up. Software only; no hardware command (see show()).

        Raises DeviceNotFoundError if the panel isn't connected.
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
        if keepalive_interval is not None:
            self._check_keepalive_interval(keepalive_interval)
        self.jpeg_quality: int = (
            self.JPEG_QUALITY if jpeg_quality is None else jpeg_quality
        )
        self._orientation: Orientation = Orientation.LANDSCAPE
        # Tracks the last host-requested brightness so screen_on() can
        # re-apply it (the device reverts to its default when the
        # screen returns).
        self._brightness: int = self.BRIGHTNESS_DEFAULT
        # Extra 180-degree rotation for an inverted mount (see show()).
        self.inverted: bool = False
        self._dev: _HidDevice | None = dev if dev is not None else self._open()
        self._lock = threading.Lock()
        # Guards _keepalive_thread/_keepalive_stop management only. Separate
        # from _lock so stop_keepalive() never holds a lock the keepalive
        # thread needs (heartbeat() takes _lock), which could deadlock under
        # specific timing: join() while the thread is blocked in heartbeat().
        self._keepalive_mgmt_lock = threading.Lock()
        self._keepalive_stop: threading.Event | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._keepalive_interval: float | None = keepalive_interval
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
            raise DeviceNotFoundError(
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
        start_keepalive() for you; AKP02(keepalive_interval=None) opts out.

        Here rather than in __init__ because the keepalive closure holds
        a strong reference to self (an auto-started panel never closed
        could never be collected, and would keep writing), a subclass
        would see heartbeats before its own __init__ finished, and a
        dev= test double would get a background writer just by being
        constructed. It also pairs with __exit__ -> close().

        No SET is sent here: the splash orientation is persisted device
        state, so pushing this instance's default at every `with` would
        overwrite a setting the caller never mentioned. It is
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
        touching the handle, so the close can't land mid-transfer: any
        thread already inside show() finishes its header + chunks +
        commit first, and any call that starts afterwards sees a closed
        device and raises RuntimeError. Closing under an active writer
        would be worse than a logic error -- libhidapi frees the handle,
        so an in-flight write() in another thread is a use-after-free.

        The lock acquisition is bounded by timeout_sec for the same
        reason stop_keepalive's join is: a write wedged against a
        hung device must not hang the caller forever. The two waits are
        sequential, so a fully wedged device can take up to twice
        timeout_sec to return. On timeout the
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
        if len(data) > self.HID_REPORT_SIZE:
            # Otherwise reaches bytes() with a negative count below and
            # fails as an unexplained "negative count".
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
        """Turn the display panel on ("DIS")."""
        with self._lock:
            self._send_command(self.CMD_SCREEN_ON)

    def set_brightness(self, percent: int | None = None) -> int:
        """Get, or set, the backlight brightness ("LIG"), 0-100 percent.

        With no argument, returns the current brightness and touches
        nothing. With one, LIG is sent under the lock and the value is
        remembered so screen_on() can re-apply it -- the device reverts
        its backlight to BRIGHTNESS_DEFAULT when the screen returns, so
        without this an off/on cycle would leave it bright again.

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
        is the command's only effect (see the Orientation docstring).
        Wire layout is "CRT" + 00,00 + "SET" + 00,00 + 0x00 + value, the
        standard 2-byte-gap pattern (unlike CLE/VER's exceptions).

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
        180-degree turn does not change the surface's shape.
        """
        return self._screen_size()

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

    def _screen_size(self, orientation: Orientation | None = None) -> tuple[int, int]:
        """(width, height) of the caller's space for an orientation.

        Landscape is the 1920x462 the panel is sold as; portrait is the
        same glass turned 90 degrees, so the caller sees 462x1920. The
        JPEG sent is 462x1920 either way -- only the caller's view
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
        `reflected` says whether it reaches the header against
        (462 - value - extent) or with it. See SHORT_AXIS_ALIGN_* for
        the rule.

        Prefers the smaller shift, falls back to the other direction if
        the preferred one leaves the panel, ties to the smaller
        coordinate. Warns on every correction, and warns and returns
        unchanged if neither fits: an occasional color glitch beats
        refusing to draw.

        Confirmed on real hardware: a region spanning the whole axis
        (extent == PANEL_SHORT_SIDE, so it can only sit at 0) needs no
        correction -- with no partial remainder there is nothing for the
        device's edge-padding to misalign. Without the check the search
        below would leave it alone anyway, but warn every time.
        """
        if extent == self.PANEL_SHORT_SIDE:
            return value

        header_x = (self.PANEL_SHORT_SIDE - value - extent) if reflected else value
        residue = header_x % self.SHORT_AXIS_ALIGN_MODULUS
        if residue == self.SHORT_AXIS_ALIGN_RESIDUE:
            return value

        # Solve in header space, then step `value` back through the sign
        # `reflected` implies.
        sign = -1 if reflected else 1
        plus = (self.SHORT_AXIS_ALIGN_RESIDUE - residue) % self.SHORT_AXIS_ALIGN_MODULUS
        for _shift, candidate in sorted(
            (abs(s), value + sign * s)
            for s in (plus, plus - self.SHORT_AXIS_ALIGN_MODULUS)
        ):
            if candidate >= 0 and candidate + extent <= self.PANEL_SHORT_SIDE:
                warnings.warn(
                    f"akp02: region {axis}={value} shifted to "
                    f"{axis}={candidate} (extent={extent}) for correct color "
                    f"rendering -- see AKP02."
                    f"SHORT_AXIS_ALIGN_MODULUS/SHORT_AXIS_ALIGN_RESIDUE",
                    stacklevel=3,
                )
                return candidate

        warnings.warn(
            f"akp02: region {axis}={value} (extent={extent}) cannot be "
            f"shifted to a color-safe position without leaving the panel -- "
            f"drawing uncorrected; this region's color may render incorrectly",
            stacklevel=3,
        )
        return value

    def _to_buffer_rect(
        self,
        rect: _Rect,
        orientation: Orientation,
        inverted: bool,
    ) -> _Rect:
        """Map a rect from the caller's space into the portrait buffer.

        Puts the rect through exactly the net rotation _TRANSPOSE applies
        to the pixels, so a region lands where the full frame would put
        it. Changing one without the other leaves full frames looking
        correct while silently misplacing every region.
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
        """JPEG-encode at the instance's quality (see show())."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()

    def show(
        self, image: Image.Image | bytes, at: tuple[int, int] | None = None
    ) -> None:
        """Display an image on the panel.

        Accepts a PIL image in the panel's active mode (orientation() --
        LANDSCAPE: 1920x462 space, PORTRAIT: 462x1920 space), or ready
        JPEG bytes of the 462x1920 portrait buffer (sent untouched;
        full-screen only).

        at=None: full-screen, letterboxed if not exactly that size.
        at=(x, y): partial update there, sized by the image; the rest of
        the screen is preserved. The device renders a region's color
        wrong unless the value landing in the header's x field satisfies
        a confirmed alignment requirement (see SHORT_AXIS_ALIGN_*); this
        is corrected automatically, nudging by a few pixels along the
        462-px axis -- y in landscape, x in portrait -- and warning when
        it does. The other coordinate and the size never change.

        PIL input is JPEG-encoded before the lock, so the keepalive
        thread isn't blocked. It first gets the one net rotation its
        mode calls for (_TRANSPOSE), and a region's rect goes through
        the same rotation, so it lands where the full frame would put
        it. Raw JPEG bytes are never transformed. The lock then holds
        for the whole header + chunks + commit sequence so no report can
        interleave.

        A region update sent immediately after a full-frame draw needs a
        brief settling delay first, or the full frame can fail to render
        at all -- see FULL_TO_REGION_SETTLE_SEC. This is tracked and
        applied automatically; callers don't need to do anything.
        """
        is_region = at is not None
        rect = _FULL_SCREEN
        # Snapshotted once: every geometry decision below has to come from
        # the same state, or a concurrent orientation() could rotate the
        # pixels one way and place their rect the other.
        orientation, inverted = self._orientation, bool(self.inverted)
        screen_w, screen_h = self._screen_size(orientation)
        if isinstance(image, Image.Image):
            img = image if image.mode == "RGB" else image.convert("RGB")
            if at is None:
                if img.size != (screen_w, screen_h):
                    img = self._letterbox(img, orientation)
            else:
                x, y = at
                if (
                    x < 0
                    or y < 0
                    or x + img.width > screen_w
                    or y + img.height > screen_h
                ):
                    raise ValueError(
                        f"region ({x},{y},{img.width}x{img.height}) does not "
                        f"fit the {screen_w}x{screen_h} screen"
                    )
                # Which coordinate reaches the header's x field, and
                # whether it runs with or against it, both follow from
                # the net rotation below.
                if orientation is Orientation.LANDSCAPE:
                    y = self._align_axis(y, img.height, "y", reflected=not inverted)
                else:
                    x = self._align_axis(x, img.width, "x", reflected=inverted)
                rect = self._to_buffer_rect(
                    _Rect(x, y, img.width, img.height), orientation, inverted
                )
            # transpose() (an exact permutation) rather than rotate().
            transpose = _TRANSPOSE[(orientation, inverted)]
            if transpose is not None:
                img = img.transpose(transpose)
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
                f"{interval_sec!r}; pass keepalive_interval=None to AKP02() "
                f"to not start one automatically"
            )

    def start_keepalive(self, interval_sec: float = KEEPALIVE_INTERVAL_SEC) -> None:
        """Start a daemon thread sending a heartbeat every interval_sec.

        No-op if already running. Thread-safe: concurrent calls can't
        spawn two threads. __enter__ calls this for you unless
        AKP02(keepalive_interval=None) was passed; calling it directly is
        still supported -- for a different interval, or to resume after a
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
                # The try wraps the whole loop rather than each beat: the
                # handler ends the thread either way, so the two are
                # equivalent here, and this keeps the hot path one level
                # shallower.
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
