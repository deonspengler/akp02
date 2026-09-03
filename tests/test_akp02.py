"""Test suite for the akp02 library.

Runs entirely against fake HID devices -- no physical AKP02 and no
hidapi installation required, safe to run in CI. Covers protocol
byte-exactness (pinned against real captured device traffic), image
encoding, input validation, error handling, and concurrency behavior.

Run with:
    pip install pytest pillow
    pytest test_akp02.py -v
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import re
import struct
import sys
import threading
import time
import types
import warnings

import pytest
from PIL import Image, ImageChops

import akp02
from akp02 import _FULL_SCREEN, AKP02, DeviceNotFoundError, Orientation, _Rect

# ---------------------------------------------------------------------
# Fake devices and helpers
# ---------------------------------------------------------------------

class FakeDevice:
    """Satisfies AKP02's _HidDevice protocol without touching hardware.

    Writes are recorded under a lock because several tests drive this
    from the keepalive thread and the main thread at once.
    """

    def __init__(self):
        self.writes: list[bytes] = []
        self.input_report_calls: list[tuple[int, int]] = []
        self.closed = False
        self.fail_after: int | None = None  # None = never fail
        self._lock = threading.Lock()

    def write(self, data: bytes) -> int:
        with self._lock:
            if self.fail_after is not None and len(self.writes) >= self.fail_after:
                return -1
            self.writes.append(bytes(data))
        return len(data)

    def get_input_report(self, report_id: int, size: int):
        self.input_report_calls.append((report_id, size))
        return list(bytes([report_id]) + b"1.0.2-test" + bytes(size - 11))

    def get_serial_number_string(self) -> str:
        return "C511D378553A-test"

    def error(self) -> str:
        return "simulated error"

    def close(self) -> None:
        self.closed = True


def payload_of(writes: list[bytes]) -> tuple[bytes, bytes]:
    """Reassemble an image transfer from the reports it was chunked into.

    Strips each report's leading hidapi report-ID placeholder, rejoins
    the stream, and slices it using the header's own declared total
    length -- so the trailing zero padding of the final chunk is
    excluded. Returns (header, jpeg).
    """
    stream = b"".join(w[1:] for w in writes)
    header = stream[:32]
    total = struct.unpack(">I", header[8:12])[0]  # payload + 0x20 header
    return header, stream[32:total]


def decode_sent_image(writes: list[bytes]) -> Image.Image:
    """Decode the JPEG the library actually put on the wire."""
    _, jpeg = payload_of(writes)
    return Image.open(io.BytesIO(jpeg))


def assert_close(actual, expected, tol=40):
    """Compare a JPEG-decoded pixel to an expected color, lossily."""
    assert all(abs(a - e) <= tol for a, e in zip(actual, expected, strict=True)), \
        f"pixel {actual} is not within {tol} of {expected}"


def alive_keepalives(before: set[int | None]) -> list[threading.Thread]:
    """Keepalive threads alive now that weren't alive at `before` time.

    Compared against a snapshot rather than a global absolute count so
    that a leftover thread from another test can't cause a false failure
    depending on test ordering.
    """
    return [t for t in threading.enumerate()
            if t.name == "akp02-keepalive" and t.ident not in before
            and t.is_alive()]


@pytest.fixture
def fake_dev():
    return FakeDevice()


@pytest.fixture
def make_panel():
    """Build panels on custom devices, guaranteeing cleanup.

    Tests needing a device other than plain FakeDevice used to construct
    AKP02 directly and leak it; this closes them all, quietly (a
    deliberately wedged device warns on close, which is not the failure
    those tests are about).
    """
    created = []

    def _make(dev, **kwargs) -> AKP02:
        p = AKP02(dev=dev, **kwargs)
        created.append(p)
        return p

    yield _make
    for p in created:
        with warnings.catch_warnings(), contextlib.suppress(Exception):
            warnings.simplefilter("ignore")
            p.close(timeout_sec=0.2)


@pytest.fixture
def panel(fake_dev, make_panel):
    return make_panel(fake_dev)


class FakeClock:
    """Stands in for the library's `time` module, recording sleeps.

    The settling-delay tests used to assert on wall-clock elapsed time,
    but a show() call also spends real time encoding a JPEG -- measured
    here at 3-13ms for a full frame, against a 10ms budget. That margin
    disappears on a loaded CI runner, making those tests flaky for a
    reason having nothing to do with the delay logic. Recording the
    sleep instead tests the actual intent, exactly and instantly.
    Patched onto the akp02 module only, so the real clock is untouched
    everywhere else.
    """

    def __init__(self):
        self.sleeps = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def __getattr__(self, name):  # perf_counter etc. pass through
        return getattr(time, name)


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(akp02, "time", fake)
    return fake


@pytest.fixture
def fake_hid(monkeypatch):
    """Inject a stub `hid` module so discovery is testable without hidapi.

    AKP02._open does `import hid` at call time, so putting a module
    object in sys.modules is enough. The old test imported the real hid
    and errored outright when hidapi wasn't installed -- which is the
    normal state of a CI box for a suite that otherwise needs no
    hardware and says so in its own docstring.
    """
    module = types.ModuleType("hid")
    module.enumerate = lambda: []

    opened_paths: list[bytes] = []

    class StubHidDevice:
        def open_path(self, path):
            opened_paths.append(path)

        def write(self, data):
            return len(data)

        def close(self):
            pass

    module.device = StubHidDevice
    module.opened_paths = opened_paths
    monkeypatch.setitem(sys.modules, "hid", module)
    return module


# ---------------------------------------------------------------------
# Protocol byte-exactness. These pin the wire format so a refactor can't
# silently change it. The expected bytes below are recorded from the
# reverse-engineered protocol the library documents, not derived from a
# published specification -- so if one of these fails, suspect the change
# rather than the test.
# ---------------------------------------------------------------------

class TestProtocolBytes:
    def test_crtdra_header_matches_real_capture(self):
        header = AKP02._crtdra_header(85459, _FULL_SCREEN)
        expected = bytes.fromhex(
            "435254000044524100014df3b1"
            "00000000000000000000000000000000000000"
        )
        assert header == expected

    def test_crtdra_header_region_field_order(self):
        # The full-screen capture above is all zeros in the geometry
        # fields, so it cannot catch a width/height/x/y transposition --
        # the single easiest thing in this header to break. Pin a
        # non-zero rect to literal bytes too.
        header = AKP02._crtdra_header(0x10, _Rect(x=0x1122, y=0x3344,
                                                  width=0x5566,
                                                  height=0x7788))
        assert len(header) == 32
        assert header[8:12] == bytes.fromhex("00000030")  # 0x10 + 0x20
        assert header[12] == 0xB1
        assert header[13:21] == bytes.fromhex("5566778811223344")
        assert header[21:32] == bytes(11)

    def test_screen_off_bytes(self, panel, fake_dev):
        panel.screen_off()
        assert fake_dev.writes[-1][1:9] == b"CRT\x00\x00HAN"

    def test_screen_on_bytes(self, panel, fake_dev):
        panel.screen_on()
        assert fake_dev.writes[-1][1:9] == b"CRT\x00\x00DIS"

    def test_brightness_bytes(self, panel, fake_dev):
        panel.set_brightness(75)
        expected = bytes([0x43, 0x52, 0x54, 0x00, 0x00,
                          0x4c, 0x49, 0x47, 0x00, 0x00, 75])
        assert fake_dev.writes[-1][1:1 + len(expected)] == expected

    def test_brightness_reads_back_without_touching_the_device(self, panel,
                                                                fake_dev):
        # No argument reads the tracked value. 80 is BRIGHTNESS_DEFAULT,
        # the device's factory default the tracking starts from, so a
        # first screen_on() re-applies a no-op rather than a value the
        # caller never asked for.
        assert panel.set_brightness() == 80
        assert fake_dev.writes == []
        panel.set_brightness(10)
        fake_dev.writes.clear()
        assert panel.set_brightness() == 10
        assert fake_dev.writes == []

    def test_clear_bytes(self, panel, fake_dev):
        # NOT the standard 2-byte-gap pattern -- 3-byte gap plus a
        # hardcoded 0xFF trailer, per the layout exception documented on
        # AKP02.clear. Gap size is not universal across CRT commands.
        panel.clear()
        expected = b"CRT" + bytes(2) + b"CLE" + bytes(3) + bytes([0xFF])
        assert fake_dev.writes[-1][1:1 + len(expected)] == expected

    def test_heartbeat_bytes(self, panel, fake_dev):
        panel.heartbeat()
        assert fake_dev.writes[-1][1:13] == b"CRT\x00\x00CONNECT"

    def test_orientation_portrait_matches_real_capture(self, panel, fake_dev):
        # Confirmed on real hardware: switching horizontal->vertical
        # produced exactly this SET command.
        panel.orientation(Orientation.PORTRAIT)
        expected = bytes.fromhex("4352540000534554000000010000")
        assert fake_dev.writes[-1][1:1 + len(expected)] == expected

    def test_orientation_landscape_matches_real_capture(self, panel, fake_dev):
        # Confirmed on real hardware: switching vertical->horizontal
        # produced exactly this SET command.
        panel.orientation(Orientation.LANDSCAPE)
        expected = bytes.fromhex("4352540000534554000000000000")
        assert fake_dev.writes[-1][1:1 + len(expected)] == expected

    def test_orientation_rejects_a_non_member(self, panel, fake_dev):
        # The enum member's value IS the wire byte, so a bare int would
        # otherwise sail through and send something unvalidated.
        with pytest.raises(ValueError, match="Orientation member"):
            panel.orientation(1)
        assert fake_dev.writes == []

    def test_orientation_reads_back_without_touching_the_device(self, panel,
                                                                fake_dev):
        assert panel.orientation() is Orientation.LANDSCAPE
        panel.orientation(Orientation.PORTRAIT)
        fake_dev.writes.clear()
        assert panel.orientation() is Orientation.PORTRAIT
        assert fake_dev.writes == []

    def test_commit_bytes_end_every_transfer(self, panel, fake_dev):
        # The commit is what actually makes the device render; nothing
        # previously asserted it was sent at all.
        panel.show(Image.new("RGB", (1920, 462)))
        assert fake_dev.writes[-1][1:9] == b"CRT\x00\x00STP"

    def test_firmware_version_request_bytes(self, panel, fake_dev):
        # Documented layout exception: a leading 0x00 device-context
        # byte before "CRT", and no gap after the mnemonic. Worth
        # pinning precisely because it is the odd one out.
        panel.firmware_version()
        assert fake_dev.writes[-1][:10] == b"\x00\x00CRT\x00\x00VER"

    def test_firmware_version_query_and_parse(self, panel, fake_dev):
        assert panel.firmware_version() == "1.0.2-test"
        assert fake_dev.input_report_calls == [
            (0x00, AKP02.INPUT_REPORT_SIZE + 1)]

    def test_firmware_version_tolerates_non_ascii(self, panel, fake_dev):
        fake_dev.get_input_report = lambda rid, size: list(
            bytes([rid]) + b"1.0\xff2" + bytes(size - 6))
        assert panel.firmware_version().startswith("1.0")  # no decode crash

    def test_serial_number_returns_hidapi_value(self, panel):
        # Not part of the custom CRT protocol at all -- just delegates
        # to hidapi's own get_serial_number_string().
        assert panel.serial_number() == "C511D378553A-test"

    def test_serial_number_writes_no_wire_report(self, panel, fake_dev):
        # Confirms this is a pure hidapi passthrough with no custom
        # protocol command sent -- unlike firmware_version(), which
        # writes a "VER" request first.
        panel.serial_number()
        assert fake_dev.writes == []

    @pytest.mark.parametrize("call", [
        lambda p: p.screen_off(),
        lambda p: p.screen_on(),
        lambda p: p.set_brightness(50),
        lambda p: p.heartbeat(),
        lambda p: p.clear(),
        lambda p: p.firmware_version(),
        lambda p: p.show(Image.new("RGB", (1920, 462))),
    ])
    def test_every_report_is_exactly_one_hid_report(self, panel, fake_dev,
                                                    call):
        # One report = HID_REPORT_SIZE payload bytes + hidapi's leading
        # report-ID placeholder. A short or oversized report is a wire
        # protocol violation regardless of whether its prefix is right,
        # and the prefix assertions above cannot see it.
        call(panel)
        assert fake_dev.writes
        assert all(len(w) == AKP02.HID_REPORT_SIZE + 1
                   for w in fake_dev.writes)

    def test_command_reports_are_zero_padded(self, panel, fake_dev):
        panel.set_brightness(75)
        assert fake_dev.writes[-1][12:] == bytes(AKP02.HID_REPORT_SIZE - 11)


LANDSCAPE, PORTRAIT = Orientation.LANDSCAPE, Orientation.PORTRAIT


class TestRegionMath:
    def test_full_frame_maps_to_full_portrait_buffer(self, panel):
        assert panel._to_buffer_rect(_Rect(0, 0, 1920, 462),
                                     LANDSCAPE, False) == (0, 0, 462, 1920)

    def test_arbitrary_region_transform(self, panel):
        assert panel._to_buffer_rect(_Rect(100, 50, 300, 150),
                                     LANDSCAPE, False) == (262, 100, 150, 300)

    def test_transform_matches_the_documented_pixel_mapping(self, panel):
        # The rect transform and the image rotation must agree or regions
        # land in the wrong place. Check the documented clockwise
        # (x, y) -> (H-1-y, x) mapping on a 1x1 rect at the far corner.
        rect = panel._to_buffer_rect(_Rect(1919, 461, 1, 1), LANDSCAPE, False)
        assert (rect.x, rect.y) == (0, 1919)


# ---------------------------------------------------------------------
# Device discovery. Runs against a stub `hid` module -- no hidapi needed.
# ---------------------------------------------------------------------

class TestDiscovery:
    def test_device_not_found_raises(self, fake_hid):
        with pytest.raises(DeviceNotFoundError):
            AKP02()

    def test_not_found_message_lists_candidates(self, fake_hid):
        fake_hid.enumerate = lambda: [
            {"vendor_id": 0x1234, "product_id": 0x5678,
             "product_string": "Some Keyboard", "path": b"/dev/x"}]
        with pytest.raises(DeviceNotFoundError, match="1234:5678"):
            AKP02()

    def test_right_vendor_wrong_product_says_so(self, fake_hid):
        # A different failure mode from "nothing plugged in" -- likely a
        # sibling Ajazz device or a firmware revision -- and the library
        # takes trouble to distinguish it.
        fake_hid.enumerate = lambda: [
            {"vendor_id": AKP02.VENDOR_ID, "product_id": 0x9999,
             "product_string": "Ajazz Other", "path": b"/dev/x"}]
        with pytest.raises(DeviceNotFoundError, match="wrong product"):
            AKP02()

    def test_wrong_hid_package_raises_actionable_importerror(self, fake_hid):
        del fake_hid.device
        with pytest.raises(ImportError, match="hidapi"):
            AKP02()

    def test_lowest_interface_number_is_chosen(self, fake_hid):
        fake_hid.enumerate = lambda: [
            {"vendor_id": AKP02.VENDOR_ID, "product_id": AKP02.PRODUCT_ID,
             "interface_number": 2, "path": b"/dev/iface2"},
            {"vendor_id": AKP02.VENDOR_ID, "product_id": AKP02.PRODUCT_ID,
             "interface_number": 0, "path": b"/dev/iface0"},
        ]
        AKP02()
        assert fake_hid.opened_paths == [b"/dev/iface0"]


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

class TestValidation:
    @pytest.mark.parametrize("value", [-1, 101, 1000])
    def test_brightness_rejects_out_of_range(self, panel, value):
        with pytest.raises(ValueError):
            panel.set_brightness(value)

    @pytest.mark.parametrize("value", [0, 50, 100])
    def test_brightness_accepts_in_range(self, panel, value):
        panel.set_brightness(value)  # should not raise

    def test_brightness_out_of_range_leaves_tracked_state_unchanged(
            self, panel):
        # Validation must precede the state update: a rejected value can't
        # become the one screen_on() later re-applies.
        panel.set_brightness(10)
        with pytest.raises(ValueError):
            panel.set_brightness(101)
        assert panel.set_brightness() == 10

    @pytest.mark.parametrize("value", [0, 96, 200])
    def test_jpeg_quality_rejects_out_of_range(self, make_panel, fake_dev,
                                               value):
        with pytest.raises(ValueError):
            make_panel(fake_dev, jpeg_quality=value)

    @pytest.mark.parametrize("value", [1, 95])
    def test_jpeg_quality_accepts_boundaries(self, make_panel, fake_dev,
                                             value):
        assert make_panel(fake_dev, jpeg_quality=value).jpeg_quality == value

    def test_jpeg_quality_defaults(self, panel):
        assert panel.jpeg_quality == AKP02.JPEG_QUALITY

    @pytest.mark.parametrize("at", [
        (1900, 50),           # overflows the right edge
        (50, 400),            # overflows the bottom edge
        (-1, 0),              # negative x
        (0, -1),              # negative y
        (1920 - 200 + 1, 0),  # exactly one pixel past the right edge
        (0, 462 - 100 + 1),   # exactly one pixel past the bottom edge
    ])
    def test_region_out_of_bounds_raises(self, panel, at):
        img = Image.new("RGB", (200, 100))
        with pytest.raises(ValueError):
            panel.show(img, at=at)

    def test_region_flush_against_bottom_right_is_allowed(self, panel):
        # Off-by-one guard: exactly filling the remaining space must not
        # be rejected. y=362 isn't color-aligned, so this also exercises
        # the auto-correction -- that's fine, just acknowledge the warning
        # explicitly rather than let it appear as unexplained noise.
        img = Image.new("RGB", (200, 100))
        with pytest.warns(UserWarning, match="shifted"):
            panel.show(img, at=(1920 - 200, 462 - 100))  # should not raise

    @pytest.mark.parametrize("value", [0, -1, -0.5])
    def test_start_keepalive_rejects_non_positive_interval(self, panel, value):
        # Event.wait() returns immediately for these, so an unchecked
        # value becomes a tight loop against the device rather than an
        # error.
        before = {t.ident for t in threading.enumerate()}
        with pytest.raises(ValueError, match="keepalive interval"):
            panel.start_keepalive(interval_sec=value)
        assert alive_keepalives(before) == []

    def test_start_keepalive_rejects_bad_interval_when_already_running(
            self, panel):
        # The check runs before the "already running" no-op, so a bad
        # value is reported either way rather than depending on whether
        # a thread happens to be alive.
        panel.start_keepalive(interval_sec=10)
        with pytest.raises(ValueError, match="keepalive interval"):
            panel.start_keepalive(interval_sec=0)
        panel.stop_keepalive()

    def test_raw_bytes_with_at_raises(self, panel):
        with pytest.raises(ValueError):
            panel.show(b"fake jpeg data", at=(0, 0))


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

class TestErrorHandling:
    def test_write_failure_raises_oserror(self, panel, fake_dev):
        fake_dev.fail_after = 0
        with pytest.raises(OSError):
            panel.clear()

    def test_write_failure_includes_device_error_string(self, panel,
                                                        fake_dev):
        fake_dev.fail_after = 0
        with pytest.raises(OSError, match="simulated error"):
            panel.clear()

    def test_failure_midway_through_a_transfer_releases_the_lock(
            self, panel, fake_dev):
        # A frame that fails on its third report must not leave the
        # device lock held -- that would wedge every later call, and
        # close() along with them.
        fake_dev.fail_after = 3
        with pytest.raises(OSError):
            panel.show(Image.new("RGB", (1920, 462)))
        assert panel._lock.acquire(timeout=0.5)
        panel._lock.release()

    @pytest.mark.parametrize("call", [
        lambda p: p.clear(),
        lambda p: p.screen_on(),
        lambda p: p.set_brightness(50),
        lambda p: p.heartbeat(),
        lambda p: p.firmware_version(),
        lambda p: p.serial_number(),
        lambda p: p.show(Image.new("RGB", (1920, 462))),
    ])
    def test_use_after_close_raises_runtimeerror(self, panel, call):
        panel.close()
        with pytest.raises(RuntimeError):
            call(panel)


# ---------------------------------------------------------------------
# Settling delay: a region update sent immediately after a full-frame
# draw needs a brief gap or the full frame can fail to render at all
# (confirmed on real hardware). Region-after-region and full-after-full
# both need no delay.
#
# These assert on the sleep the library requests (via the `clock`
# fixture) rather than on wall-clock elapsed time, which also includes
# JPEG encoding and sits too close to the threshold to be reliable.
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Region color-alignment auto-correction. Confirmed on real hardware:
# a region update renders with the wrong color unless the position that
# actually lands in the header (header x = PANEL_SHORT_SIDE - y - height in landscape)
# satisfies portrait_x % 8 == 2. show() corrects y automatically to
# reach this, trying both directions and preferring the smaller shift.
# The specific y/height pairs below are real values tested on the
# physical device (y=88/86 confirmed bad, y=84 confirmed good, all with
# height=128) -- not arbitrary numbers.
# ---------------------------------------------------------------------

def _align_landscape_y(panel, y, height):
    """The landscape-upright call, as these hardware cases were recorded."""
    return panel._align_axis(y, height, "y", reflected=True)


class TestColorAlignment:
    def test_already_aligned_y_is_unchanged_and_silent(self, panel):
        # y=84, height=128 -> portrait_x=250, 250%8=2: already aligned.
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning here is a failure
            assert _align_landscape_y(panel, 84, 128) == 84

    def test_known_bad_case_is_corrected(self, panel):
        # y=88, height=128 -> portrait_x=246, 246%8=6: confirmed bad on
        # real hardware; corrects to the confirmed-good y=84.
        with pytest.warns(UserWarning, match="shifted"):
            assert _align_landscape_y(panel, 88, 128) == 84

    def test_original_reported_bug_is_corrected(self, panel):
        # The y=86 case that started this whole investigation.
        with pytest.warns(UserWarning, match="shifted"):
            corrected = _align_landscape_y(panel, 86, 128)
        portrait_x = panel.PANEL_SHORT_SIDE - corrected - 128
        assert portrait_x % panel.SHORT_AXIS_ALIGN_MODULUS == \
            panel.SHORT_AXIS_ALIGN_RESIDUE

    def test_correction_prefers_smaller_shift(self, panel):
        # y=88 is 4 away from y=84 (down) and 4 away from y=92 (up) --
        # a genuine tie. y=87 is 3 away from y=84 (down) and 5 away from
        # y=92 (up), so down should be strictly preferred.
        with pytest.warns(UserWarning, match="shifted"):
            corrected = _align_landscape_y(panel, 87, 128)
        assert abs(corrected - 87) <= 4

    def test_near_top_edge_shifts_up_not_negative(self, panel):
        # y=2 can't shift down without going negative -- must shift up.
        with pytest.warns(UserWarning, match="shifted"):
            corrected = _align_landscape_y(panel, 2, 128)
        assert corrected >= 0
        portrait_x = panel.PANEL_SHORT_SIDE - corrected - 128
        assert portrait_x % 8 == 2

    def test_near_bottom_edge_shifts_down_not_past_screen(self, panel):
        # Shifting up would push height=128 past PANEL_SHORT_SIDE=462.
        y = panel.PANEL_SHORT_SIDE - 128 - 1
        with pytest.warns(UserWarning, match="shifted"):
            corrected = _align_landscape_y(panel, y, 128)
        assert corrected + 128 <= panel.PANEL_SHORT_SIDE
        portrait_x = panel.PANEL_SHORT_SIDE - corrected - 128
        assert portrait_x % 8 == 2

    def test_impossible_case_warns_and_returns_uncorrected(self, panel):
        # height=461 (not 462 -- see the full-height exception below)
        # leaves too little room on either side to fit either shift.
        with pytest.warns(UserWarning, match="cannot be shifted"):
            result = _align_landscape_y(panel, 0, 461)
        assert result == 0  # returned unchanged, not silently altered

    def test_full_height_needs_no_correction_and_warns_never(self, panel):
        # Confirmed on real hardware: height == PANEL_SHORT_SIDE (a region
        # spanning the whole short axis, only ever at y=0) renders
        # correctly with no correction -- consistent with our
        # understanding of the bug, since there's no partial remainder
        # on this axis for the device's edge-padding to misalign.
        # portrait_x=0 here, residue 0 -- would need "fixing" by the
        # general rule, but must NOT warn, unlike a real unfixable case.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _align_landscape_y(panel, 0, panel.PANEL_SHORT_SIDE) == 0

    def test_only_y_is_adjusted_x_and_size_are_not(self, panel, fake_dev):
        img = Image.new("RGB", (200, 100))
        with pytest.warns(UserWarning, match="shifted"):
            panel.show(img, at=(1600, 50))
        header = fake_dev.writes[0][1:33]
        w, h, x, y = struct.unpack(">HHHH", header[13:21])
        # x and size must match a call with the CORRECTED y and nothing
        # else changed.
        assert (x, y, w, h) == panel._to_buffer_rect(
            _Rect(1600, 48, 200, 100), LANDSCAPE, False)


class TestSettlingDelay:
    FULL = (1920, 462)
    REGION = (200, 100)

    def test_full_then_region_has_delay(self, panel, clock):
        panel.show(Image.new("RGB", self.FULL))
        clock.sleeps.clear()
        panel.show(Image.new("RGB", self.REGION), at=(32, 32))
        assert clock.sleeps == [AKP02.FULL_TO_REGION_SETTLE_SEC]

    def test_region_then_region_has_no_delay(self, panel, clock):
        region = Image.new("RGB", self.REGION)
        panel.show(region, at=(32, 32))  # first call: has delay (safe default)
        clock.sleeps.clear()
        panel.show(region, at=(32, 32))  # second: no delay expected
        assert clock.sleeps == []

    def test_full_then_full_has_no_delay(self, panel, clock):
        full = Image.new("RGB", self.FULL)
        panel.show(full)
        clock.sleeps.clear()
        panel.show(full)
        assert clock.sleeps == []

    def test_first_call_ever_being_region_has_delay(self, panel, clock):
        # No prior show() at all -- safe default assumes settling may be
        # needed, since there's no hardware evidence either way.
        panel.show(Image.new("RGB", self.REGION), at=(32, 32))
        assert clock.sleeps == [AKP02.FULL_TO_REGION_SETTLE_SEC]

    def test_failed_transfer_does_not_clear_the_full_screen_flag(
            self, panel, fake_dev, clock):
        # If a full-frame draw fails partway, the panel may still be
        # showing a settling full frame, so the next region update must
        # keep its delay rather than assume the state advanced.
        panel.show(Image.new("RGB", self.FULL))
        fake_dev.fail_after = len(fake_dev.writes) + 2
        with pytest.raises(OSError):
            panel.show(Image.new("RGB", self.FULL))
        fake_dev.fail_after = None
        clock.sleeps.clear()
        panel.show(Image.new("RGB", self.REGION), at=(32, 32))
        assert clock.sleeps == [AKP02.FULL_TO_REGION_SETTLE_SEC]

    def test_delay_precedes_the_first_report_of_the_region(self, panel,
                                                           fake_dev):
        # Ordering matters as much as duration: sleeping after the header
        # has already gone out would defeat the purpose. Recorded against
        # the real write stream rather than the clock.
        observed = []
        original = akp02.time.sleep

        def spy(seconds):
            observed.append(len(fake_dev.writes))
            original(seconds)

        panel.show(Image.new("RGB", self.FULL))
        writes_after_full = len(fake_dev.writes)
        akp02.time.sleep = spy
        try:
            panel.show(Image.new("RGB", self.REGION), at=(32, 32))
        finally:
            akp02.time.sleep = original
        assert observed == [writes_after_full]

    def test_delay_is_actually_awaited(self, panel):
        # One real-clock check that the delay is a genuine wait, so a
        # refactor that recorded the sleep without performing it could
        # not slip past the clock-fixture tests above. The >= direction
        # is unaffected by encoding time, so it is not flaky.
        panel.show(Image.new("RGB", self.FULL))
        start = time.perf_counter()
        panel.show(Image.new("RGB", self.REGION), at=(32, 32))
        assert time.perf_counter() - start >= AKP02.FULL_TO_REGION_SETTLE_SEC


# ---------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_start_keepalive_spawns_one_thread(self, panel):
        before = {t.ident for t in threading.enumerate()}
        starters = [threading.Thread(
            target=lambda: panel.start_keepalive(interval_sec=10))
            for _ in range(20)]
        for t in starters:
            t.start()
        for t in starters:
            t.join()
        assert len(alive_keepalives(before)) == 1
        panel.stop_keepalive()

    def test_start_keepalive_twice_reuses_the_running_thread(self, panel):
        panel.start_keepalive(interval_sec=10)
        first = panel._keepalive_thread
        panel.start_keepalive(interval_sec=10)
        assert panel._keepalive_thread is first
        panel.stop_keepalive()

    def test_stop_keepalive_without_start_is_safe(self, panel):
        panel.stop_keepalive()  # should not raise

    def test_keepalive_actually_sends_heartbeats(self, make_panel):
        # Nothing previously checked that the thread does its one job.
        # Event-signalled rather than sleep-and-count, so it is neither
        # slow nor timing-dependent.
        seen = threading.Event()

        class SignallingDevice:
            def write(self, data):
                if data[1:13] == b"CRT\x00\x00CONNECT":
                    seen.set()
                return len(data)

            def close(self):
                pass

        p = make_panel(SignallingDevice())
        p.start_keepalive(interval_sec=0.01)
        try:
            assert seen.wait(5), "no heartbeat within 5s"
        finally:
            p.stop_keepalive()

    def test_lock_prevents_interleaving_during_transfer(self, make_panel):
        class SlowTrackingDevice:
            def __init__(self):
                self.violations = 0
                self._busy = False
                self._vlock = threading.Lock()

            def write(self, data):
                with self._vlock:
                    if self._busy:
                        self.violations += 1
                    self._busy = True
                time.sleep(0.001)
                with self._vlock:
                    self._busy = False
                return len(data)

            def close(self):
                pass

        dev = SlowTrackingDevice()
        p = make_panel(dev)
        p.start_keepalive(interval_sec=0.001)
        img = Image.new("RGB", (1920, 462))
        t = threading.Thread(target=lambda: p.show(img))
        t.start()
        t.join()
        p.stop_keepalive()
        assert dev.violations == 0

    def test_keepalive_survives_disconnect_and_reconnect(self, make_panel):
        class FlakyDevice:
            def __init__(self):
                self.fail = True
                self.beat = threading.Event()

            def write(self, data):
                if self.fail:
                    return -1
                self.beat.set()
                return len(data)

            def error(self):
                return "simulated disconnect"

            def close(self):
                pass

        dev = FlakyDevice()
        p = make_panel(dev)
        before = {t.ident for t in threading.enumerate()}
        p.start_keepalive(interval_sec=0.01)

        # Wait for the thread to die of the write failure rather than
        # guessing at a sleep duration.
        deadline = time.monotonic() + 5
        while p._keepalive_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not p._keepalive_thread.is_alive(), "thread should have exited"

        dev.fail = False
        p.start_keepalive(interval_sec=0.01)  # attempt resume
        assert dev.beat.wait(5), "resumed keepalive sent no heartbeat"
        assert len(alive_keepalives(before)) == 1
        p.stop_keepalive()

    def test_stop_keepalive_bounded_on_wedged_device(self, make_panel):
        # The wedge is released on the way out so this test stops leaving
        # a permanently blocked thread -- and a permanently held device
        # lock -- behind for whatever runs next.
        release = threading.Event()
        entered = threading.Event()

        class WedgedDevice:
            def write(self, data):
                entered.set()
                release.wait(10)
                return len(data)

            def close(self):
                pass

        p = make_panel(WedgedDevice())
        p.start_keepalive(interval_sec=0.01)
        try:
            assert entered.wait(5), "keepalive never reached the device"
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                p.stop_keepalive(timeout_sec=0.5)
            assert time.perf_counter() - start < 1.0
            assert len(caught) == 1
            assert issubclass(caught[0].category, UserWarning)
            assert "keepalive" in str(caught[0].message)
        finally:
            release.set()

    def test_close_avoids_use_after_free_when_transfer_in_flight(
            self, make_panel):
        # Drives the block through a real clear()/write() rather than
        # grabbing the private lock directly, so this exercises the
        # actual code path close() is defending against.
        release = threading.Event()
        entered = threading.Event()

        class SlowDevice:
            def __init__(self):
                self.closed = False

            def write(self, data):
                entered.set()
                release.wait(10)
                return len(data)

            def close(self):
                self.closed = True

        dev = SlowDevice()
        p = make_panel(dev)
        writer = threading.Thread(target=p.clear, daemon=True)
        writer.start()
        try:
            assert entered.wait(5), "writer never reached the device"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                p.close(timeout_sec=0.1)
            # leaked, not closed, while a transfer is (simulated) in flight
            assert dev.closed is False
            assert len(caught) == 1
            assert "leaking" in str(caught[0].message)
        finally:
            release.set()
            writer.join(5)


# ---------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------

class TestImageHandling:
    def test_correctly_sized_image_declares_full_panel(self, panel, fake_dev):
        img = Image.new("RGB", (1920, 462), color=(10, 20, 30))
        panel.show(img)
        header = fake_dev.writes[0][1:33]
        assert struct.unpack(">HHHH", header[13:21]) == (0, 0, 0, 0)

    def test_region_header_fields_correct(self, panel, fake_dev):
        img = Image.new("RGB", (200, 100))
        with pytest.warns(UserWarning, match="shifted"):
            panel.show(img, at=(100, 50))
        header = fake_dev.writes[0][1:33]
        w, h, x, y = struct.unpack(">HHHH", header[13:21])
        # y=50 is not color-aligned; the library corrects it to y=48
        # automatically (see AKP02._align_axis) -- the header must
        # reflect the CORRECTED position, not the one originally requested.
        assert (x, y, w, h) == panel._to_buffer_rect(
            _Rect(100, 48, 200, 100), LANDSCAPE, False)

    def test_declared_length_matches_the_bytes_actually_sent(self, panel,
                                                             fake_dev):
        panel.show(Image.new("RGB", (1920, 462)))
        header, jpeg = payload_of(fake_dev.writes[:-1])  # exclude the commit
        assert struct.unpack(">I", header[8:12])[0] == len(jpeg) + 0x20
        assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")

    def test_payload_is_chunked_into_whole_reports(self, panel, fake_dev):
        panel.show(Image.new("RGB", (1920, 462)))
        _, jpeg = payload_of(fake_dev.writes[:-1])
        expected = -(-(len(jpeg) + 32) // AKP02.HID_REPORT_SIZE)  # ceil
        assert expected > 1, "test image should span several reports"
        assert len(fake_dev.writes) == expected + 1  # + the commit report

    def test_sent_jpeg_is_portrait(self, panel, fake_dev):
        panel.show(Image.new("RGB", (1920, 462)))
        assert decode_sent_image(fake_dev.writes[:-1]).size == (462, 1920)

    def test_rotation_is_clockwise(self, panel, fake_dev):
        # Pins ROTATE_270 against ROTATE_90, which would produce an
        # equally valid-looking portrait JPEG that is upside down on the
        # panel -- and which no size or header assertion can catch.
        # Landscape maps (x, y) -> (H-1-y, x), so the landscape top-left
        # quadrant ends up top-right in the portrait buffer.
        img = Image.new("RGB", (1920, 462))
        quadrants = {(0, 0): (255, 0, 0), (960, 0): (0, 255, 0),
                     (0, 231): (0, 0, 255), (960, 231): (255, 255, 0)}
        for (qx, qy), color in quadrants.items():
            img.paste(color, (qx, qy, qx + 960, qy + 231))
        panel.show(img)
        out = decode_sent_image(fake_dev.writes[:-1]).convert("RGB")

        assert_close(out.getpixel((115, 480)), (0, 0, 255))     # top-left
        assert_close(out.getpixel((346, 480)), (255, 0, 0))     # top-right
        assert_close(out.getpixel((115, 1440)), (255, 255, 0))  # bottom-left
        assert_close(out.getpixel((346, 1440)), (0, 255, 0))    # bottom-right

    def test_region_lands_where_the_full_frame_puts_it(self, make_panel):
        # The pixel rotation and _to_buffer_rect are two descriptions
        # of the same landscape -> portrait mapping, so they must agree.
        # Rather than pinning each separately, this checks them against
        # each other: render a scene as a full frame, then render one
        # region of that same scene and compare it against the buffer at
        # the coordinates the header asked for. Changing the transpose
        # without changing the rect transform (or the reverse) makes the
        # region land somewhere else and fails here -- whichever
        # rotation direction turns out to be right for the panel.
        scene = Image.new("RGB", (1920, 462), (20, 20, 20))
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                                   (255, 255, 0), (255, 0, 255)]):
            scene.paste(color, (i * 384, 0, (i + 1) * 384, 231))
            scene.paste(tuple(c // 3 for c in color),
                        (i * 384, 231, (i + 1) * 384, 462))

        full_dev = FakeDevice()
        make_panel(full_dev).show(scene)
        full_fb = decode_sent_image(full_dev.writes[:-1]).convert("RGB")

        box = (600, 100, 904, 300)  # y+height=300, 300%8=4 -- color-aligned
                                     # per AKP02.SHORT_AXIS_ALIGN_MODULUS/
                                     # RESIDUE, so this test stays focused on
                                     # rotation/rect consistency without also
                                     # incidentally exercising auto-correction
        region_dev = FakeDevice()
        make_panel(region_dev).show(scene.crop(box), at=box[:2])
        header, _ = payload_of(region_dev.writes[:-1])
        w, h, x, y = struct.unpack(">HHHH", header[13:21])
        region = decode_sent_image(region_dev.writes[:-1]).convert("RGB")

        assert (w, h) == region.size
        assert x + w <= full_fb.width and y + h <= full_fb.height, \
            "header rect falls outside the portrait buffer"
        diff = ImageChops.difference(
            full_fb.crop((x, y, x + w, y + h)), region).convert("L")
        mad = sum(i * n for i, n in enumerate(diff.histogram())) / (w * h)
        assert mad < 12, (
            f"region landed in the wrong place (mean abs difference {mad:.1f})"
            " -- the transpose and _to_buffer_rect disagree")

    def test_wrong_sized_image_is_letterboxed_not_stretched(self, panel,
                                                            fake_dev):
        # Was only asserting "does not raise", which a stretch-to-fit
        # regression would also satisfy. A 4:3 source is height-limited
        # on this panel, so it must end up with black bars.
        panel.show(Image.new("RGB", (800, 600), (255, 255, 255)))
        out = decode_sent_image(fake_dev.writes[:-1]).convert("RGB")
        assert out.size == (462, 1920)
        assert_close(out.getpixel((231, 960)), (255, 255, 255))  # content
        assert_close(out.getpixel((231, 60)), (0, 0, 0))         # bar
        assert_close(out.getpixel((231, 1860)), (0, 0, 0))       # bar

    @pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
    def test_non_rgb_modes_are_converted(self, panel, fake_dev, mode):
        # JPEG cannot encode these directly -- without the conversion
        # step PIL raises, so this pins a real failure mode.
        panel.show(Image.new(mode, (1920, 462)))
        assert decode_sent_image(fake_dev.writes[:-1]).size == (462, 1920)

    def test_raw_jpeg_bytes_are_sent_untouched(self, panel, fake_dev):
        # The bytes-input branch had no success-path coverage at all.
        buf = io.BytesIO()
        Image.new("RGB", (462, 1920), (7, 8, 9)).save(buf, format="JPEG")
        raw = buf.getvalue()
        panel.show(raw)
        header, sent = payload_of(fake_dev.writes[:-1])
        assert sent == raw
        assert struct.unpack(">HHHH", header[13:21]) == (0, 0, 0, 0)

    def test_jpeg_quality_affects_the_encoded_payload(self, make_panel):
        img = Image.effect_noise((1920, 462), 64).convert("RGB")
        sizes = []
        for quality in (1, 95):
            dev = FakeDevice()
            make_panel(dev, jpeg_quality=quality).show(img)
            sizes.append(len(payload_of(dev.writes[:-1])[1]))
        assert sizes[0] < sizes[1], "jpeg_quality is not reaching the encoder"

    def test_source_image_is_not_mutated(self, panel):
        # Callers commonly re-show a cached frame; rotating or converting
        # in place would corrupt it.
        img = Image.new("RGB", (1920, 462), (1, 2, 3))
        panel.show(img)
        assert img.size == (1920, 462)
        assert img.getpixel((0, 0)) == (1, 2, 3)


# ---------------------------------------------------------------------
# Orientation and inverted. The wire format never changes -- the JPEG is
# always 462x1920 -- so these are about the one net rotation each
# combination applies, and about a region's rect going through that same
# rotation. Checked against PIL and against the full frame rather than
# against the library's own arithmetic.
# ---------------------------------------------------------------------

ORIENTATIONS = (LANDSCAPE, PORTRAIT)
COMBOS = [(o, i) for o in ORIENTATIONS for i in (False, True)]
DIHEDRAL = {
    "identity": None,
    "ROTATE_90": Image.Transpose.ROTATE_90,
    "ROTATE_180": Image.Transpose.ROTATE_180,
    "ROTATE_270": Image.Transpose.ROTATE_270,
    "FLIP_LEFT_RIGHT": Image.Transpose.FLIP_LEFT_RIGHT,
    "FLIP_TOP_BOTTOM": Image.Transpose.FLIP_TOP_BOTTOM,
    "TRANSPOSE": Image.Transpose.TRANSPOSE,
    "TRANSVERSE": Image.Transpose.TRANSVERSE,
}
REFLECTIONS = {"FLIP_LEFT_RIGHT", "FLIP_TOP_BOTTOM", "TRANSPOSE", "TRANSVERSE"}


def _configured(make_panel, dev, orientation, inverted):
    p = make_panel(dev)
    p.orientation(orientation)
    p.inverted = inverted
    dev.writes.clear()          # drop the SET, leave only the transfer
    return p


def _asymmetric_scene(size):
    """A scene no rotation or reflection of itself can be mistaken for."""
    w, h = size
    img = Image.new("RGB", (w, h), (12, 12, 12))
    img.paste((255, 0, 0), (0, 0, w // 4, h // 4))
    img.paste((0, 255, 0), (w - w // 5, 0, w, h // 6))
    img.paste((0, 0, 255), (0, h - h // 3, w // 6, h))
    return img


def _identify(src, out):
    """Name the dihedral transform taking src to out."""
    best, best_err = None, None
    for name, transpose in DIHEDRAL.items():
        cand = src if transpose is None else src.transpose(transpose)
        if cand.size != out.size:
            continue
        a, b = cand.load(), out.load()
        w, h = cand.size
        pts = [(x, y) for x in range(3, w, max(1, w // 30))
               for y in range(3, h, max(1, h // 30))]
        err = sum(sum(abs(m - n) for m, n in zip(a[q], b[q], strict=True))
                  for q in pts) / len(pts)
        if best_err is None or err < best_err:
            best, best_err = name, err
    return best


class TestOrientationGeometry:
    @pytest.mark.parametrize("orientation,inverted,expected", [
        (LANDSCAPE, False, "ROTATE_270"),
        (LANDSCAPE, True, "ROTATE_90"),
        (PORTRAIT, False, "identity"),
        (PORTRAIT, True, "ROTATE_180"),
    ])
    def test_each_combination_applies_one_net_rotation(
            self, fake_dev, make_panel, orientation, inverted, expected):
        # `inverted` must be a ROTATION. Composing a flip with the
        # landscape turn instead gives a reflection, which renders text
        # backwards -- it looks plausible in a solid-colour test and is
        # obviously wrong on a real panel, so the check is what transform
        # was applied, not merely that pixels moved.
        panel = _configured(make_panel, fake_dev, orientation, inverted)
        src = _asymmetric_scene(panel.size)
        panel.show(src)
        name = _identify(src, decode_sent_image(fake_dev.writes[:-1]))
        assert name == expected
        assert name not in REFLECTIONS

    @pytest.mark.parametrize("orientation", ORIENTATIONS)
    def test_inverted_is_the_upright_frame_seen_upside_down(
            self, make_panel, orientation):
        # The property a user actually cares about: walk around the panel
        # and you see the same picture, not a mirrored one.
        #
        # Both sides of the pair are built here, so `inverted` is the
        # subject of the comparison rather than an input to it.
        # Parametrizing over it as well would run each orientation twice
        # to do the same work once.
        up_dev, inv_dev = FakeDevice(), FakeDevice()
        up = _configured(make_panel, up_dev, orientation, False)
        inv = _configured(make_panel, inv_dev, orientation, True)
        src = _asymmetric_scene(up.size)
        up.show(src)
        inv.show(src)
        want = decode_sent_image(up_dev.writes[:-1]).transpose(
            Image.Transpose.ROTATE_180)
        got = decode_sent_image(inv_dev.writes[:-1])
        assert _identify(want, got) == "identity"

    @pytest.mark.parametrize("orientation,inverted", COMBOS)
    def test_a_region_lands_where_the_full_frame_puts_it(
            self, make_panel, orientation, inverted):
        # The rect and the pixels are two descriptions of one rotation.
        # Drawing a crop as a region must reproduce exactly the bytes the
        # full frame put at that rect -- this is what catches a rect
        # transform that has drifted from _TRANSPOSE.
        full_dev, region_dev = FakeDevice(), FakeDevice()
        full = _configured(make_panel, full_dev, orientation, inverted)
        region = _configured(make_panel, region_dev, orientation, inverted)
        src = _asymmetric_scene(full.size)
        full.show(src)
        whole = decode_sent_image(full_dev.writes[:-1])

        w, h = (96, 64) if full.size[1] == AKP02.PANEL_SHORT_SIDE else (64, 96)
        placed = False
        for pos in range(8, 120):
            at = (40, pos) if full.size[1] == AKP02.PANEL_SHORT_SIDE else (pos, 40)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                region_dev.writes.clear()
                region.show(src.crop((at[0], at[1], at[0] + w, at[1] + h)),
                            at=at)
            if caught:
                continue        # the aligner moved it; compare like with like
            placed = True
            break
        assert placed, "no unshifted position found"

        header, _ = payload_of(region_dev.writes[:-1])
        rw, rh, rx, ry = struct.unpack(">HHHH", header[13:21])
        patch = whole.crop((rx, ry, rx + rw, ry + rh))
        sent = decode_sent_image(region_dev.writes[:-1])
        assert sent.size == (rw, rh)
        assert _identify(patch, sent) == "identity"

    @pytest.mark.parametrize("orientation,inverted", COMBOS)
    def test_correction_lands_the_header_x_on_the_rule(
            self, make_panel, orientation, inverted):
        # Which coordinate moves, and which way, differs across all four
        # combinations -- inverting swaps whether the header's x runs
        # with or against the caller's. Sweep positions and check the
        # value that actually reaches the header.
        dev = FakeDevice()
        panel = _configured(make_panel, dev, orientation, inverted)
        short_is_y = panel.size[1] == AKP02.PANEL_SHORT_SIDE
        w, h = (96, 64) if short_is_y else (64, 96)
        for start in range(0, 24):
            dev.writes.clear()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                panel.show(Image.new("RGB", (w, h), (200, 100, 50)),
                           at=(40, start) if short_is_y else (start, 40))
            header, _ = payload_of(dev.writes[:-1])
            rx = struct.unpack(">HHHH", header[13:21])[2]
            rw, rh = struct.unpack(">HHHH", header[13:21])[:2]
            assert rx % AKP02.SHORT_AXIS_ALIGN_MODULUS == \
                AKP02.SHORT_AXIS_ALIGN_RESIDUE, (orientation, inverted, start)
            # Only the position moves: the region keeps its size.
            assert sorted((rw, rh)) == sorted((w, h))

    @pytest.mark.parametrize("orientation,inverted", COMBOS)
    def test_the_transfer_is_always_a_portrait_buffer(
            self, make_panel, orientation, inverted):
        dev = FakeDevice()
        panel = _configured(make_panel, dev, orientation, inverted)
        panel.show(Image.new("RGB", panel.size))
        assert decode_sent_image(dev.writes[:-1]).size == (462, 1920)

    def test_size_follows_the_orientation_and_ignores_inverted(self, panel):
        assert panel.size == (1920, 462)
        panel.inverted = True
        assert panel.size == (1920, 462)   # a half turn keeps the shape
        panel.orientation(PORTRAIT)
        assert panel.size == (462, 1920)

    def test_oversized_report_is_rejected_clearly(self, panel):
        # The only uncovered branch otherwise. Internal callers all chunk,
        # so this guards a future edit -- and without it the failure is an
        # unexplained "negative count" from bytes() further down.
        with pytest.raises(ValueError, match="HID_REPORT_SIZE"):
            panel._write_report(b"x" * (AKP02.HID_REPORT_SIZE + 1))

    def test_a_misspelled_attribute_is_rejected(self, panel):
        # `inverted` is the one knob with no method behind it, so a typo
        # binds a dead attribute and the panel just keeps doing the
        # default thing -- a frame that looks wrong with nothing pointing
        # at the line that caused it. __slots__ turns that into an error
        # at the assignment.
        for name in ("invert", "inverted_", "inverted2"):
            with pytest.raises(AttributeError, match=name):
                setattr(panel, name, True)
        panel.inverted = True          # the real one still works
        assert panel.inverted is True

    def test_subclasses_stay_open(self, fake_dev, make_panel):
        # __slots__ closes AKP02 itself, not anything built on it: a
        # subclass without its own __slots__ still gets a __dict__.
        class Panel(AKP02):
            pass

        panel = make_panel(fake_dev, keepalive_interval=None)
        sub = Panel(dev=FakeDevice(), keepalive_interval=None)
        sub.my_own_state = 1
        assert sub.my_own_state == 1
        del panel

    def test_inverted_takes_any_truthy_value(self, fake_dev, make_panel):
        # A plain public attribute takes whatever is assigned, and the
        # transform is picked by a dict keyed on (orientation, bool) --
        # so a truthy non-bool has to be coerced, not KeyError.
        panel = make_panel(fake_dev, keepalive_interval=None)
        for value in (1, "yes", 0, "", None):
            panel.inverted = value
            panel.show(Image.new("RGB", panel.size))

    def test_failed_orientation_write_leaves_the_mode_alone(self, make_panel):
        # The call raises, so the caller assumes nothing happened; the
        # host must not go on rendering for a mode the device rejected.
        class Unplugged(FakeDevice):
            def write(self, data):
                raise OSError("unplugged")

        panel = make_panel(Unplugged(), keepalive_interval=None)
        with pytest.raises(OSError):
            panel.orientation(PORTRAIT)
        assert panel.orientation() is LANDSCAPE
        assert panel.size == (1920, 462)

    def test_entering_the_context_sends_no_orientation_command(
            self, fake_dev, make_panel):
        # The splash orientation is persisted device state; a `with` must
        # not overwrite a setting the caller never mentioned.
        panel = make_panel(fake_dev, keepalive_interval=None)
        with panel:
            pass
        assert fake_dev.writes == []

# ---------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------

class TestPackaging:
    def test_version_is_a_release_string(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?",
                            akp02.__version__), akp02.__version__

    def test_version_matches_installed_metadata(self):
        # pyproject declares the version dynamic and hatchling parses it
        # out of akp02/__init__.py, so a broken config surfaces as a
        # built distribution whose metadata disagrees with the source --
        # not as anything that fails at import. Skipped in a bare
        # checkout, where there is no metadata to compare against.
        try:
            installed = importlib.metadata.version("akp02")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("akp02 is not installed in this environment")
        assert installed == akp02.__version__


# ---------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------

class TestLifecycle:
    def test_context_manager_closes_device(self, fake_dev):
        with AKP02(dev=fake_dev) as p:
            p.clear()
        assert fake_dev.closed is True

    def test_context_manager_closes_on_exception(self, fake_dev):
        with pytest.raises(ZeroDivisionError), AKP02(dev=fake_dev):
                raise ZeroDivisionError
        assert fake_dev.closed is True

    def test_context_manager_stops_keepalive(self, fake_dev):
        before = {t.ident for t in threading.enumerate()}
        with AKP02(dev=fake_dev) as p:
            p.start_keepalive(interval_sec=10)
        assert alive_keepalives(before) == []

    def test_context_manager_starts_keepalive(self, fake_dev):
        before = {t.ident for t in threading.enumerate()}
        with AKP02(dev=fake_dev):
            # alive_keepalives filters on is_alive() already.
            assert len(alive_keepalives(before)) == 1
        assert alive_keepalives(before) == []

    def test_keepalive_none_starts_no_thread(self, fake_dev):
        before = {t.ident for t in threading.enumerate()}
        with AKP02(dev=fake_dev, keepalive_interval=None) as p:
            assert alive_keepalives(before) == []
            assert p._keepalive_thread is None

    def test_construction_alone_starts_no_thread(self, make_panel, fake_dev):
        # The thread belongs to __enter__, not __init__: constructing a
        # panel (as every test double here does) must have no background
        # side effect.
        before = {t.ident for t in threading.enumerate()}
        p = make_panel(fake_dev)
        assert alive_keepalives(before) == []
        assert p._keepalive_thread is None

    def test_context_manager_uses_the_configured_interval(self, make_panel,
                                                          fake_dev):
        # The interval reaches the thread, rather than __enter__ falling
        # back to start_keepalive's own default.
        p = make_panel(fake_dev, keepalive_interval=0.01)
        with p:
            assert fake_dev is p._dev
            deadline = time.monotonic() + 5
            while not fake_dev.writes and time.monotonic() < deadline:
                time.sleep(0.005)
        assert fake_dev.writes, "no heartbeat within 5s at a 0.01s interval"
        assert fake_dev.writes[0][1:9] == b"CRT\x00\x00CON"

    @pytest.mark.parametrize("bad", [0, -1, -0.5])
    def test_invalid_keepalive_interval_rejected(self, fake_dev, bad):
        # Caught at construction, so the traceback points at the caller
        # rather than surfacing later as a tight loop on the device.
        with pytest.raises(ValueError, match="keepalive interval"):
            AKP02(dev=fake_dev, keepalive_interval=bad)

    def test_explicit_start_inside_context_is_a_noop(self, fake_dev):
        # Pre-1.1 code called start_keepalive() itself; that must stay
        # harmless rather than spawning a second thread.
        before = {t.ident for t in threading.enumerate()}
        with AKP02(dev=fake_dev) as p:
            first = p._keepalive_thread
            p.start_keepalive()
            assert p._keepalive_thread is first
            assert len(alive_keepalives(before)) == 1

    def test_stop_keepalive_inside_context_stays_stopped(self, fake_dev):
        # Letting the panel sleep while still holding the handle open.
        before = {t.ident for t in threading.enumerate()}
        with AKP02(dev=fake_dev) as p:
            p.stop_keepalive()
            assert alive_keepalives(before) == []
            p.clear()  # device still usable
        assert fake_dev.closed is True

    def test_close_is_idempotent(self, fake_dev):
        p = AKP02(dev=fake_dev)
        p.close()
        p.close()  # should not raise

    def test_close_after_keepalive_died_is_clean(self, make_panel):
        # start_keepalive leaves a dead Thread object behind by design;
        # close() must not warn about failing to join it.
        class DeadDevice:
            def write(self, data):
                return -1

            def error(self):
                return "gone"

            def close(self):
                pass

        p = make_panel(DeadDevice())
        p.start_keepalive(interval_sec=0.01)
        deadline = time.monotonic() + 5
        while p._keepalive_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.005)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p.close(timeout_sec=1)
        assert caught == []
