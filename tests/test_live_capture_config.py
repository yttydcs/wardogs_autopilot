"""ROI edits must reach the next capture without restarting localization."""

import queue
import threading

import numpy as np

from autopilot.common.config import AppConfig
from autopilot.vision import tracker


def test_capture_reopens_with_updated_ui_roi(monkeypatch):
    cfg = AppConfig().to_dict()
    cfg["capture"]["mmap_roi"] = [45, 1009, 336, 277]
    loc = tracker.LiveLocator(cfg, mask=np.zeros((277, 336), dtype=bool))
    stop = threading.Event()
    frames = queue.Queue(maxsize=1)
    captures = []
    closed = []
    grab_count = 0

    class Capture:
        def __init__(self, monitor, roi):
            self.roi = roi
            captures.append(roi)

        def grab(self):
            nonlocal grab_count
            grab_count += 1
            if grab_count == 1:
                # Same dictionary edit performed by the ROI tab.
                cfg["capture"]["mmap_roi"] = [42, 993, 340, 304]
            else:
                stop.set()
            return np.zeros((self.roi[3], self.roi[2], 3), dtype=np.uint8)

        def close(self):
            closed.append(self.roi)

    monkeypatch.setattr(tracker, "ScreenCapture", Capture)
    producer = tracker._CaptureProducer(loc.cfg, loc.mask, None, stop, frames)
    producer.run()
    assert producer.error is None
    assert captures == [(45, 1009, 336, 277), (42, 993, 340, 304)]
    assert closed == captures
    frame = frames.get_nowait()
    assert frame["gray"].shape == (304, 340)
    assert frame["mask"].shape == (304, 340)
    assert frame["roi"] == (42, 993, 340, 304)
