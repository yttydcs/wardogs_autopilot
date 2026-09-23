"""Windows scan-code keyboard driver using the standard SendInput API."""

from __future__ import annotations

import ctypes
import sys
import threading
import time


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("vk", ctypes.c_uint16),
        ("scan", ctypes.c_uint16),
        ("flags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("extra", ctypes.c_size_t),
    ]


class _MouseInput(ctypes.Structure):
    # INPUT's union must also accommodate MOUSEINPUT, even for keyboard events.
    _fields_ = [
        ("dx", ctypes.c_int32),
        ("dy", ctypes.c_int32),
        ("data", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("extra", ctypes.c_size_t),
    ]


class _Payload(ctypes.Union):
    _fields_ = [("keyboard", _KeyboardInput), ("mouse", _MouseInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("payload", _Payload)]


class SoftwareKeyDriver:
    """Hold only requested keys; release stale commands after 200 ms.

    close() is permanent and synchronized with set_state(), so a navigation
    tick racing with emergency stop cannot press keys again.
    """

    SCANS = {"W": 0x11, "A": 0x1E, "S": 0x1F, "D": 0x20, "SPACE": 0x39}

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Software keyboard requires Windows")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._send_input = self._user32.SendInput
        self._send_input.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Input), ctypes.c_int]
        self._send_input.restype = ctypes.c_uint32
        self._lock = threading.RLock()
        self._held: set[str] = set()
        self._closed = False
        self._error: OSError | None = None
        self._last_command = time.monotonic()
        self._stop = threading.Event()
        self._watcher = threading.Thread(target=self._watchdog, daemon=True)
        self._watcher.start()

    def _send(self, key: str, down: bool) -> None:
        event = _Input(type=1)
        event.payload.keyboard = _KeyboardInput(
            0, self.SCANS[key], 0x0008 | (0 if down else 0x0002), 0, 0
        )
        if self._send_input(1, ctypes.byref(event), ctypes.sizeof(event)) != 1:
            raise OSError(
                ctypes.get_last_error(),
                "SendInput failed; check the target window and application permissions",
            )

    def set_state(self, keys: dict[str, bool]) -> None:
        with self._lock:
            if self._closed:
                return
            if self._error is not None:
                raise self._error
            wanted = {k for k in self.SCANS if keys.get(k, False)}
            try:
                for key in self.SCANS:
                    if key in self._held and key not in wanted:
                        self._send(key, False)
                        self._held.remove(key)
                for key in self.SCANS:
                    if key in wanted and key not in self._held:
                        self._send(key, True)
                        self._held.add(key)
            except OSError as exc:
                self._error = exc
                self.release_all()
                raise
            self._last_command = time.monotonic()

    def release_all(self) -> None:
        with self._lock:
            error = None
            for key in list(self._held):
                try:
                    self._send(key, False)
                    self._held.remove(key)
                except OSError as exc:
                    error = exc
            if error is not None:
                raise error

    def held(self) -> list[str]:
        with self._lock:
            return [key for key in self.SCANS if key in self._held]

    def _watchdog(self) -> None:
        while not self._stop.wait(0.05):
            with self._lock:
                if time.monotonic() - self._last_command > 0.2:
                    try:
                        self.release_all()
                    except OSError as exc:
                        self._error = exc

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop.set()
            self.release_all()
