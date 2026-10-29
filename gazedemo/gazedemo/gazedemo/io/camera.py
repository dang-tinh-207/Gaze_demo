"""Frame sources. read() -> (frame_idx, t, frame_bgr) | None, where t is time.monotonic() at
capture (live) or the recorded/virtual time (offline, fake); clock() gives 'now' in the same base."""
from __future__ import annotations

import contextlib
import csv
import sys
import threading
import time
from pathlib import Path

import cv2

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2, "any": cv2.CAP_ANY}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def backend_id(name: str | None) -> int:
    name = (name or "auto").lower()
    if name == "auto":
        if sys.platform.startswith("win"):
            return cv2.CAP_DSHOW
        return cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    return BACKENDS[name]


@contextlib.contextmanager
def quiet_opencv():
    """Silence OpenCV's own log while probing camera indices that do not exist."""
    try:
        level = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        level = None
    try:
        yield
    finally:
        if level is not None:
            cv2.utils.logging.setLogLevel(level)


def open_capture(device: int, backend="auto", width=None, height=None, fps=None, fourcc=None):
    cap = cv2.VideoCapture(int(device), backend_id(backend))
    if not cap.isOpened():
        cap.release()
        return None
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


class LatestFrameCamera:
    """Capture thread keeps only the newest frame (1 slot): a slower Gazelle skips old frames
    instead of queueing them. Every frame also goes to on_frame (the recorder)."""

    def __init__(self, device=0, backend="auto", width=None, height=None, fps=None, fourcc=None,
                 on_frame=None):
        self.args = (device, backend, width, height, fps, fourcc)
        self.on_frame = on_frame
        self.finished = False
        self.error = None
        self.size = None
        self.nominal_fps = None
        self._cond = threading.Condition()
        self._latest = None
        self._returned = -1
        self._stop = threading.Event()
        self._thread = None
        self.cap = None

    def start(self):
        device = self.args[0]
        self.cap = open_capture(*self.args)
        if self.cap is None:
            raise RuntimeError(f"Cannot open camera {device} (run `python -m gazedemo doctor`)")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            raise RuntimeError(f"Camera {device} opened but returns no frames")
        self.size = (frame.shape[1], frame.shape[0])
        self.nominal_fps = self.cap.get(cv2.CAP_PROP_FPS) or None
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        idx, fails = 0, 0
        while not self._stop.is_set():
            ok = self.cap.grab()
            t = time.monotonic()  # capture time: assigned here, not when Gazelle finishes
            frame = None
            if ok:
                ok, frame = self.cap.retrieve()
            if not ok or frame is None:
                fails += 1
                if fails > 100:
                    self.error = "camera stopped sending frames (unplugged?)"
                    break
                time.sleep(0.01)
                continue
            fails = 0
            if self.on_frame is not None:
                self.on_frame(idx, t, frame)
            with self._cond:
                self._latest = (idx, t, frame)
                self._cond.notify_all()
            idx += 1
        with self._cond:
            self.finished = True
            self._cond.notify_all()

    def read(self, timeout: float = 1.0):
        with self._cond:
            self._cond.wait_for(lambda: self.finished or (
                self._latest is not None and self._latest[0] > self._returned), timeout)
            if self._latest is None or self._latest[0] <= self._returned:
                return None
            self._returned = self._latest[0]
            return self._latest

    def clock(self) -> float:
        return time.monotonic()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()


def read_frames_csv(path) -> list[tuple[int, float]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [(int(r["frame_idx"]), float(r["t"])) for r in csv.DictReader(f)]


class VideoFileSource:
    """Replays a video. With frames.csv (offline) each frame gets its recorded capture time and
    clock() is that time; otherwise frames are paced at the file's fps on time.monotonic().
    stride k keeps every k-th frame."""

    def __init__(self, path, frames_csv=None, stride: int = 1, on_frame=None, loop: bool = False):
        self.path = Path(path)
        self.rows = read_frames_csv(frames_csv) if frames_csv else None
        self.stride = max(1, int(stride))
        self.on_frame = on_frame
        self.loop = loop and self.rows is None
        self.finished = False
        self.error = None
        self.size = None
        self.cap = None
        self._i = 0
        self._now = 0.0
        self._t0_mono = None
        self.fps = None

    def start(self):
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video {self.path}")
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.rows is not None and n and abs(n - len(self.rows)) > 1:
            print(f"[video] WARNING: {n} frames in the video but {len(self.rows)} rows in frames.csv")
        self._t0_mono = time.monotonic()
        return self

    def read(self, timeout=None):
        while True:
            if self.rows is not None and self._i >= len(self.rows):
                self.finished = True
                return None
            if self._i % self.stride:
                if not self.cap.grab():
                    self.finished = True
                    return None
                self._i += 1
                continue
            ok, frame = self.cap.read()
            if not ok:
                if self.loop and self._i > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._t0_mono, self._i = time.monotonic(), 0
                    continue
                self.finished = True
                return None
            i = self._i
            self._i += 1
            if self.rows is not None:
                idx, t = self.rows[i]
            else:
                idx, t = i, self._t0_mono + i / self.fps
                wait = t - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            self._now = t
            if self.on_frame is not None:
                self.on_frame(idx, t, frame)
            return idx, t, frame

    def clock(self) -> float:
        return self._now if self.rows is not None else time.monotonic()

    def stop(self):
        self.finished = True
        if self.cap is not None:
            self.cap.release()


class ImageSource:
    """A still image served as a live stream (setup-aoi / demos on a photo)."""

    def __init__(self, path, fps: float = 15.0, on_frame=None):
        self.path = Path(path)
        self.fps = fps
        self.on_frame = on_frame
        self.finished = False
        self.error = None
        self.frame = None
        self.size = None
        self._i = 0
        self._next = 0.0

    def start(self):
        from gazedemo.io.logs import load_image
        self.frame = load_image(self.path)
        if self.frame is None:
            raise RuntimeError(f"Cannot read image {self.path}")
        self.size = (self.frame.shape[1], self.frame.shape[0])
        self._next = time.monotonic()
        return self

    def read(self, timeout=None):
        wait = self._next - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        t = time.monotonic()
        self._next = t + 1.0 / self.fps
        idx = self._i
        self._i += 1
        if self.on_frame is not None:
            self.on_frame(idx, t, self.frame)
        return idx, t, self.frame

    def clock(self) -> float:
        return time.monotonic()

    def stop(self):
        self.finished = True


def open_source(spec, cfg, on_frame=None, aois=None):
    """spec: None (camera from config) | camera index | 'synthetic' | image path | video path."""
    s = cfg.source
    if spec is None or str(spec).isdigit():
        device = s.device if spec is None else int(spec)
        return LatestFrameCamera(device, s.backend, s.width, s.height, s.fps, s.fourcc, on_frame)
    if str(spec) == "synthetic":
        from gazedemo.perception.fake import FakeCamera, FakeWorld, default_aois, demo_scenario
        world = FakeWorld.from_aois(aois or default_aois(s.width, s.height), s.width, s.height)
        return FakeCamera(world, demo_scenario(world), fps=min(float(s.fps), 15.0), realtime=True,
                          on_frame=on_frame)
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"No such source '{spec}' (camera index, 'synthetic', image or video)")
    if path.suffix.lower() in IMAGE_EXTS:
        return ImageSource(path, fps=min(float(s.fps), 15.0), on_frame=on_frame)
    return VideoFileSource(path, on_frame=on_frame, loop=True)
