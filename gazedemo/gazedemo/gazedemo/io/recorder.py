"""video.mp4 + frames.csv for every captured frame (own thread), so `offline` can replay at full fps."""
from __future__ import annotations

import csv
import queue
import threading
from pathlib import Path

import cv2


class Recorder:
    """blocking=False (live camera): a full queue drops the frame rather than stall capture.
    blocking=True (virtual-time sources such as FakeCamera): wait, so every frame is recorded."""

    def __init__(self, run_dir, fps: float = 30.0, codec: str = "mp4v", max_queue: int = 64,
                 blocking: bool = False):
        self.dir = Path(run_dir)
        self.fps = float(fps or 30.0)
        self.codec = codec
        self.blocking = blocking
        self.written = 0
        self.dropped = 0
        self.error = None
        self._q: queue.Queue = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._loop, name="recorder", daemon=True)
        self._thread.start()

    def push(self, idx: int, t: float, frame) -> None:
        """Called from the camera thread for every frame."""
        if self.blocking:
            self._q.put((idx, t, frame))
            return
        try:
            self._q.put_nowait((idx, t, frame))
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=60)
        if self.dropped:
            print(f"[recorder] dropped {self.dropped} frames (slow disk/encoder)")

    def _loop(self):
        writer = None
        with open(self.dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
            out = csv.writer(f)
            out.writerow(["video_idx", "frame_idx", "t"])
            while True:
                item = self._q.get()
                if item is None:
                    break
                idx, t, frame = item
                if self.error:
                    continue
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(self.dir / "video.mp4"),
                                             cv2.VideoWriter_fourcc(*self.codec), self.fps, (w, h))
                    if not writer.isOpened():
                        self.error = f"cannot open VideoWriter ({self.codec})"
                        print(f"[recorder] {self.error} → not recording video")
                        continue
                writer.write(frame)
                out.writerow([self.written, idx, t])
                self.written += 1
                if self.written % 30 == 0:
                    f.flush()
        if writer is not None:
            writer.release()
