"""Optional MJPEG preview of the live pipeline, for tuning thresholds.

Binds to 127.0.0.1 by default: the camera image never leaves the Pi unless
you tunnel it (ssh -L 8080:localhost:8080 <pi>). Frames are encoded at most
`max_fps` times per second so the preview costs little inference time.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class PreviewServer:
    def __init__(self, port: int = 8080, host: str = "127.0.0.1",
                 max_fps: float = 10.0) -> None:
        import cv2  # pyright: ignore[reportMissingImports]

        self._cv2 = cv2
        self._jpeg = b""
        self._lock = threading.Lock()
        self._interval = 1.0 / max_fps
        self._last = 0.0
        self._closed = False
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while not preview._closed:
                        with preview._lock:
                            jpg = preview._jpeg
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg"
                                             b"\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(preview._interval)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args) -> None:  # keep the console quiet
                pass

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True,
                         name="preview").start()

    def update(self, frame, top: list[tuple[str, float]], *,
               threshold: float, active: str | None) -> None:
        """Annotate and publish a frame. Rate-limited, cheap when skipped."""
        now = time.monotonic()
        if now - self._last < self._interval:
            return
        self._last = now

        cv2 = self._cv2
        img = frame.copy()
        h, w = img.shape[:2]
        m = (w - h) // 2  # the model only sees this centre square (fit="crop")
        cv2.rectangle(img, (m, 0), (m + h, h), (255, 255, 255), 1)
        for i, (label, score) in enumerate(top):
            color = (0, 255, 0) if score >= threshold else (0, 200, 255)
            cv2.putText(img, f"{label}: {score:.2f}", (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"episode: {active or '-'}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._lock:
                self._jpeg = jpg.tobytes()

    def close(self) -> None:
        self._closed = True
        self._httpd.shutdown()
        self._httpd.server_close()