# realsense_stream.py
# D435i capture module. Grabs aligned RGB + depth frames in a background thread
# and pushes them into a queue for fusion_pipeline.py.
# Standalone test: python realsense_stream.py

import queue
import threading
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARN] pyrealsense2 not installed. RealSenseStream will emit dummy frames.")


# Depth is unreliable beyond this distance (D435i spec: ~3m accurate)
REALSENSE_MAX_RELIABLE_DEPTH_M = 3.0

# Resolution and frame rate
COLOR_WIDTH  = 848
COLOR_HEIGHT = 480
DEPTH_WIDTH  = 848
DEPTH_HEIGHT = 480
FPS = 30


class RealSenseStream:
    # Manages the D435i pipeline in a background thread. Call start() before reading, stop() when done.

    def __init__(self, maxsize: int = 4):
        # maxsize: queue capacity; oldest frame dropped when full so consumer always gets newest.
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pipeline = None
        self._depth_scale: float = 0.001  # default RealSense depth scale (mm -> m)

    def start(self):
        if not REALSENSE_AVAILABLE:
            self._thread = threading.Thread(target=self._dummy_loop, daemon=True)
        else:
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._stop_event.clear()
        self._thread.start()
        print("[RealSense] Stream started.")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._pipeline is not None:
            self._pipeline.stop()
        print("[RealSense] Stream stopped.")

    def get_frame(self, timeout: float = 0.1) -> dict | None:
        # Returns the latest frame dict. Drains stale frames so caller always gets the newest.
        frame = None
        try:
            while True:
                frame = self._queue.get_nowait()
        except queue.Empty:
            pass
        if frame is None:
            try:
                frame = self._queue.get(timeout=timeout)
            except queue.Empty:
                pass
        return frame

    def _push(self, frame: dict):
        # Non-blocking push; drops oldest frame if queue is full.
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(frame)

    def _capture_loop(self):
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT,
                             rs.format.bgr8, FPS)
        config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT,
                             rs.format.z16, FPS)

        profile = pipeline.start(config)
        self._pipeline = pipeline

        # Get depth scale (raw units to meters) and align depth to color frame
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        align = rs.align(rs.stream.color)

        # Spatial + temporal filters to reduce depth noise
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        temporal = rs.temporal_filter()

        print(f"[RealSense] Depth scale: {self._depth_scale:.6f} m/unit")

        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                aligned = align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)

                color_image = np.asanyarray(color_frame.get_data())

                # Convert depth to meters and zero out unreliable readings
                depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                depth_meters = depth_raw * self._depth_scale
                depth_meters[depth_meters > REALSENSE_MAX_RELIABLE_DEPTH_M] = 0.0

                ts_ms = color_frame.get_timestamp()

                self._push({
                    "timestamp_ms": ts_ms,
                    "color":        color_image,
                    "depth":        depth_meters,
                    "depth_scale":  self._depth_scale,
                })
        finally:
            pipeline.stop()

    def _dummy_loop(self):
        # Emits synthetic frames for testing without hardware.
        print("[RealSense] Running in DUMMY mode (no hardware).")
        frame_id = 0
        while not self._stop_event.is_set():
            h, w = COLOR_HEIGHT, COLOR_WIDTH
            color = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(color, f"DUMMY FRAME {frame_id}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Synthetic depth gradient: 0.5m (left) to 5m (right)
            depth = np.tile(np.linspace(0.5, 5.0, w, dtype=np.float32), (h, 1))
            depth[depth > REALSENSE_MAX_RELIABLE_DEPTH_M] = 0.0

            self._push({
                "timestamp_ms": time.time() * 1000.0,
                "color":        color,
                "depth":        depth,
                "depth_scale":  0.001,
            })
            frame_id += 1
            time.sleep(1.0 / FPS)


def sample_depth_at_bbox(depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                          patch_frac: float = 0.3) -> float:
    # Returns median depth (meters) from a center patch of the bbox.
    # Using a center patch avoids background pixels at box edges. Returns 0.0 if no valid data.
    h, w = depth_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.0

    # Shrink bbox to its center patch
    margin_x = int(bw * patch_frac / 2)
    margin_y = int(bh * patch_frac / 2)
    cx1 = x1 + margin_x
    cy1 = y1 + margin_y
    cx2 = x2 - margin_x
    cy2 = y2 - margin_y

    patch = depth_map[cy1:cy2, cx1:cx2]
    valid = patch[patch > 0.0]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


if __name__ == "__main__":
    stream = RealSenseStream(maxsize=2)
    stream.start()
    print("Press 'q' to quit.")
    try:
        while True:
            frame = stream.get_frame(timeout=1.0)
            if frame is None:
                print("[MAIN] Waiting for frame...")
                continue

            color = frame["color"]
            depth = frame["depth"]
            ts    = frame["timestamp_ms"]

            # Colorize depth map for display
            depth_display = cv2.convertScaleAbs(depth, alpha=255.0 / REALSENSE_MAX_RELIABLE_DEPTH_M)
            depth_colorized = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            combined = np.hstack([color, depth_colorized])
            cv2.putText(combined, f"ts={ts:.0f}ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("RealSense: Color | Depth", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
