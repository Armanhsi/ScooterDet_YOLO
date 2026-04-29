# lidar_stream.py
# RPLidar S3 capture module. Reads scans in a background thread and pushes them
# into a queue for fusion_pipeline.py.
# Standalone test: python lidar_stream.py --port /dev/ttyUSB0  (or --dummy for no hardware)

import argparse
import math
import queue
import threading
import time

import numpy as np

try:
    from rplidar import RPLidar, RPLidarException
    RPLIDAR_AVAILABLE = True
except ImportError:
    RPLIDAR_AVAILABLE = False
    print("[WARN] rplidar package not installed. LidarStream will emit dummy scans.")

# Minimum quality threshold; S3 quality values range 0-15, reject low-confidence points
MIN_QUALITY = 5

# S3 max range is 25 m but we cap at a practical outdoor distance
MAX_DISTANCE_M = 10.0

# Motor speed setting (default for S3)
MOTOR_PWM = 660


class LidarStream:
    # Manages the RPLidar S3 in a background thread. Call start() before reading, stop() when done.

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 1000000,
                 maxsize: int = 4):
        # port: serial port (Linux: /dev/ttyUSB0, Windows: COM3). S3 baudrate is 1,000,000 bps.
        self._port = port
        self._baudrate = baudrate
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lidar = None

    def start(self):
        if not RPLIDAR_AVAILABLE:
            self._thread = threading.Thread(target=self._dummy_loop, daemon=True)
        else:
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._stop_event.clear()
        self._thread.start()
        print(f"[LiDAR] Stream started on {self._port}.")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._lidar is not None:
            try:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
            except Exception:
                pass
        print("[LiDAR] Stream stopped.")

    def get_scan(self, timeout: float = 0.2) -> dict | None:
        # Returns the latest scan dict. Drains stale entries so caller always gets the newest.
        scan = None
        try:
            while True:
                scan = self._queue.get_nowait()
        except queue.Empty:
            pass
        if scan is None:
            try:
                scan = self._queue.get(timeout=timeout)
            except queue.Empty:
                pass
        return scan

    def _push(self, scan: dict):
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(scan)

    def _capture_loop(self):
        try:
            lidar = RPLidar(self._port, baudrate=self._baudrate)
            self._lidar = lidar
            lidar.connect()
            info = lidar.get_info()
            health = lidar.get_health()
            print(f"[LiDAR] Info: {info}")
            print(f"[LiDAR] Health: {health}")
            lidar.start_motor()
            time.sleep(1.0)  # wait for motor to spin up

            for scan_data in lidar.iter_scans(max_buf_meas=500):
                if self._stop_event.is_set():
                    break

                ts = time.time() * 1000.0

                # Keep only points above quality threshold and within max range
                filtered = [
                    (q, a, d)
                    for q, a, d in scan_data
                    if q >= MIN_QUALITY and 0 < d <= MAX_DISTANCE_M * 1000.0
                ]

                polar = self._to_polar_array(filtered)

                self._push({
                    "timestamp_ms": ts,
                    "scan":         filtered,
                    "polar":        polar,
                })

        except RPLidarException as e:
            print(f"[LiDAR] RPLidarException: {e}")
        except Exception as e:
            print(f"[LiDAR] Unexpected error: {e}")

    def _dummy_loop(self):
        # Emits synthetic 360-degree scans for testing without hardware.
        print("[LiDAR] Running in DUMMY mode (no hardware).")
        while not self._stop_event.is_set():
            angles = np.linspace(0, 359.9, 360)
            # Simulated: 2m ahead, increasing at the sides
            distances_mm = (2000 + 1000 * np.abs(np.sin(np.deg2rad(angles)))).tolist()

            scan_data = [(15, float(a), float(d)) for a, d in zip(angles, distances_mm)]
            polar = self._to_polar_array(scan_data)

            self._push({
                "timestamp_ms": time.time() * 1000.0,
                "scan":         scan_data,
                "polar":        polar,
            })
            time.sleep(0.1)  # ~10 scans/sec (S3 does ~10 Hz)

    @staticmethod
    def _to_polar_array(scan_data: list) -> np.ndarray:
        # Converts raw scan tuples to a (N, 2) float32 array of [angle_rad, distance_m].
        if not scan_data:
            return np.empty((0, 2), dtype=np.float32)
        arr = np.array([[math.radians(a), d / 1000.0] for _, a, d in scan_data],
                       dtype=np.float32)
        return arr


def get_lidar_distance_in_fov(polar: np.ndarray,
                               fov_center_deg: float,
                               fov_half_width_deg: float = 15.0) -> float:
    # Returns the closest LiDAR distance (meters) within the given angular sector.
    # Returns 0.0 if no points found. fov_center_deg is 0=forward, increasing clockwise.
    if polar is None or len(polar) == 0:
        return 0.0

    center_rad = math.radians(fov_center_deg % 360.0)
    half_rad   = math.radians(fov_half_width_deg)

    angles = polar[:, 0]

    # Compute angular difference with wrap-around at 0/2pi
    diff = np.abs(angles - center_rad)
    diff = np.minimum(diff, 2 * math.pi - diff)

    mask = diff <= half_rad
    if not np.any(mask):
        return 0.0

    return float(np.min(polar[mask, 1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPLidar S3 stream test")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0",
                        help="Serial port for RPLidar S3")
    parser.add_argument("--dummy", action="store_true",
                        help="Run in dummy mode without hardware")
    args = parser.parse_args()

    if args.dummy or not RPLIDAR_AVAILABLE:
        stream = LidarStream(port=args.port)
        stream._capture_loop = stream._dummy_loop  # force dummy
    else:
        stream = LidarStream(port=args.port)

    stream.start()
    print("Reading scans for 10 seconds... (Ctrl+C to stop)")
    try:
        t_end = time.time() + 10.0
        scan_count = 0
        while time.time() < t_end:
            scan = stream.get_scan(timeout=0.5)
            if scan is None:
                continue
            scan_count += 1
            polar = scan["polar"]
            ts    = scan["timestamp_ms"]
            nearest = float(np.min(polar[:, 1])) if len(polar) else float("inf")
            print(f"  Scan #{scan_count:03d} | points={len(polar):4d} | "
                  f"nearest={nearest:.2f}m | ts={ts:.0f}ms")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
