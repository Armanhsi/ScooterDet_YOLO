# imu_stream.py
# Captures accelerometer and gyroscope data from the Intel RealSense D435i IMU.
# Runs two background threads (one per sensor) and merges readings into a queue
# consumed by fusion_pipeline.py and vibration_detector.py.
# Standalone test: python imu_stream.py

import queue
import threading
import time

import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARN] pyrealsense2 not installed. IMUStream will emit dummy data.")

# D435i IMU sample rates
ACCEL_RATE = 200   # Hz (options: 63, 200)
GYRO_RATE  = 400   # Hz (options: 200, 400)

# Gravity constant for normalizing accelerometer data
GRAVITY_MS2 = 9.80665


class IMUStream:
    # Captures accel + gyro from D435i in background threads.
    # Call start() before reading, stop() when done.
    # Each frame in the queue is a dict with merged accel + gyro + timestamps.

    def __init__(self, maxsize: int = 200):
        # maxsize: internal queue depth. At 200 Hz accel this holds ~1 second of data.
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._pipeline = None
        self._thread: threading.Thread | None = None

        # Latest readings shared between the two sensor callbacks
        self._latest_accel = {"x": 0.0, "y": 0.0, "z": 0.0, "ts": 0.0}
        self._latest_gyro  = {"x": 0.0, "y": 0.0, "z": 0.0, "ts": 0.0}
        self._lock = threading.Lock()

    def start(self):
        if not REALSENSE_AVAILABLE:
            self._thread = threading.Thread(target=self._dummy_loop, daemon=True)
        else:
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._stop_event.clear()
        self._thread.start()
        print("[IMU] Stream started.")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        print("[IMU] Stream stopped.")

    def get_sample(self, timeout: float = 0.05) -> dict | None:
        # Returns the latest IMU sample dict, or None if none available.
        # Drains stale samples so caller gets the most recent reading.
        sample = None
        try:
            while True:
                sample = self._queue.get_nowait()
        except queue.Empty:
            pass
        if sample is None:
            try:
                sample = self._queue.get(timeout=timeout)
            except queue.Empty:
                pass
        return sample

    def get_batch(self, n: int) -> list:
        # Returns up to n recent samples as a list (oldest first).
        # Used by vibration_detector for FFT windowing.
        batch = []
        try:
            while len(batch) < n:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        return batch

    def _push(self, sample: dict):
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(sample)

    def _capture_loop(self):
        pipeline = rs.pipeline()
        config   = rs.config()
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, ACCEL_RATE)
        config.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, GYRO_RATE)

        pipeline.start(config, self._frame_callback)
        self._pipeline = pipeline
        print(f"[IMU] Accel @ {ACCEL_RATE} Hz | Gyro @ {GYRO_RATE} Hz")

        # Block until stop is requested; callbacks handle all data
        self._stop_event.wait()
        pipeline.stop()

    def _frame_callback(self, frame):
        # Called by the RealSense SDK for every IMU frame on a SDK-managed thread.
        if frame.is_motion_frame():
            motion = frame.as_motion_frame()
            data   = motion.get_motion_data()
            ts_ms  = frame.get_timestamp()
            profile = frame.get_profile()

            with self._lock:
                if profile.stream_type() == rs.stream.accel:
                    self._latest_accel = {
                        "x": data.x, "y": data.y, "z": data.z, "ts": ts_ms
                    }
                elif profile.stream_type() == rs.stream.gyro:
                    self._latest_gyro = {
                        "x": data.x, "y": data.y, "z": data.z, "ts": ts_ms
                    }

            # Merge latest accel + gyro into one sample and push
            with self._lock:
                accel = dict(self._latest_accel)
                gyro  = dict(self._latest_gyro)

            # Compute scalar magnitude of acceleration (subtract gravity on Z if mounted upright)
            accel_mag = float(np.sqrt(accel["x"]**2 + accel["y"]**2 + accel["z"]**2))
            gyro_mag  = float(np.sqrt(gyro["x"]**2  + gyro["y"]**2  + gyro["z"]**2))

            self._push({
                "timestamp_ms":  ts_ms,
                "accel_x":       accel["x"],
                "accel_y":       accel["y"],
                "accel_z":       accel["z"],
                "accel_mag":     accel_mag,
                "gyro_x":        gyro["x"],
                "gyro_y":        gyro["y"],
                "gyro_z":        gyro["z"],
                "gyro_mag":      gyro_mag,
            })

    def _dummy_loop(self):
        # Emits synthetic IMU data for testing without hardware.
        print("[IMU] Running in DUMMY mode (no hardware).")
        t0 = time.time()
        while not self._stop_event.is_set():
            t = time.time() - t0
            ts_ms = t * 1000.0

            # Simulate smooth road with occasional pothole spike at t=3s and t=7s
            spike = 15.0 if (2.9 < t < 3.1 or 6.9 < t < 7.1) else 0.0
            accel_z   = -GRAVITY_MS2 + 0.3 * np.sin(2 * np.pi * 5 * t) + spike
            accel_x   = 0.1 * np.sin(2 * np.pi * 1.2 * t)
            accel_y   = 0.1 * np.cos(2 * np.pi * 0.8 * t)
            accel_mag = float(np.sqrt(accel_x**2 + accel_y**2 + accel_z**2))

            gyro_x = 0.02 * np.sin(2 * np.pi * 0.5 * t)
            gyro_y = 0.01 * np.cos(2 * np.pi * 0.3 * t)
            gyro_z = 0.005 * np.sin(2 * np.pi * 0.2 * t)
            gyro_mag = float(np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2))

            self._push({
                "timestamp_ms": ts_ms,
                "accel_x":      float(accel_x),
                "accel_y":      float(accel_y),
                "accel_z":      float(accel_z),
                "accel_mag":    accel_mag,
                "gyro_x":       float(gyro_x),
                "gyro_y":       float(gyro_y),
                "gyro_z":       float(gyro_z),
                "gyro_mag":     gyro_mag,
            })
            time.sleep(1.0 / ACCEL_RATE)


if __name__ == "__main__":
    stream = IMUStream(maxsize=400)
    stream.start()
    print("Reading IMU for 10 seconds... (Ctrl+C to stop)")
    try:
        t_end = time.time() + 10.0
        count = 0
        while time.time() < t_end:
            sample = stream.get_sample(timeout=0.1)
            if sample is None:
                continue
            count += 1
            if count % 20 == 0:  # print every 20th sample to avoid flooding
                print(f"  accel=({sample['accel_x']:+.3f}, {sample['accel_y']:+.3f}, "
                      f"{sample['accel_z']:+.3f}) mag={sample['accel_mag']:.3f} m/s2 | "
                      f"gyro mag={sample['gyro_mag']:.4f} rad/s | "
                      f"ts={sample['timestamp_ms']:.0f}ms")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
