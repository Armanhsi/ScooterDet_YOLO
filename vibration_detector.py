# vibration_detector.py
# Classifies road surface and detects pothole events from IMU accelerometer data.
# Uses a Z-axis spike detector for discrete events (potholes, cracks) and
# an FFT-based classifier for sustained surface texture (smooth/rough/cobblestone).
# Designed to feed both fusion_pipeline.py and the Unity VR CSV log.

import collections
import time
from dataclasses import dataclass, field

import numpy as np

# Spike detector: a pothole is a sudden Z-axis acceleration exceeding this
# threshold above the rolling baseline (m/s^2).
POTHOLE_SPIKE_THRESHOLD_MS2 = 8.0

# Minimum time between two pothole events to avoid double-counting (seconds)
POTHOLE_COOLDOWN_S = 0.5

# FFT window size (number of samples). At 200 Hz accel this is 0.5 seconds of data.
FFT_WINDOW_SIZE = 100

# FFT sample rate (must match ACCEL_RATE in imu_stream.py)
FFT_SAMPLE_RATE_HZ = 200

# Frequency band energy thresholds for surface classification (Hz ranges)
# Smooth road:      energy concentrated in 0-5 Hz
# Rough road:       elevated energy in 5-20 Hz
# Cobblestone:      elevated energy in 20-50 Hz + regular spacing
BAND_LOW_HZ   = (0.5, 5.0)
BAND_MID_HZ   = (5.0, 20.0)
BAND_HIGH_HZ  = (20.0, 50.0)

# Surface label thresholds (ratio of mid+high energy to total energy)
ROUGH_RATIO_THRESHOLD       = 0.35
COBBLESTONE_RATIO_THRESHOLD = 0.55


@dataclass
class VibrationEvent:
    # Represents a single detected vibration event for logging and Unity export.
    timestamp_ms:    float
    event_type:      str    # "pothole" | "surface_update"
    surface_class:   str    # "smooth" | "rough" | "cobblestone" | "unknown"
    accel_mag:       float  # peak or current acceleration magnitude (m/s^2)
    accel_x:         float
    accel_y:         float
    accel_z:         float
    gyro_mag:        float
    fft_energy_low:  float  # energy in 0.5-5 Hz band
    fft_energy_mid:  float  # energy in 5-20 Hz band
    fft_energy_high: float  # energy in 20-50 Hz band
    yolo_detections: list = field(default_factory=list)  # objects detected in same frame


class VibrationDetector:
    # Maintains a rolling window of IMU samples and produces VibrationEvents.
    # Call update() with each new IMU sample. Call get_event() to retrieve results.

    def __init__(self, window_size: int = FFT_WINDOW_SIZE,
                 sample_rate: int = FFT_SAMPLE_RATE_HZ):
        self._window_size  = window_size
        self._sample_rate  = sample_rate

        # Rolling buffers for each axis
        self._accel_z_buf  = collections.deque(maxlen=window_size)
        self._accel_x_buf  = collections.deque(maxlen=window_size)
        self._accel_y_buf  = collections.deque(maxlen=window_size)
        self._accel_mag_buf = collections.deque(maxlen=window_size)
        self._gyro_mag_buf  = collections.deque(maxlen=window_size)
        self._ts_buf        = collections.deque(maxlen=window_size)

        # Spike detection state
        self._baseline_z       = 0.0   # rolling mean of Z accel
        self._last_pothole_ts  = 0.0   # wall time of last pothole event

        # Current surface classification
        self._surface_class = "unknown"

        # Output event queue
        self._pending_events: list[VibrationEvent] = []

        # FFT frequency axis (precomputed)
        self._freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)

    def update(self, sample: dict, yolo_detections: list = None) -> None:
        # Feed one IMU sample into the detector. Triggers classification if window is full.
        if yolo_detections is None:
            yolo_detections = []

        ts   = sample["timestamp_ms"]
        az   = sample["accel_z"]
        ax   = sample["accel_x"]
        ay   = sample["accel_y"]
        amag = sample["accel_mag"]
        gmag = sample["gyro_mag"]

        self._accel_z_buf.append(az)
        self._accel_x_buf.append(ax)
        self._accel_y_buf.append(ay)
        self._accel_mag_buf.append(amag)
        self._gyro_mag_buf.append(gmag)
        self._ts_buf.append(ts)

        # Update rolling Z baseline (exponential moving average)
        alpha = 0.02
        self._baseline_z = alpha * az + (1 - alpha) * self._baseline_z

        # Spike check: detect pothole
        z_deviation = abs(az - self._baseline_z)
        now_wall    = time.time()
        if (z_deviation > POTHOLE_SPIKE_THRESHOLD_MS2 and
                now_wall - self._last_pothole_ts > POTHOLE_COOLDOWN_S):
            self._last_pothole_ts = now_wall
            e_low, e_mid, e_high = self._compute_fft_bands()
            event = VibrationEvent(
                timestamp_ms    = ts,
                event_type      = "pothole",
                surface_class   = self._surface_class,
                accel_mag       = amag,
                accel_x         = ax,
                accel_y         = ay,
                accel_z         = az,
                gyro_mag        = gmag,
                fft_energy_low  = e_low,
                fft_energy_mid  = e_mid,
                fft_energy_high = e_high,
                yolo_detections = yolo_detections,
            )
            self._pending_events.append(event)

        # FFT surface classification — run when window is full
        if len(self._accel_z_buf) == self._window_size:
            self._classify_surface(ts, ax, ay, az, amag, gmag, yolo_detections)

    def get_events(self) -> list:
        # Returns and clears all pending events since last call.
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    @property
    def current_surface(self) -> str:
        return self._surface_class

    def _compute_fft_bands(self) -> tuple:
        # Runs FFT on the Z-axis buffer and returns energy in each frequency band.
        if len(self._accel_z_buf) < self._window_size:
            return 0.0, 0.0, 0.0

        signal  = np.array(self._accel_z_buf, dtype=np.float32)
        signal -= signal.mean()  # remove DC offset
        window  = np.hanning(len(signal))
        fft_mag = np.abs(np.fft.rfft(signal * window))

        def band_energy(fmin, fmax):
            mask = (self._freqs >= fmin) & (self._freqs < fmax)
            return float(np.sum(fft_mag[mask] ** 2))

        e_low  = band_energy(*BAND_LOW_HZ)
        e_mid  = band_energy(*BAND_MID_HZ)
        e_high = band_energy(*BAND_HIGH_HZ)
        return e_low, e_mid, e_high

    def _classify_surface(self, ts, ax, ay, az, amag, gmag, yolo_detections):
        e_low, e_mid, e_high = self._compute_fft_bands()
        total = e_low + e_mid + e_high + 1e-9
        mid_high_ratio = (e_mid + e_high) / total
        high_ratio     = e_high / total

        if high_ratio > COBBLESTONE_RATIO_THRESHOLD:
            new_class = "cobblestone"
        elif mid_high_ratio > ROUGH_RATIO_THRESHOLD:
            new_class = "rough"
        else:
            new_class = "smooth"

        # Only emit a surface_update event when the class changes
        if new_class != self._surface_class:
            self._surface_class = new_class
            event = VibrationEvent(
                timestamp_ms    = ts,
                event_type      = "surface_update",
                surface_class   = new_class,
                accel_mag       = amag,
                accel_x         = ax,
                accel_y         = ay,
                accel_z         = az,
                gyro_mag        = gmag,
                fft_energy_low  = e_low,
                fft_energy_mid  = e_mid,
                fft_energy_high = e_high,
                yolo_detections = yolo_detections,
            )
            self._pending_events.append(event)


def event_to_dict(event: VibrationEvent) -> dict:
    # Converts a VibrationEvent to a flat dict suitable for CSV/JSON logging.
    return {
        "timestamp_ms":    round(event.timestamp_ms, 2),
        "event_type":      event.event_type,
        "surface_class":   event.surface_class,
        "accel_x":         round(event.accel_x, 4),
        "accel_y":         round(event.accel_y, 4),
        "accel_z":         round(event.accel_z, 4),
        "accel_mag":       round(event.accel_mag, 4),
        "gyro_mag":        round(event.gyro_mag, 4),
        "fft_energy_low":  round(event.fft_energy_low, 2),
        "fft_energy_mid":  round(event.fft_energy_mid, 2),
        "fft_energy_high": round(event.fft_energy_high, 2),
        "yolo_objects":    ",".join(event.yolo_detections) if event.yolo_detections else "",
    }


# Unity-compatible CSV column order
UNITY_CSV_COLUMNS = [
    "timestamp_ms", "event_type", "surface_class",
    "accel_x", "accel_y", "accel_z", "accel_mag",
    "gyro_mag",
    "fft_energy_low", "fft_energy_mid", "fft_energy_high",
    "yolo_objects",
]


if __name__ == "__main__":
    # Quick standalone test using dummy IMU data
    from imu_stream import IMUStream

    imu = IMUStream(maxsize=400)
    detector = VibrationDetector()
    imu.start()

    print("Running vibration detector for 12 seconds...")
    try:
        t_end = time.time() + 12.0
        while time.time() < t_end:
            sample = imu.get_sample(timeout=0.05)
            if sample is None:
                continue
            detector.update(sample)
            for event in detector.get_events():
                d = event_to_dict(event)
                print(f"  [{d['event_type'].upper():15s}] surface={d['surface_class']:12s} "
                      f"accel_mag={d['accel_mag']:.3f} m/s2 | "
                      f"FFT low={d['fft_energy_low']:.1f} mid={d['fft_energy_mid']:.1f} "
                      f"high={d['fft_energy_high']:.1f} | ts={d['timestamp_ms']:.0f}ms")
    except KeyboardInterrupt:
        pass
    finally:
        imu.stop()
