# Live fusion pipeline: D435i RGB+depth + RPLidar S3 + YOLO11n.
# Priority queue syncs sensor frames by timestamp. Fused distance per detection
# is weighted RS/LiDAR (RS preferred <3m, LiDAR preferred >=3m).
# Outputs annotated display window and CSV/JSONL logs.
# Usage: python fusion_pipeline.py [--weights yolo11n.pt] [--conf 0.3] [--lidar-port /dev/ttyUSB0]

import argparse
import csv
import heapq
import json
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from realsense_stream import RealSenseStream, sample_depth_at_bbox, REALSENSE_MAX_RELIABLE_DEPTH_M
from lidar_stream import LidarStream, get_lidar_distance_in_fov

# Constants
CAMERA_HFOV_DEG = 87.0             # D435i horizontal FOV (degrees)
SYNC_WINDOW_MS = 150.0             # max timestamp gap (ms) to treat two sensor readings as synced
REALSENSE_PREFERRED_BELOW_M = 3.0  # RS is more accurate below this distance
WEIGHT_RS_NEAR  = 0.7              # RS weight when object is close (<3m)
WEIGHT_RS_FAR   = 0.3              # RS weight when object is far (>=3m)
LIDAR_FOV_HALF_DEG = 12.0          # half-angle (degrees) of LiDAR acceptance cone per detection
DISPLAY_SCALE = 0.8                # scale factor for the display window

# Box/label colors per distance source (BGR)
COLOR_BOX   = (0, 220, 0)
COLOR_TEXT  = (255, 255, 255)
COLOR_WARN  = (0, 60, 200)
COLOR_LIDAR = (0, 180, 255)
COLOR_RS    = (255, 140, 0)
COLOR_FUSED = (180, 0, 255)


# Priority queue item
class PQItem:
    # Timestamped wrapper for sensor data. Sorted ascending so heappop gives oldest.
    # The pipeline always drains to newest before inferring to minimize latency.
    __slots__ = ("timestamp_ms", "source", "data", "_seq")
    _counter = 0
    _lock = threading.Lock()

    def __init__(self, timestamp_ms: float, source: str, data: dict):
        self.timestamp_ms = timestamp_ms
        self.source       = source   # "realsense" | "lidar"
        self.data         = data
        with PQItem._lock:
            PQItem._counter += 1
            self._seq = PQItem._counter

    def __lt__(self, other):
        if self.timestamp_ms == other.timestamp_ms:
            return self._seq < other._seq
        return self.timestamp_ms < other.timestamp_ms


# Utility functions
def pixel_to_bearing(cx_pixel: int, frame_width: int, hfov_deg: float) -> float:
    # Converts bbox center x-pixel to bearing angle (degrees). 0=ahead, negative=left, positive=right.
    center = frame_width / 2.0
    frac   = (cx_pixel - center) / frame_width   # -0.5 to +0.5
    return frac * hfov_deg


def fuse_distances(rs_depth: float, lidar_dist: float) -> tuple:
    # Combines RS depth and LiDAR distance. Returns (fused_m, source_label).
    # source_label is "RS", "LiDAR", or "Fused" depending on which sources are valid.
    rs_valid    = rs_depth    > 0.0
    lidar_valid = lidar_dist  > 0.0

    if rs_valid and lidar_valid:
        if rs_depth < REALSENSE_PREFERRED_BELOW_M:
            w_rs, w_li = WEIGHT_RS_NEAR, 1.0 - WEIGHT_RS_NEAR
        else:
            w_rs, w_li = WEIGHT_RS_FAR,  1.0 - WEIGHT_RS_FAR
        fused = w_rs * rs_depth + w_li * lidar_dist
        return fused, "Fused"

    if rs_valid:
        return rs_depth, "RS"

    if lidar_valid:
        return lidar_dist, "LiDAR"

    return 0.0, "N/A"


def draw_detection(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   class_name: str, conf: float, fused_dist: float,
                   source_label: str) -> np.ndarray:
    # Draws a bounding box with class, confidence, fused distance, and source label.
    if source_label == "Fused":
        box_color = COLOR_FUSED
    elif source_label == "RS":
        box_color = COLOR_RS
    elif source_label == "LiDAR":
        box_color = COLOR_LIDAR
    else:
        box_color = COLOR_BOX

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    dist_str = f"{fused_dist:.2f}m" if fused_dist > 0 else "dist=N/A"
    label    = f"{class_name} {conf:.2f} | {dist_str} [{source_label}]"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, fps: float, frame_id: int,
             n_rs: int, n_lidar: int, n_det: int) -> np.ndarray:
    # Draws FPS, frame count, queue sizes, and detection count in the top-left corner.
    lines = [
        f"FPS: {fps:.1f}",
        f"Frame: {frame_id}",
        f"RS queue: {n_rs}",
        f"LiDAR queue: {n_lidar}",
        f"Detections: {n_det}",
    ]
    y = 22
    for line in lines:
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (240, 240, 240), 1, cv2.LINE_AA)
        y += 22
    return frame


# Logger
class DetectionLogger:
    # Writes each detection to a timestamped CSV and JSONL file in log_dir.

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path   = log_dir / f"detections_{ts}.csv"
        jsonl_path = log_dir / f"detections_{ts}.jsonl"

        self._csv_file   = open(csv_path,   "w", newline="")
        self._jsonl_file = open(jsonl_path, "w")

        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "wall_time", "frame_id", "class_id", "class_name",
            "confidence", "x1", "y1", "x2", "y2",
            "bbox_cx", "bbox_cy",
            "rs_depth_m", "lidar_dist_m", "fused_dist_m", "source",
            "camera_bearing_deg"
        ])

        print(f"[Logger] CSV   -> {csv_path}")
        print(f"[Logger] JSONL -> {jsonl_path}")

    def log(self, frame_id: int, class_id: int, class_name: str, conf: float,
            x1: int, y1: int, x2: int, y2: int,
            rs_depth: float, lidar_dist: float, fused_dist: float,
            source: str, bearing_deg: float):
        wall_time = datetime.now().isoformat(timespec="milliseconds")
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        self._csv_writer.writerow([
            wall_time, frame_id, class_id, class_name,
            f"{conf:.4f}", x1, y1, x2, y2, cx, cy,
            f"{rs_depth:.4f}", f"{lidar_dist:.4f}", f"{fused_dist:.4f}",
            source, f"{bearing_deg:.2f}"
        ])

        record = {
            "wall_time":      wall_time,
            "frame_id":       frame_id,
            "class_id":       class_id,
            "class_name":     class_name,
            "confidence":     round(conf, 4),
            "bbox":           [x1, y1, x2, y2],
            "bbox_center":    [cx, cy],
            "rs_depth_m":     round(rs_depth, 4),
            "lidar_dist_m":   round(lidar_dist, 4),
            "fused_dist_m":   round(fused_dist, 4),
            "source":         source,
            "bearing_deg":    round(bearing_deg, 2),
        }
        self._jsonl_file.write(json.dumps(record) + "\n")

    def flush(self):
        self._csv_file.flush()
        self._jsonl_file.flush()

    def close(self):
        self._csv_file.close()
        self._jsonl_file.close()


# Fusion pipeline
class FusionPipeline:

    def __init__(self, args):
        self.args = args
        self.hfov = args.camera_hfov

        print(f"[Pipeline] Loading YOLO model: {args.weights}")
        self.model = YOLO(args.weights)
        self.class_names = (
            self.model.names if isinstance(self.model.names, list)
            else list(self.model.names.values())
        )
        print(f"[Pipeline] Classes: {self.class_names}")

        self.rs_stream    = RealSenseStream(maxsize=4)
        self.lidar_stream = LidarStream(port=args.lidar_port, maxsize=4)

        if args.realsense_dummy:
            self.rs_stream._capture_loop = self.rs_stream._dummy_loop

        if args.lidar_dummy or not self._lidar_hw_available():
            self.lidar_stream._capture_loop = self.lidar_stream._dummy_loop

        self.pq: list = []          # heapq of PQItem
        self.pq_lock  = threading.Lock()

        log_dir = Path(args.log_dir)
        self.logger = DetectionLogger(log_dir)

        self._frame_id   = 0
        self._fps        = 0.0
        self._t_last_fps = time.perf_counter()
        self._fps_count  = 0

    @staticmethod
    def _lidar_hw_available() -> bool:
        try:
            from rplidar import RPLidar  # noqa: F401
            return True
        except ImportError:
            return False

    def _rs_feeder(self):
        # Background thread: pulls RealSense frames and pushes them into the priority queue.
        while not self._stop.is_set():
            frame = self.rs_stream.get_frame(timeout=0.1)
            if frame is None:
                continue
            item = PQItem(frame["timestamp_ms"], "realsense", frame)
            with self.pq_lock:
                heapq.heappush(self.pq, item)

    def _lidar_feeder(self):
        # Background thread: pulls LiDAR scans and pushes them into the priority queue.
        while not self._stop.is_set():
            scan = self.lidar_stream.get_scan(timeout=0.2)
            if scan is None:
                continue
            item = PQItem(scan["timestamp_ms"], "lidar", scan)
            with self.pq_lock:
                heapq.heappush(self.pq, item)

    def _drain_pq(self) -> tuple:
        # Drains the priority queue and returns the newest RS frame and LiDAR scan.
        # Ensures inference always runs on the most recent data.
        latest_rs    = None
        latest_lidar = None

        with self.pq_lock:
            while self.pq:
                item = heapq.heappop(self.pq)
                if item.source == "realsense":
                    latest_rs    = item.data
                elif item.source == "lidar":
                    latest_lidar = item.data

        return latest_rs, latest_lidar

    def _pq_sizes(self) -> tuple:
        with self.pq_lock:
            rs_n    = sum(1 for i in self.pq if i.source == "realsense")
            lidar_n = sum(1 for i in self.pq if i.source == "lidar")
        return rs_n, lidar_n

    def run(self):
        self._stop = threading.Event()

        self.rs_stream.start()
        self.lidar_stream.start()

        rs_feeder_t    = threading.Thread(target=self._rs_feeder,    daemon=True)
        lidar_feeder_t = threading.Thread(target=self._lidar_feeder, daemon=True)
        rs_feeder_t.start()
        lidar_feeder_t.start()

        print("[Pipeline] Running. Press 'q' in display window or Ctrl+C to stop.")
        try:
            while True:
                rs_frame, lidar_scan = self._drain_pq()

                if rs_frame is None:
                    time.sleep(0.005)
                    continue

                color_bgr = rs_frame["color"]
                depth_map = rs_frame["depth"]
                frame_h, frame_w = color_bgr.shape[:2]

                # Check if the two sensor readings are within the sync window
                sync_ok = False
                if lidar_scan is not None:
                    dt = abs(rs_frame["timestamp_ms"] - lidar_scan["timestamp_ms"])
                    sync_ok = dt < SYNC_WINDOW_MS

                results      = self.model(color_bgr, conf=self.args.conf, verbose=False)
                annotated    = color_bgr.copy()
                n_detections = len(results[0].boxes)

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf     = float(box.conf[0])
                    cls_id   = int(box.cls[0])
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)

                    # Sample RS depth at bbox center and convert bearing to LiDAR angle convention
                    rs_depth     = sample_depth_at_bbox(depth_map, x1, y1, x2, y2)
                    bbox_cx      = (x1 + x2) / 2.0
                    bearing_deg  = pixel_to_bearing(bbox_cx, frame_w, self.hfov)
                    lidar_angle  = (bearing_deg + 360.0) % 360.0

                    lidar_dist = 0.0
                    if lidar_scan is not None and sync_ok:
                        lidar_dist = get_lidar_distance_in_fov(
                            lidar_scan["polar"],
                            fov_center_deg=lidar_angle,
                            fov_half_width_deg=LIDAR_FOV_HALF_DEG
                        )

                    fused_dist, source_label = fuse_distances(rs_depth, lidar_dist)

                    draw_detection(annotated, x1, y1, x2, y2,
                                   cls_name, conf, fused_dist, source_label)

                    self.logger.log(
                        frame_id=self._frame_id,
                        class_id=cls_id, class_name=cls_name, conf=conf,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        rs_depth=rs_depth, lidar_dist=lidar_dist,
                        fused_dist=fused_dist, source=source_label,
                        bearing_deg=bearing_deg
                    )

                # Update FPS counter and flush logs once per second
                self._fps_count += 1
                now = time.perf_counter()
                if now - self._t_last_fps >= 1.0:
                    self._fps      = self._fps_count / (now - self._t_last_fps)
                    self._fps_count = 0
                    self._t_last_fps = now
                    self.logger.flush()

                n_rs, n_lidar = self._pq_sizes()
                draw_hud(annotated, self._fps, self._frame_id, n_rs, n_lidar, n_detections)

                # Show sync status on frame
                sync_text  = "SYNC OK" if sync_ok else "NO SYNC"
                sync_color = (0, 200, 0) if sync_ok else (0, 0, 200)
                cv2.putText(annotated, sync_text, (frame_w - 120, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, sync_color, 2, cv2.LINE_AA)

                self._frame_id += 1

                if not self.args.no_display:
                    disp = cv2.resize(annotated,
                                      (int(frame_w * DISPLAY_SCALE),
                                       int(frame_h * DISPLAY_SCALE)))
                    cv2.imshow("Fusion Pipeline", disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted by user.")
        finally:
            self._stop.set()
            self.rs_stream.stop()
            self.lidar_stream.stop()
            self.logger.close()
            if not self.args.no_display:
                cv2.destroyAllWindows()
            print(f"[Pipeline] Done. Processed {self._frame_id} frames.")


# Entry point
def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11n + RealSense D435i + RPLidar S3 fusion")
    parser.add_argument("--weights",          type=str,   default="yolo11n.pt",
                        help="YOLO model weights (default: yolo11n.pt)")
    parser.add_argument("--conf",             type=float, default=0.3,
                        help="YOLO confidence threshold (default: 0.3)")
    parser.add_argument("--lidar-port",       type=str,   default="/dev/ttyUSB0",
                        help="Serial port for RPLidar S3 (default: /dev/ttyUSB0)")
    parser.add_argument("--camera-hfov",      type=float, default=CAMERA_HFOV_DEG,
                        help=f"Camera horizontal FOV in degrees (default: {CAMERA_HFOV_DEG})")
    parser.add_argument("--log-dir",          type=str,   default="logs",
                        help="Directory for CSV and JSONL logs (default: logs/)")
    parser.add_argument("--lidar-dummy",      action="store_true",
                        help="Use dummy LiDAR data (no hardware)")
    parser.add_argument("--realsense-dummy",  action="store_true",
                        help="Use dummy RealSense data (no hardware)")
    parser.add_argument("--no-display",       action="store_true",
                        help="Disable OpenCV display window (headless mode)")
    return parser.parse_args()


if __name__ == "__main__":
    FusionPipeline(parse_args()).run()
