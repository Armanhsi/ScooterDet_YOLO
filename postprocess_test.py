# postprocess_test.py
# Runs YOLO11n on a folder of static image frames. No hardware required.
# Usage: python postprocess_test.py --images <path> --weights yolo11n.pt --conf 0.25 --save-csv

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

CLASS_COLORS = {}


def get_color(class_id: int) -> tuple:
    # Returns a consistent BGR color for the given class index.
    if class_id not in CLASS_COLORS:
        import random
        rng = random.Random(class_id * 37)
        CLASS_COLORS[class_id] = (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))
    return CLASS_COLORS[class_id]


def draw_detections(frame: "cv2.Mat", results, class_names: list) -> "cv2.Mat":
    # Draws bounding boxes and class labels onto the frame.
    annotated = frame.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        color = get_color(cls_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def run(args):
    images_dir = Path(args.images)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = sorted([
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ])
    if not image_paths:
        raise RuntimeError(f"No supported images found in {images_dir}")

    print(f"[INFO] Loaded {len(image_paths)} images from {images_dir}")

    model = YOLO(args.weights)
    print(f"[INFO] Model loaded: {args.weights}")

    class_names = model.names if isinstance(model.names, list) else list(model.names.values())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "detections.csv"
    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_id", "image_name", "class_id", "class_name",
            "confidence", "x1", "y1", "x2", "y2",
            "bbox_center_x", "bbox_center_y", "inference_ms"
        ])

    total_frames = len(image_paths)
    total_detections = 0
    total_time_ms = 0.0

    for idx, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read {img_path.name}, skipping.")
            continue

        t0 = time.perf_counter()
        results = model(frame, conf=args.conf, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        total_time_ms += inference_ms

        boxes = results[0].boxes
        num_det = len(boxes)
        total_detections += num_det

        print(f"[{idx+1:04d}/{total_frames}] {img_path.name} | "
              f"detections={num_det} | inference={inference_ms:.1f}ms")

        if csv_writer is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                csv_writer.writerow([
                    idx, img_path.name, cls_id, class_name,
                    f"{conf:.4f}", x1, y1, x2, y2, cx, cy,
                    f"{inference_ms:.2f}"
                ])

        annotated = draw_detections(frame, results, class_names)
        save_path = annotated_dir / img_path.name
        cv2.imwrite(str(save_path), annotated)

        if args.show:
            cv2.imshow("PostProcess Test", annotated)
            key = cv2.waitKey(1 if args.auto else 0)
            if key == ord("q"):
                print("[INFO] Quit requested.")
                break

    if csv_file is not None:
        csv_file.close()
        print(f"[INFO] CSV saved to {csv_path}")

    if args.show:
        cv2.destroyAllWindows()

    avg_ms = total_time_ms / max(total_frames, 1)
    print(f"\n[SUMMARY]")
    print(f"  Frames processed : {total_frames}")
    print(f"  Total detections : {total_detections}")
    print(f"  Avg inference    : {avg_ms:.1f} ms ({1000/max(avg_ms,0.001):.1f} FPS theoretical)")
    print(f"  Annotated frames : {annotated_dir}")
    if args.save_csv:
        print(f"  Detection CSV    : {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline YOLO11n post-processing on image frames")
    parser.add_argument("--images", type=str, required=True,
                        help="Path to directory containing image frames")
    parser.add_argument("--weights", type=str, default="yolo11n.pt",
                        help="YOLO model weights (default: yolo11n.pt, auto-downloaded)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for annotated images and CSV (default: results/)")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save detections to a CSV file")
    parser.add_argument("--show", action="store_true",
                        help="Display annotated frames in a window (requires display)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-advance frames (no keypress needed, use with --show)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
