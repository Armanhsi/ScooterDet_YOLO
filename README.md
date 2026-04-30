# E-Scooter Benchmark: Benchmarking the Deep Learning Techniques for Object Detection in E-scooters

## 1. Installation
- Create a conda environment: `conda create -n escooters python=3.12 -y`
- Active the virtual environment: `conda activate escooters`
- Install requirements: `pip install -r requirements.txt`

## 2. Preparing the Dataset
### 2.1 Dataset Preparation
- Download the data (https://zenodo.org/records/10578641).
- Run the script to convert the labeled data into YOLO format: `python yolov5/commons/labelme2yolo.py`

## 3. Training and Testing
- Download the pre-trained models from the official YOLO websites and unzip them to the corresponding folders. For example, you need to put the `yolov3.pt`, `yolov3-spp.pt` and `yolov3-tiny.pt` under the *YOLOV3/* folder.
- You can run the 0st data folder, we can run:`bash -i train.sh`.
- To test the models, we can run: `bash -i test.sh`.

## 4. Performance
The YOLO algorithms[1-6] used for our experiments are not maintained by us, please give credit to the authors of the YOLO algorithms[1-6].

# Video Demos
The video demos can be accessed at [[Demo]](https://drive.google.com/file/d/1YYxj8OWewmerNA7jAEGmmwdH6v-_TYXZ/view?usp=sharing)

# Citation
If you find the models and or the dataset useful, consider citing the following article:
```
Coming soon
```

# Reference
- [1-1] YOLOv3: Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
- [1-2] YOLOv3 Implementation: https://github.com/ultralytics/yolov3.
- [2-1] YOLOv4: Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "Yolov4: Optimal speed and accuracy of object detection." arXiv preprint arXiv:2004.10934 (2020).
- [2-2] YOLOv4 Implementation: https://github.com/WongKinYiu/PyTorch_YOLOv4.
- [3-1] YOLOv5: None
- [3-2] YOLOv5 Implementation: https://github.com/ultralytics/yolov5.
- [4-1] YOLOv6: Li, Chuyi, Lulu Li, Hongliang Jiang, Kaiheng Weng, Yifei Geng, Liang Li, Zaidan Ke et al. "YOLOv6: A single-stage object detection framework for industrial applications." arXiv preprint arXiv:2209.02976 (2022).
- [4-2] YOLOv6 Implementation: https://github.com/meituan/YOLOv6.
- [5-1] YOLOv7: Wang, Chien-Yao, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7464-7475. 2023.
- [5-2] YOLOv7 Implementation: https://github.com/WongKinYiu/yolov7
- [6-1] YOLOv8 Implementation: https://github.com/ultralytics/ultralytics

## Salem Additions:
- YOLO11n chosen: fastest/smallest YOLO11, ~20-30 FPS on Orin Nano GPU, well within real-time threshold
- Priority queue holds timestamped items from both sensors, pipeline always drains to the newest pair before inferring, minimizes latency while preserving ordering
- Fusion: weighted average, RS-preferred below 3m, LiDAR-preferred above 3m; degrades gracefully if one sensor is unavailable
- Dummy modes on both sensors let you develop/test the full pipeline on any machine without the hardware present
- Log files stamped with datetime: both .csv (tabular) and .jsonl (full fidelity) per run
- Could implement a rigid body shake after pothole detection to simulate a real-world scenario.

## SSH from laptop (downtown run)

Connect both the Jetson and your laptop to your mobile hotspot. Find the Jetson's IP on the hotspot admin page or by running `hostname -I` on the Jetson directly.

**1. SSH into the Jetson from your laptop (Windows terminal or PowerShell):**
```powershell
ssh username@<jetson-ip>
```

**2. Start the pipeline (headless, no display needed):**
```bash
cd ScooterDet_YOLO
python fusion_pipeline.py --lidar-port /dev/ttyUSB0 --no-display --log-dir logs/run1
```

**3. Stop recording when done (Ctrl+C in the SSH terminal).**

**4. Copy logs back to your Windows machine:**
```powershell
scp username@<jetson-ip>:~/ScooterDet_YOLO/logs/run1/* C:\Users\TheFl\CascadeProjects\windsurf-project\ScooterDet_YOLO\logs\
```

The three log files written per session are:
- `detections_*.csv` and `detections_*.jsonl` — full object detection + IMU per frame
- `imu_unity_*.csv` — Unity-ready pothole and surface change events only
    - To be used in Unity for pothole and surface change events in order to read the file and map the values to game physics.