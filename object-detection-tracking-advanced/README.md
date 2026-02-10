# Object Detection Pipeline - Production Ready

Hệ thống Object Detection + Tracking + Counting chuẩn production với YOLOv8 và ByteTrack.

## ✨ Tính năng

- ✅ **Object Detection**: YOLOv8n (tối ưu cho CPU)
- ✅ **Object Tracking**: ByteTrack tracking
- ✅ **Line Counting**: Đếm đối tượng qua đường kẻ
- ✅ **Zone Counting**: Đếm đối tượng vào/ra zone
- ✅ **Real-time Display**: Hiển thị trực tiếp kết quả
- ✅ **JSON Reports**: Báo cáo chi tiết JSON
- ✅ **FPS Optimization**: Tối ưu cho CPU
- ✅ **Production Logging**: Hệ thống logging đầy đủ
- ✅ **Flexible Configuration**: YAML config

## 📋 Yêu cầu hệ thống

- Python 3.8+
- CPU (hoặc GPU CUDA để tăng tốc)
- 4GB RAM
- macOS / Linux / Windows

## 🚀 Cài đặt

### 1. Clone/Download project

```bash
cd object-detection-courses
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 weights (tự động hoặc thủ công)

Model sẽ tự động download khi chạy lần đầu, hoặc download thủ công:

```bash
# Weights đã có sẵn tại weights/yolov8n.pt
# Nếu chưa có, sẽ tự động download
```

## 📖 Sử dụng

### Xử lý Video

```bash
# Cơ bản
python run.py --source inputs/videos/traffic.mp4

# Lưu output
python run.py --source inputs/videos/traffic.mp4 --output outputs/videos/result.mp4

# Không hiển thị realtime (chỉ xử lý)
python run.py --source video.mp4 --no-display

# Tùy chỉnh confidence threshold
python run.py --source video.mp4 --conf 0.5
```

### Xử lý Image

```bash
python run.py --source inputs/images/test.jpg --output outputs/images/result.jpg
```

### Webcam (Real-time)

```bash
python run.py --source 0
```

### Với custom config

```bash
python run.py --source video.mp4 --config configs/custom.yaml
```

## ⚙️ Cấu hình

Chỉnh sửa [configs/settings.yaml](configs/settings.yaml) để tùy chỉnh:

### Model Configuration
```yaml
model:
  weights_path: "weights/yolov8n.pt"
  confidence_threshold: 0.3
  iou_threshold: 0.5
  device: "cpu"  # Đổi thành "cuda" nếu có GPU
  classes: [0, 2]  # 0: person, 2: car
```

### Tracking Configuration
```yaml
tracking:
  tracker_type: "bytetrack"
  track_activation_threshold: 0.25
  lost_track_buffer: 30
```

### Line Counting
```yaml
counting:
  line_counting:
    enabled: true
    lines:
      - name: "line_1"
        start: [100, 300]  # (x, y)
        end: [500, 300]
        count_in: true
        count_out: true
```

### Zone Counting
```yaml
counting:
  zone_counting:
    enabled: true
    zones:
      - name: "zone_1"
        polygon: [[200, 100], [600, 100], [600, 400], [200, 400]]
        count_in: true
        count_out: true
```

### Performance Optimization
```yaml
performance:
  resize_frame: true
  target_width: 1280
  target_height: 720
  skip_frames: 0  # Skip mỗi N frames để tăng FPS
```

## 📁 Cấu trúc Project

```
object-detection-courses/
├── configs/
│   ├── logging.yaml          # Cấu hình logging
│   └── settings.yaml         # Cấu hình chính
├── core/
│   ├── logging_setup.py      # Setup logging
│   └── settings_loader.py    # Load config
├── src/
│   ├── detection/
│   │   └── detector.py       # YOLO detector
│   ├── tracking/
│   │   └── tracker.py        # ByteTrack tracker
│   ├── counting/
│   │   ├── direction.py      # Direction enum
│   │   ├── line_counter.py   # Line counting
│   │   └── zone_counter.py   # Zone counting
│   ├── visualization/
│   │   └── draw.py           # Visualization
│   └── pipeline.py           # Main pipeline
├── inputs/
│   ├── images/               # Input images
│   └── videos/               # Input videos
├── outputs/
│   ├── images/               # Output images
│   ├── videos/               # Output videos
│   └── logs/                 # Logs & reports
├── weights/
│   └── yolov8n.pt           # YOLOv8 weights
├── run.py                    # Entry point
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 📊 Output

### Video/Image Output
- Saved to `outputs/videos/` hoặc `outputs/images/`
- Annotated với bounding boxes, tracking IDs, counts

### JSON Reports
- Saved to `outputs/logs/report_*.json`
- Chứa:
  - Total frames processed
  - Detection statistics
  - Counting results (line & zone)
  - Processing performance

### Logs
- `outputs/logs/app.log` - General logs
- `outputs/logs/error.log` - Error logs

## 🎯 COCO Classes

Classes được detect (có thể customize trong config):
- **0**: person
- **2**: car

Full COCO classes: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

## 🔧 Troubleshooting

### Lỗi "Model weights not found"
```bash
# Download weights thủ công
# Weights sẽ tự động download khi chạy lần đầu
```

### FPS thấp trên CPU
```yaml
# Trong settings.yaml
performance:
  resize_frame: true
  target_width: 1280
  target_height: 720
  skip_frames: 1  # Skip mỗi frame thứ 2
```

### Out of Memory
```yaml
performance:
  resize_frame: true
  target_width: 640
  target_height: 480
```

## 🎨 Customization

### Thay đổi classes detect

```yaml
model:
  classes: [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
```

### Thêm line counter mới

```yaml
counting:
  line_counting:
    lines:
      - name: "entrance"
        start: [100, 200]
        end: [400, 200]
      - name: "exit"
        start: [100, 500]
        end: [400, 500]
```

### Thay đổi màu visualization

```yaml
visualization:
  colors:
    person: [0, 255, 0]    # Green (BGR)
    car: [255, 0, 0]       # Blue
    line: [0, 255, 255]    # Yellow
    zone: [255, 0, 255]    # Magenta
```

## 📈 Performance Tips

1. **CPU Optimization**:
   - Resize frames nhỏ hơn
   - Skip frames
   - Use YOLOv8n (smallest model)

2. **GPU Acceleration**:
   ```yaml
   model:
     device: "cuda"
     use_half_precision: true  # FP16
   ```

3. **Batch Processing**:
   - Disable real-time display: `--no-display`
   - Process offline

## 🤝 Support

Nếu gặp vấn đề:
1. Check logs tại `outputs/logs/`
2. Verify config trong `configs/settings.yaml`
3. Ensure dependencies installed: `pip install -r requirements.txt`

## 📝 License

MIT License

## 🙏 Credits

- **YOLOv8**: Ultralytics
- **ByteTrack**: ByteTrack paper
- **Supervision**: Roboflow Supervision library

---

**Enjoy tracking! 🚀**
