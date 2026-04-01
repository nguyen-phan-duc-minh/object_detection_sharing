# Object Detection Sharing

Dự án Object Detection và Tracking sử dụng YOLO với 2 phiên bản: **Basic** và **Advanced**.

## Tổng quan

Project này cung cấp 2 phiên bản implementation cho bài toán phát hiện và theo dõi đối tượng:

1. **Basic**: Phiên bản đơn giản, dễ hiểu cho người mới bắt đầu
2. **Advanced**: Phiên bản nâng cao với kiến trúc module hóa, logging, configuration management

## Cấu trúc dự án

```
object_detection_sharing/
├── object-detection-basic/           # Phiên bản cơ bản
│   ├── detector.py                   # YOLO detector với tracking
│   ├── counter.py                    # Line/Zone/Lane counters
│   ├── visualize.py                  # Vẽ bounding boxes
│   ├── saver.py                      # Lưu kết quả
│   ├── main.py                       # Script chính
│   └── requirements.txt
│
├── object-detection-tracking-advanced/   # Phiên bản nâng cao
│   ├── config/                       # Configuration files
│   │   ├── settings.yaml             # Cấu hình chính
│   │   └── logging.yaml              # Cấu hình logging
│   ├── core/                         # Core utilities
│   │   ├── settings_loader.py        # Load settings
│   │   └── logging_setup.py          # Setup logging
│   ├── src/                          # Source code
│   │   ├── detector/                 # Detector module
│   │   ├── counter/                  # Counter modules
│   │   ├── saver/                    # Saver module
│   │   ├── visualize/                # Visualization module
│   │   └── pipeline.py               # Main pipeline
│   ├── run.py                        # Entry point
│   └── requirements.txt
│
└── README.md                         # File này
```

## Nhanh chóng bắt đầu

### Cài đặt dependencies

```bash
# Cho phiên bản Basic
cd object-detection-basic
pip install -r requirements.txt

# Cho phiên bản Advanced
cd object-detection-tracking-advanced
pip install -r requirements.txt
```

### Download YOLO model

```bash
# Tải model YOLOv8 (hoặc YOLO11)
# Đặt file .pt vào thư mục models/
```

### Chạy phiên bản Basic

```bash
cd object-detection-basic
python main.py
```

### Chạy phiên bản Advanced

```bash
cd object-detection-tracking-advanced
python run.py
```

## Phiên bản Basic

### Đặc điểm
- Code đơn giản, dễ hiểu
- Tất cả trong 1 file `main.py`
- Phù hợp cho học tập và demo nhanh
- Dễ dàng tùy chỉnh và thử nghiệm

### Tính năng
- YOLO detection với ByteTrack tracking
- 3 loại counters:
  - **LineCounter**: Đếm objects qua một đường thẳng
  - **ZoneCounter**: Đếm objects trong vùng hình chữ nhật
  - **LaneZoneCounter**: Đếm objects trong vùng polygon
- Visualization với bounding boxes và track IDs
- Lưu frames, text files, và crops

### Sử dụng

```python
from detector import YOLODetector
from counter import LineCounter, ZoneCounter, LaneZoneCounter
from visualize import draw_boxes
from saver import ResultSaver

# Khởi tạo
detector = YOLODetector("models/yolo26n.pt", classes=[2])
counter = LineCounter((100, 200), (500, 200))
saver = ResultSaver("outputs")

# Sử dụng
detections = detector.detect(frame)
counter.update(detections)
frame = draw_boxes(frame, detections, detector.model.names)
frame = counter.draw(frame)
```

### Xem thêm
[object-detection-basic/README.md](object-detection-basic/README.md)

## Phiên bản Advanced

### Đặc điểm
- Kiến trúc module hóa, dễ bảo trì
- Configuration management (YAML + Environment Variables)
- Logging system
- Command-line interface
- Phù hợp cho production

### Tính năng
- Tất cả tính năng của Basic
- Configuration qua YAML files
- Environment variables support (.env)
- Structured logging
- CLI với nhiều options
- Pipeline pattern cho xử lý
- Dễ dàng mở rộng và maintain

### Sử dụng

```bash
# Chạy với video mặc định và lane counter
python run.py

# Tùy chọn counter type
python run.py --counter line
python run.py --counter zone
python run.py --counter lane

# Lọc theo classes
python run.py --classes 0 2   # person và car

# Custom video
python run.py --input inputs/videos/traffic.mp4

# Tăng tốc với frame stride
python run.py --stride 2

# Không hiển thị window
python run.py --no-display

# Kết hợp nhiều options
python run.py --input inputs/videos/traffic.mp4 \
              --counter lane \
              --classes 2 7 \
              --stride 1
```

### Configuration

Có 2 cách để cấu hình:

**1. YAML files** - Chỉnh sửa `config/settings.yaml`:

```yaml
detector:
  type: yolo26n
  conf_threshold: 0.5
  iou_threshold: 0.7
  classes: [2]  # chỉ detect xe
  
saver:
  save_images: true
  save_detections: true
```

**2. Environment Variables** - Copy `.env.sample` thành `.env`:

```bash
cp .env.sample .env
```

Sau đó chỉnh sửa `.env`. Environment variables sẽ override YAML settings.

Xem chi tiết: [ENV_VARIABLES.md](object-detection-tracking-advanced/ENV_VARIABLES.md)


### Xem thêm
[object-detection-tracking-advanced/README.md](object-detection-tracking-advanced/README.md)

## So sánh 2 phiên bản

| Tính năng | Basic | Advanced |
|-----------|-------|----------|
| Độ phức tạp | Đơn giản | Vừa phải |
| Cấu trúc | Single file | Modular |
| Configuration | Hardcoded | YAML + .env |
| Environment Variables | Không | Có |
| Logging | Print | Structured logging |
| CLI | Không | Có (argparse) |
| Phù hợp cho | Học tập, demo | Production |
| Dễ tùy chỉnh | 5/5 | 4/5 |
| Maintainability | 3/5 | 5/5 |
| Scalability | 2/5 | 5/5 |

## Khi nào dùng phiên bản nào?

### Dùng Basic khi:
- Bạn đang học YOLO và object detection
- Cần demo nhanh một tính năng
- Muốn hiểu rõ từng bước
- Code đơn giản, dễ thay đổi

### Dùng Advanced khi:
- Triển khai production system
- Cần quản lý nhiều configurations
- Làm việc trong team
- Cần logging và debugging tốt
- Dự án lớn, cần mở rộng

## YOLO Classes

Một số class IDs thường dùng (COCO dataset):

| Class ID | Tên | Mô tả |
|----------|-----|-------|
| 0 | person | Người |
| 2 | car | Xe ô tô |
| 3 | motorcycle | Xe máy |
| 5 | bus | Xe buýt |
| 7 | truck | Xe tải |
| 1 | bicycle | Xe đạp |

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- PyYAML (cho Advanced)
- NumPy

## Tài liệu tham khảo

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [OpenCV Documentation](https://docs.opencv.org/)

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## License

Project này được chia sẻ với mục đích học tập và nghiên cứu.

## Tips

1. **Hiệu suất**: Dùng `--stride` để tăng tốc xử lý video
2. **GPU**: Model tự động dùng GPU nếu có (CUDA/MPS)
3. **Classes**: Lọc classes để giảm false positives
4. **Counter position**: Tùy chỉnh vị trí counter trong code để phù hợp với video
5. **Tracking**: ByteTrack hoạt động tốt hơn SORT trong hầu hết trường hợp

## Troubleshooting

### Lỗi "Model not found"
```bash
# Đảm bảo model file nằm đúng thư mục
ls models/yolo26n.pt
```

### Video không mở được
```bash
# Kiểm tra đường dẫn
ls inputs/videos/your_video.mp4
```

### Chạy chậm
```bash
# Tăng stride hoặc giảm resolution
python run.py --stride 2
```

### Out of memory
```bash
# Giảm max_det hoặc image resolution
# Sửa trong config/settings.yaml:
detector:
  max_det: 50
```

## Liên hệ

Nếu có câu hỏi hoặc issues, vui lòng tạo GitHub issue.

---

**Happy Coding!**
| Đối số                          | Loại      | Mặc định | Ý nghĩa chính                                                  |
| ------------------------------- | --------- | -------- | -------------------------------------------------------------- |
| `model`                         | str       | None     | Chọn model (.pt hoặc .yaml), quyết định kiến trúc & pretrained |
| `data`                          | str       | None     | File dataset (.yaml), chứa train/val/classes                   |
| `epochs`                        | int       | 100      | Số vòng lặp train                                              |
| `batch`                         | int/float | 16       | Batch size (ảnh hưởng VRAM & tốc độ)                           |
| `imgsz`                         | int       | 640      | Kích thước ảnh train                                           |
| `lr0`                           | float     | 0.01     | Learning rate ban đầu (rất quan trọng)                         |
| `lrf`                           | float     | 0.01     | Learning rate cuối                                             |
| `optimizer`                     | str       | auto     | Bộ tối ưu (SGD, AdamW,...)                                     |
| `momentum`                      | float     | 0.937    | Động lượng (SGD/Adam)                                          |
| `weight_decay`                  | float     | 0.0005   | Giảm overfitting                                               |
| `pretrained`                    | bool/str  | True     | Có dùng weight pretrained không                                |
| `freeze`                        | int/list  | None     | Đóng băng layer (transfer learning)                            |
| `device`                        | str/int   | None     | GPU/CPU sử dụng                                                |
| `workers`                       | int       | 8        | Số luồng load data                                             |
| `cache`                         | bool      | False    | Load data vào RAM để tăng tốc                                  |
| `amp`                           | bool      | True     | Mixed precision → nhanh hơn, ít VRAM                           |
| `cos_lr`                        | bool      | False    | Scheduler cosine (ổn định training)                            |
| `warmup_epochs`                 | float     | 3.0      | Warmup learning rate                                           |
| `patience`                      | int       | 100      | Early stopping                                                 |
| `augment (mosaic/close_mosaic)` | int       | 10       | Tắt mosaic ở cuối để ổn định                                   |
| `multi_scale`                   | float     | 0.0      | Train đa kích thước ảnh                                        |
| `box`                           | float     | 7.5      | Trọng số loss bbox                                             |
| `cls`                           | float     | 0.5      | Trọng số loss phân loại                                        |
| `dfl`                           | float     | 1.5      | Loss cho localization chi tiết                                 |
| `val`                           | bool      | True     | Có validate hay không                                          |
| `save`                          | bool      | True     | Lưu model                                                      |
| `resume`                        | bool      | False    | Train tiếp từ checkpoint                                       |
