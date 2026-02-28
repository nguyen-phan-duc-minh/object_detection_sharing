# Object Detection and Tracking - Advanced

Phiên bản nâng cao của hệ thống phát hiện và theo dõi đối tượng với YOLO.

## Cấu trúc dự án

```
object-detection-tracking-advanced/
├── config/
│   ├── logging.yaml          # Cấu hình logging
│   └── settings.yaml         # Cấu hình chính
├── core/
│   ├── logging_setup.py      # Thiết lập logging
│   └── settings_loader.py    # Load cấu hình
├── src/
│   ├── detector/
│   │   └── detector.py       # YOLODetector với tracking
│   ├── saver/
│   │   └── saver.py          # Lưu kết quả
│   ├── counter/
│   │   ├── line_counter.py   # Đếm qua line
│   │   └── zone_counter.py   # Đếm trong zone/lane
│   ├── visualize/
│   │   └── draw.py           # Vẽ bounding boxes
│   └── pipeline.py           # Pipeline chính
├── inputs/                   # Input videos/images
├── outputs/                  # Output results
├── models/                   # YOLO models
├── run.py                    # Script chạy chính
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu hình

### 1. Sử dụng YAML (khuyến nghị cho settings cơ bản)

Chỉnh sửa `config/settings.yaml`:

```yaml
detector:
  type: yolo26n
  conf_threshold: 0.5
  iou_threshold: 0.7
  classes: [2]  # chỉ detect xe
```

### 2. Sử dụng Environment Variables (khuyến nghị cho môi trường khác nhau)

Copy file `.env.sample` thành `.env` và chỉnh sửa:

```bash
cp .env.sample .env
```

File `.env` sẽ override các settings trong YAML:

```bash
# .env
DETECTOR_CONF_THRESHOLD=0.6
DETECTOR_CLASSES=0,2  # person và car
SAVER_SAVE_IMAGES=true
```

Environment variables có độ ưu tiên cao hơn YAML settings.

**Xem chi tiết**: [ENV_VARIABLES.md](ENV_VARIABLES.md)

## Sử dụng

### Chạy với video mặc định

```bash
python run.py
```

### Chạy với video tùy chỉnh

```bash
python run.py --input inputs/videos/your_video.mp4
```

### Chọn loại counter

```bash
# Line counter
python run.py --counter line

# Zone counter
python run.py --counter zone

# Lane counter (polygon)
python run.py --counter lane
```

### Lọc theo class

```bash
# Chỉ detect xe (class 2)
python run.py --classes 2

# Detect người và xe (class 0 và 2)
python run.py --classes 0 2
```

### Tùy chọn frame stride

```bash
# Xử lý mỗi 2 frame (tăng tốc độ)
python run.py --stride 2
```

### Không hiển thị cửa sổ

```bash
python run.py --no-display
```

### Ví dụ đầy đủ

```bash
python run.py \
    --input inputs/videos/traffic.mp4 \
    --counter lane \
    --classes 2 7 \
    --stride 1
```

## Cấu hình

Chỉnh sửa `config/settings.yaml` để thay đổi:

- **Detector**: confidence threshold, IOU threshold, classes
- **Saver**: lưu frames, text files, crops
- **Tracker**: loại tracker (bytetrack/strongsort)

## Các tính năng chính

### 1. YOLODetector
- Detect và track objects với YOLO
- Hỗ trợ ByteTrack và SORT tracker
- Gán track_id cho mỗi object

### 2. Counter
- **LineCounter**: Đếm objects qua một đường thẳng
- **ZoneCounter**: Đếm objects trong một vùng hình chữ nhật
- **LaneZoneCounter**: Đếm objects trong một vùng polygon (lane)

### 3. Saver
- Lưu frames
- Lưu detection text files
- Lưu crop từng object

### 4. Visualizer
- Vẽ bounding boxes
- Hiển thị track_id và confidence
- Vẽ counter regions

## YOLO Classes

Một số class IDs phổ biến:
- 0: person
- 2: car
- 3: motorcycle
- 5: bus
- 7: truck

## Lưu ý

- Model mặc định: `yolo26n.pt` (cần đặt trong thư mục `models/`)
- Video input: đặt trong thư mục `inputs/videos/`
- Kết quả: lưu trong thư mục `outputs/`
- Nhấn ESC để thoát khi đang chạy

## Tùy chỉnh Counter

Để thay đổi vị trí counter, chỉnh sửa trong `run.py`:

```python
# Line counter
counter_kwargs = {
    "start": (100, 200),
    "end": (500, 200)
}

# Zone counter
counter_kwargs = {
    "top_left": (100, 100),
    "bottom_right": (500, 300)
}

# Lane counter
counter_kwargs = {
    "points": [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
}
```
