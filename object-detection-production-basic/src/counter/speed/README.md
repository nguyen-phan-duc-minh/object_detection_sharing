# Speed Estimation Module

## Tổng quan
Module này tính vận tốc của các đối tượng được tracking trong video, dựa trên:
- Tracking ID của object qua các frame
- Vị trí center point của bounding box
- FPS của video
- Tỷ lệ pixel-to-meter (cần calibration)

## Cách sử dụng

### 1. Bật Speed Estimation trong settings.yaml

```yaml
speed:
  enabled: true
  fps: 30
  pixel_per_meter: 21.7
  smooth_window: 5
```

### 2. Chạy pipeline

```bash
python3 run.py
```

Speed sẽ được hiển thị trên bounding box: `car - 1 - 0.85 - 45.2 km/h`

## Calibration - Quan trọng!

Để speed chính xác, bạn PHẢI calibrate `pixel_per_meter` cho video cụ thể.

### Phương pháp 1: Đo đối tượng thực tế

1. **Chọn một đối tượng có kích thước biết trước** trong video:
   - Chiều dài xe: 4-5 meters
   - Chiều rộng làn đường: 3-3.5 meters
   - Vạch kẻ đường: 3-6 meters

2. **Đo số pixels** trong video:
   - Pause video tại frame có object rõ ràng
   - Dùng tool measure (Paint, Photoshop, GIMP, v.v.)
   - Đếm số pixels tương ứng

3. **Tính pixel_per_meter**:
   ```
   pixel_per_meter = số_pixels / kích_thước_thực_tế_meters
   ```

**Ví dụ**:
- Chiều dài xe: 4.5 meters
- Đo trong video: 98 pixels
- `pixel_per_meter = 98 / 4.5 = 21.7`

### Phương pháp 2: Tính từ thông số camera/drone

Nếu biết thông số camera:

```
pixel_per_meter = (resolution_width * tan(FOV_horizontal/2)) / (2 * altitude)
```

**Ví dụ** (drone từ trên cao):
- Altitude: 50m
- FOV: 80°
- Resolution: 1920x1080
- `pixel_per_meter ≈ 19.2`

### Phương pháp 3: Thử nghiệm với speed đã biết

1. Tìm một object có speed đã biết (ví dụ: xe ô tô 60 km/h)
2. Chạy detection với `pixel_per_meter` mặc định
3. So sánh speed hiển thị vs speed thực tế
4. Điều chỉnh: `pixel_per_meter_new = pixel_per_meter_old * (speed_measured / speed_actual)`

## Các tham số config

### `fps` (Frame Per Second)
- FPS của video source
- Kiểm tra FPS thực tế: `ffprobe -v error -select_streams v -show_entries stream=r_frame_rate -of json video.mp4`
- Mặc định: 30

### `pixel_per_meter`
- Số pixel tương ứng 1 mét trong video
- **Phải calibrate cho từng video/camera**
- Mặc định: 21.7 (chỉ là ví dụ)

### `smooth_window`
- Số frame để làm mượt speed (giảm nhiễu)
- Giá trị nhỏ (3-5): Phản ứng nhanh nhưng nhiễu cao
- Giá trị lớn (10-15): Mượt mà nhưng chậm cập nhật
- Khuyến nghị: 5-7
- Mặc định: 5

## Lưu ý quan trọng

### 1. Perspective distortion
- Video từ góc nghiêng → object ở gần/xa có tỷ lệ pixel khác nhau
- Giải pháp: Dùng perspective transform hoặc calibrate cho vùng quan tâm

### 2. Tracking quality
- Speed chỉ chính xác khi tracking ổn định
- Nếu tracking bị mất/nhầm ID → speed sẽ sai
- Cải thiện: Tăng conf_threshold, giảm track loss

### 3. Video quality
- Video mờ, blur → detection không chính xác → speed sai
- Khuyến nghị: Video ít nhất 720p, 30 FPS

### 4. Speed calculation
Formula:
```python
distance_pixels = sqrt((x2-x1)^2 + (y2-y1)^2)
distance_meters = distance_pixels / pixel_per_meter
time_seconds = num_frames / fps
speed_m_s = distance_meters / time_seconds
speed_kmh = speed_m_s * 3.6
```

## Ví dụ thực tế

### Traffic monitoring (drone view, 50m altitude)
```yaml
speed:
  enabled: true
  fps: 30
  pixel_per_meter: 19.2  # Calibrated for 50m altitude
  smooth_window: 7
```

### Highway surveillance (side camera)
```yaml
speed:
  enabled: true
  fps: 25
  pixel_per_meter: 35.0  # Calibrated for 10m distance
  smooth_window: 5
```

### Indoor tracking (close range)
```yaml
speed:
  enabled: true
  fps: 30
  pixel_per_meter: 120.0  # High resolution, close distance
  smooth_window: 3
```

## Troubleshooting

### Speed quá cao/thấp
→ Kiểm tra lại `pixel_per_meter` và `fps`

### Speed nhảy lung tung
→ Tăng `smooth_window` (ví dụ: từ 5 lên 10)

### Speed luôn = 0
→ Kiểm tra tracking có hoạt động không (track_id có thay đổi qua frames không)

### Speed không hiển thị
→ Kiểm tra `enabled: true` trong settings.yaml
