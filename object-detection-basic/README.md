| Đối số         | Loại      | Vai trò chính                                    | Khi nào cần chỉnh                       | Gợi ý dùng thực tế                          |
| -------------- | --------- | ------------------------------------------------ | --------------------------------------- | ------------------------------------------- |
| **source**     | str       | Chỉ định nguồn dữ liệu (ảnh/video/webcam/folder) | Khi đổi input test hoặc deploy realtime | `"video.mp4"`, `0` (webcam), `"folder/"`    |
| **conf**       | float     | Ngưỡng tin cậy để giữ detect                     | Khi bị detect sai hoặc bỏ sót object    | `0.3 – 0.5` (cân bằng tốt)                  |
| **iou**        | float     | Lọc box trùng (NMS)                              | Khi có nhiều box chồng lên nhau         | `0.5 – 0.7`                                 |
| **imgsz**      | int/tuple | Kích thước ảnh đầu vào                           | Khi cần trade-off tốc độ vs chính xác   | `640` (chuẩn), `960–1280` nếu cần detect xa |
| **device**     | str       | Chọn CPU/GPU để chạy model                       | Khi deploy trên máy khác                | `0` (GPU), `"cpu"`                          |
| **classes**    | list[int] | Lọc class cần detect                             | Khi chỉ quan tâm object cụ thể          | Ví dụ: `[0]` (person), `[2]` (car)          |
| **max_det**    | int       | Giới hạn số object/frame                         | Scene đông object gây lag               | `100–300`                                   |
| **vid_stride** | int       | Bỏ qua frame video để tăng tốc                   | Realtime system bị chậm                 | `2–4` nếu cần tăng FPS                      |
| **stream**     | bool      | Xử lý video dạng generator (tiết kiệm RAM)       | Video dài hoặc stream realtime          | `True` cho production                       |
| **half**       | bool      | FP16 inference (tăng tốc GPU)                    | Khi chạy GPU hỗ trợ FP16                | `True`                                      |
| **batch**      | int       | Batch size khi xử lý nhiều ảnh/video             | Khi infer folder dataset                | `4–16` tùy GPU                              |
| **show**       | bool      | Hiển thị cửa sổ detect                           | Khi debug nhanh                         | `True` lúc test                             |
| **save**       | bool      | Lưu ảnh/video kết quả                            | Khi cần lưu output                      | `True` cho report/demo                      |
| **save_txt**   | bool      | Xuất file label                                  | Khi build dataset hoặc analytics        | `True` nếu cần annotation                   |
| **save_crop**  | bool      | Lưu ảnh crop object                              | Khi tạo dataset con                     | `True` nếu cần data augmentation            |
