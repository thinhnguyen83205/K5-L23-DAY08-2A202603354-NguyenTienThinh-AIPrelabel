# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A2026033554 - Nguyễn Tiến Thịnh

> Chỉ ghi phần thay đổi so với v1.0. Mỗi ca khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 2 | Khối xe máy đỗ dày gần camera: từng xe hay ignore_region? | Theo chiều cao box, không theo khoảng cách: vật < 20 px không vẽ từng cái, cụm ≥ 5 vật như vậy → một `ignore_region`; vật ≥ 20 px gán từng cái dù đông | img_018, img_024 | CAL-02 |
| 5 | Kính chắn gió / cửa sổ xe thuộc mask xe? | Có: kính và khung mui thuộc mask xe; người nhìn thấy bên trong là `person` riêng, bỏ khỏi mask xe | img_013, img_017 | CAL-04 |

Chưa chốt: câu 1 (xe đạp điện), 3 (hàng nhô ra), 4 (ngưỡng 20% người trong xe có mui), 6 (box chồng ảnh đêm), 7 (conf tối thiểu). Lý do: lô ảnh này không có đủ ca ví dụ để quyết; tiếp tục dùng `can_review`.

## 2. Ca khó mới (từ lỗi máy, bẫy lọt)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định | Quy tắc mới (mục nào của v1.0) |
|---|---|---|---|---|
| img_013, img_004 | Tuk-tuk có mui | YOLO26 gọi `car` / `truck` | `three_wheeler` | Mục 2 đã rõ; thêm ghi chú review: box `car`/`truck` có mui hoặc 3 bánh phải kiểm lớp (CAL-01) |
| img_003, img_018, img_033 | Box không có vật thật (bẫy extra lọt) | Box ma trong pre-label | Xoá | Quy trình review: với mỗi box máy hỏi "có vật thật không" trước khi giữ (CAL-03) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (`case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu hỏi mở 2 và 5; thêm ghi chú review cho tuk-tuk bị gọi car/truck và kiểm box thừa | CAL-01, CAL-02, CAL-03, CAL-04 |