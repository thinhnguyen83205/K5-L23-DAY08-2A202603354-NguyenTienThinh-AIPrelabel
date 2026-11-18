# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 434; đối tượng sau duyệt: 533; ngưỡng accepted: IoU >= 0.9
- Đã **loại 22 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 393 | 393/434 = 90.6% | 434 box máy |
| edited (sửa) | 15 | 15/434 = 3.5% | 434 box máy |
| deleted (xoá) | 17 | 17/434 = 3.9% | 434 box máy |
| added (vẽ thêm) | 123 | 123/533 = 23.1% | 533 đối tượng sau duyệt |

Trong số deleted + added, **4 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 119.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 237 | 5 | 4 | 40 |
| bicycle | 7 | 0 | 2 | 4 |
| motorbike | 88 | 3 | 0 | 42 |
| car | 44 | 0 | 3 | 24 |
| bus | 2 | 0 | 1 | 1 |
| truck | 4 | 0 | 1 | 5 |
| three_wheeler | 11 | 7 | 6 | 7 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 75 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 47 | 11 | 28 | 81.0% (KTC 95%: 69–89%) | 62.7% (KTC 95%: 51–73%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 5/12 = 41.7% (KTC 95%: 19–68%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 2/3
  - missing: 1/3
  - loose: 2/3
  - extra: 0/3
  - lọt: img_022.jpg (wrong_class), img_008.jpg (loose), img_003.jpg (extra), img_018.jpg (extra), img_035.jpg (missing), img_025.jpg (missing), img_033.jpg (extra)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
