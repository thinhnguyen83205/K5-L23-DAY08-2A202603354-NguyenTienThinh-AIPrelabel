# Báo cáo Day 08 — <<MSHV - Họ và tên>>

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | `bicycle=0.1`; các lớp còn lại dùng `--conf 0.3` | `threshold_sweep.md` trên gold 4 ảnh: bicycle đạt recall 100%, precision 75% (6 vật gold); person/motorbike/car không đạt recall 90% ở ngưỡng ≤ 0.5; bus (1 vật) và truck (0 vật) quá ít để chọn ngưỡng riêng. |
| three_wheeler | YOLOE-26s, prompt theo `taxonomy.json` | `three_wheeler=0.1` | `threshold_sweep.md`: recall 100%, precision 78% (7 vật gold). |

**Nhận xét pilot (chưa phải kết luận chọn recipe):** Trên cùng 8 ảnh, YOLO26m tạo 215 box ở conf 0.25 (r01) và 139 box ở conf 0.5 (r02), giảm 76/215 = 35.3%. Ngưỡng cao hơn làm giảm số box, nhưng chưa chứng minh box nào chính xác hơn hay vật nào bị bỏ sót; cần đối chiếu ảnh `viz/` trước khi chọn. YOLOE tạo 8 box `three_wheeler` trên 8 ảnh (r03); cần kiểm tra từng box với ảnh gốc. Các detection thuộc lớp ngoài ánh xạ đã bị loại, xem cột `dropped` trong `model_log.csv`. Chạy YOLO26 + SAM 2.1 trên 2 ảnh tạo 16 box có mask, tốc độ 4.48 giây/ảnh (r04); số mask chưa phản ánh mask có đúng/ôm sát vật hay không.

- Chuỗi `--thr` được sweep gợi ý: `--thr "bicycle=0.1,three_wheeler=0.1"`; các lớp khác dùng `--conf 0.3`. Sweep chọn và đánh giá trên cùng gold 4 ảnh nên số đo lạc quan, chỉ là ước lượng thô.
- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): <<>>
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: <<chất lượng mask, tốc độ, RAM, quyền truy cập>>
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? <<>>
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): <<>>
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: <<>>

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 393 | 90.6% | 434 box máy |
| edited | 15 | 3.5% | 434 box máy |
| deleted | 17 | 3.9% | 434 box máy |
| added | 123 | 23.1% | 533 đối tượng sau duyệt |

(Đã loại 22 đối tượng quanh vùng bẫy khỏi bảng vì đó là lỗi nhân tạo, không phải lỗi máy. 4 trường hợp trong deleted + added là đổi lớp, nên số vật máy sót thật ≈ 119.)

- Lớp máy tệ nhất: **three_wheeler**. Trong 24 box máy chỉ 11 box được giữ nguyên (11/24 ≈ 46%), 7 sửa, 6 xoá, thêm 7 vật bị sót. Box YOLOE lỏng và có box nhầm, ví dụ <<img_0xx: ...>>. Máy bỏ sót nhiều nhất ở motorbike (42 added / 133 sau duyệt) và person (40 added), chủ yếu là vật nhỏ, ở xa hoặc bị che <<ví dụ img_0xx>>.
- So với gold (75 đối tượng): precision của máy 47/58 = 81.0% (KTC 95%: 69–89%), recall 47/75 = 62.7% (KTC 95%: 51–73%). Recall thấp khớp với `added` cao. Con số này lạc quan vì ngưỡng được chọn trên chính 4 ảnh gold này. Dùng YOLO26n (cỡ nhỏ, chạy CPU) cũng làm recall thấp hơn.
- accepted 90.6% **không** có nghĩa máy tốt: máy vẫn sót khoảng 1/4 số vật (added 23.1%, recall 62.7%), và accepted cao một phần do chính tôi duyệt lỏng (xem mục 3).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **5/12 = 41.7%** (KTC 95%: 19–68%), dưới ngưỡng 10/12.
  - extra 0/3: lọt cả ba (img_003, img_018, img_033)
  - missing 1/3: lọt img_025, img_035
  - loose 2/3: lọt img_008
  - wrong_class 2/3: lọt img_022
- Vì sao lọt: <<bạn tự viết trung thực. Gợi ý tự hỏi: có ẩn nhãn máy và đếm vật trước ở mọi ảnh không? Có xét từng box xem "có vật thật không" không, hay chỉ tìm vật thiếu? Lọt nhiều ở cuối lô do mệt? img_018 là ảnh dày xe máy, box ma lẫn trong đám đông?>>
- Không bắt được bẫy `extra` nào và chỉ xoá 17/434 box máy cho thấy tôi gần như bỏ qua bước 2 (Thừa) của checklist. Tôi tập trung tìm vật thiếu (added 123) nhưng mặc định coi box máy có sẵn là đúng. Accepted cao (90.6%) đi cùng bắt bẫy thấp (41.7%) nghĩa là accepted cao phần lớn do tôi bấm chấp nhận chứ không do máy đúng: đây là thiên kiến tự động hóa. Chỉ 12 bẫy nên KTC rất rộng (19–68%), nhưng cả cận trên 68% vẫn dưới 80%.

## 4. Máy có đáng dùng không?

- Thời gian (`time_log.csv`): gán tay <<X>> phút / 4 ảnh = <<X/4>> phút/ảnh; duyệt <<Y>> phút / 24 ảnh = <<Y/24>> phút/ảnh. Thiên lệch: tôi gán gold trước nên đã quen kiểu cảnh, và một phần tốc độ duyệt nhanh là do duyệt lỏng (bắt bẫy 41.7%), nên mức tiết kiệm thật nhỏ hơn con số này.
- Kết luận: <<có / có điều kiện>> cho person, motorbike, car (precision trên gold 81%, accepted cao) **với điều kiện** người duyệt bắt buộc kiểm box thừa. Với three_wheeler thì máy chỉ hỗ trợ một phần: 11/24 box giữ nguyên, nên vẫn phải soát kỹ từng box.
- **Ba quy tắc giảm thiên kiến:**
  1. Ẩn nhãn máy và ghi số vật tự đếm cho 100% ảnh **trước** khi hiện nhãn (kiểm bằng cột đếm trong log duyệt).
  2. Mỗi ảnh phải tự đặt câu hỏi "box này có vật thật không?" cho từng box máy, và với ảnh có từ 10 box trở lên phải xoá hoặc gắn `can_review` ít nhất các box conf < 0.4 sau khi xem (kiểm bằng tỉ lệ deleted khác 0 theo từng ảnh trong review_stats).
  3. Nghỉ 5 phút sau mỗi 8 ảnh, và đạt ≥ 10/12 bẫy ở vòng duyệt sau (kiểm bằng `time_log.csv` có dòng `break` và mục 3 của `review_stats.md`).

| biến thể | #ảnh train | số seed | map50 (TB ± độ lệch) | map50_95 (TB ± độ lệch) | precision (TB ± độ lệch) | recall (TB ± độ lệch) |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.726 ± 0.000 | 0.512 ± 0.004 | 0.764 ± 0.061 | 0.649 ± 0.028 |
| clean | 202 | 2 | 0.701 ± 0.025 | 0.487 ± 0.008 | 0.759 ± 0.003 | 0.625 ± 0.019 |

Chênh mAP50 (clean − raw) = **-0.025**.


## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 | SAM 2.1 tiny (sam2.1_t.pt) | <<2 dương + 1 âm>> | <<có/không: người lái bị loại / còn dính>> | <<...>> | tuktuk_img013.jpg |
| img_004 | SAM 2.1 tiny | <<2 + 1>> | <<...>> | <<có tràn sang tuk-tuk xanh lá bên cạnh không>> | tuktuk_img004.jpg |
| img_017 | SAM 2.1 tiny | <<2 + 2>> | <<...>> | <<bị người che: mask có dính người, mấy polygon>> | tuktuk_occluded_img017.jpg |

- Chọn SAM 2.1 tiny vì máy chỉ có CPU: chạy nhanh, đủ chất lượng; SAM 3 cần GPU và quyền truy cập.
- Đối chiếu guideline 4.2: mục 4.2.4 ("hai vật chạm nhau: mỗi vật một polygon, vùng chạm thuộc vật phía trước") giúp quyết định khi người đứng trước xe ở img_017. Guideline chưa nói rõ <<mui xe / kính chắn gió có thuộc mask không (câu hỏi mở 5)>>. Một click không đủ: mask có thể là bộ phận hoặc cả xe, nên cần thêm click âm để loại người lái theo đúng label contract.



## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
