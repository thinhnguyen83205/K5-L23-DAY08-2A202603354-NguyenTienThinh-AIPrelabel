# Báo cáo Day 08 — 2A202603354 - Nguyễn Tiến Thịnh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | `bicycle=0.1`; các lớp còn lại dùng `--conf 0.3` | `threshold_sweep.md` trên gold 4 ảnh: bicycle đạt recall 100%, precision 75% (6 vật gold); person/motorbike/car không đạt recall 90% ở ngưỡng ≤ 0.5; bus (1 vật) và truck (0 vật) quá ít để chọn ngưỡng riêng. |
| three_wheeler | YOLOE-26s, prompt theo `taxonomy.json` | `three_wheeler=0.1` | `threshold_sweep.md`: recall 100%, precision 78% (7 vật gold). |

**Nhận xét pilot (chưa phải kết luận chọn recipe):** Trên cùng 8 ảnh, YOLO26m tạo 215 box ở conf 0.25 (r01) và 139 box ở conf 0.5 (r02), giảm 76/215 = 35.3%. Ngưỡng cao hơn làm giảm số box, nhưng chưa chứng minh box nào chính xác hơn hay vật nào bị bỏ sót; cần đối chiếu ảnh `viz/` trước khi chọn. YOLOE tạo 8 box `three_wheeler` trên 8 ảnh (r03); cần kiểm tra từng box với ảnh gốc. Các detection thuộc lớp ngoài ánh xạ đã bị loại, xem cột `dropped` trong `model_log.csv`. Chạy YOLO26 + SAM 2.1 trên 2 ảnh tạo 16 box có mask, tốc độ 4.48 giây/ảnh (r04); số mask chưa phản ánh mask có đúng/ôm sát vật hay không.

- Chuỗi `--thr` được sweep gợi ý: `--thr "car=0.45,three_wheeler=0.15"`; các lớp khác dùng `--conf 0.3`. Sweep chọn và đánh giá trên cùng gold 4 ảnh nên số đo lạc quan, chỉ là ước lượng thô.
- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts\merge_prelabels.py work\prelabel\yoloe_all\pre.coco.json work\prelabel\y26_all\pre.coco.json --thr "car=0.45,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work\prelabel\merged_review\pre.coco.json`
- **Phiên bản SAM:** SAM 2.1 tiny (`sam2.1_t.pt`). Lý do: máy không có GPU; bản tiny chạy được trên CPU (khoảng 2 s/ảnh khi box → mask) và đủ RAM; SAM 3 cần GPU và quyền truy cập trên HuggingFace.
- **Lỗi không sửa được bằng đổi ngưỡng:** YOLO26 không có lớp `three_wheeler` nên gọi tuk-tuk / ba gác là `car` hoặc `truck` (img_013, img_004); ở conf 0.5 lỗi vẫn còn. Đây là lỗi sai lớp, phải sửa bằng model khác: ghép YOLOE (`three_wheeler`) đặt trước YOLO26 trong `merge_prelabels.py`, box `three_wheeler` thắng box `car`/`truck` khi IoU ≥ 0.6. Recall thấp của person / motorbike cũng không sửa được bằng ngưỡng (hạ tới 0.05 vẫn tối đa 56% / 47%): cần model lớn hơn (YOLO26m) hoặc vẽ tay.
- **Điểm cao mà sai:** img_003 box `car` 0.76 nằm trên mặt đường / thùng gỗ xe ba gác, không có xe (tìm thấy khi QC); img_014 box `three_wheeler` 0.63 trên gánh hàng rong.
- **Giới hạn:** ngưỡng được chọn và chấm trên cùng 4 ảnh gold nên precision / recall của máy trên gold là lạc quan; lớp ít vật (bus 1, truck 0, bicycle 6) thì ngưỡng chỉ là ước lượng thô.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 393 | 90.6% | 434 box máy |
| edited | 15 | 3.5% | 434 box máy |
| deleted | 17 | 3.9% | 434 box máy |
| added | 123 | 23.1% | 533 đối tượng sau duyệt |

(Đã loại 22 đối tượng quanh vùng bẫy khỏi bảng vì đó là lỗi nhân tạo, không phải lỗi máy. 4 trường hợp trong deleted + added là đổi lớp, nên số vật máy sót thật ≈ 119.)

- Lớp máy tệ nhất: **three_wheeler**. Trong 24 box máy chỉ 11 box được giữ nguyên (11/24 ≈ 46%), 7 sửa, 6 xoá, thêm 7 vật bị sót. Box YOLOE lỏng và có box nhầm. Máy bỏ sót nhiều nhất ở motorbike (42 added / 133 sau duyệt) và person (40 added), chủ yếu là vật nhỏ, ở xa hoặc bị che.
- So với gold (75 đối tượng): precision của máy 47/58 = 81.0% (KTC 95%: 69–89%), recall 47/75 = 62.7% (KTC 95%: 51–73%). Recall thấp khớp với `added` cao. Con số này lạc quan vì ngưỡng được chọn trên chính 4 ảnh gold này. Dùng YOLO26n (cỡ nhỏ, chạy CPU) cũng làm recall thấp hơn.
- accepted 90.6% **không** có nghĩa máy tốt: máy vẫn sót khoảng 1/4 số vật (added 23.1%, recall 62.7%), và accepted cao một phần do chính tôi duyệt lỏng (xem mục 3).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **5/12 = 41.7%** (KTC 95%: 19–68%), dưới ngưỡng 10/12.
  - extra 0/3: lọt cả ba (img_003, img_018, img_033)
  - missing 1/3: lọt img_025, img_035
  - loose 2/3: lọt img_008
  - wrong_class 2/3: lọt img_022
- Vì sao lọt: bạn tự viết trung thực. Gợi ý tự hỏi: có ẩn nhãn máy và đếm vật trước ở mọi ảnh không? Có xét từng box xem "có vật thật không" không, hay chỉ tìm vật thiếu? Lọt nhiều ở cuối lô do mệt? img_018 là ảnh dày xe máy, box ma lẫn trong đám đông?
- Không bắt được bẫy `extra` nào và chỉ xoá 17/434 box máy cho thấy tôi gần như bỏ qua bước 2 (Thừa) của checklist. Tôi tập trung tìm vật thiếu (added 123) nhưng mặc định coi box máy có sẵn là đúng. Accepted cao (90.6%) đi cùng bắt bẫy thấp (41.7%) nghĩa là accepted cao phần lớn do tôi bấm chấp nhận chứ không do máy đúng: đây là thiên kiến tự động hóa. Chỉ 12 bẫy nên KTC rất rộng (19–68%), nhưng cả cận trên 68% vẫn dưới 80%.

## 4. Máy có đáng dùng không?

- Thời gian (`time_log.csv`): gán tay <<X>> phút / 4 ảnh = <<X/4>> phút/ảnh; duyệt <<Y>> phút / 24 ảnh = <<Y/24>> phút/ảnh. Thiên lệch: tôi gán gold trước nên đã quen kiểu cảnh, và một phần tốc độ duyệt nhanh là do duyệt lỏng (bắt bẫy 41.7%), nên mức tiết kiệm thật nhỏ hơn con số này.
- Kết luận: <<có>> cho person, motorbike, car (precision trên gold 81%, accepted cao) **với điều kiện** người duyệt bắt buộc kiểm box thừa. Với three_wheeler thì máy chỉ hỗ trợ một phần: 11/24 box giữ nguyên, nên vẫn phải soát kỹ từng box.
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
| img_013 | SAM 2.1 tiny (sam2.1_t.pt) | <<2 dương + 1 âm>> | <<có/không: người lái bị loại / còn dính>> | <<ok>> | tuktuk_img013.jpg |
| img_004 | SAM 2.1 tiny | <<2 + 1>> | <<2 + 1>> | <<có tràn sang tuk-tuk xanh lá bên cạnh không>> | tuktuk_img004.jpg |
| img_017 | SAM 2.1 tiny | <<2 + 2>> | <<2 + 2>> | <<bị người che: mask có dính người, mấy polygon>> | tuktuk_occluded_img017.jpg |

- Chọn SAM 2.1 tiny vì máy chỉ có CPU: chạy nhanh, đủ chất lượng; SAM 3 cần GPU và quyền truy cập.
- Đối chiếu guideline 4.2: mục 4.2.4 ("hai vật chạm nhau: mỗi vật một polygon, vùng chạm thuộc vật phía trước") giúp quyết định khi người đứng trước xe ở img_017. Guideline chưa nói rõ <<mui xe / kính chắn gió có thuộc mask không (câu hỏi mở 5)>>. Một click không đủ: mask có thể là bộ phận hoặc cả xe, nên cần thêm click âm để loại người lái theo đúng label contract.



## 6. Calibration (Part 6)

- Số quyết định: 4 (`qc/calibration_decisions.csv`): 1 `loi_may`, 2 `cau_hoi_mo`, 1 `bay_lot`.
- Quan trọng nhất: **CAL-03** (bẫy lọt). Cả 3/3 bẫy `extra` lọt (img_003, img_018, img_033) và tôi chỉ xoá 17/434 box máy. QC ở mục 7 xác nhận đây là lỗi lặp: 3/7 lỗi QC là `thua` (img_003 box `car` 0.76, img_014 hai box `three_wheeler`). Đây là nguyên nhân chính dẫn tới quyết định Rework.
- Quy tắc mới đã vào `guidelines/guideline_v1.1_calibrated.md`:
  - **CAL-02** (câu hỏi mở 2): vật cao < 20 px không vẽ từng cái, bất kể gần hay xa camera; cụm ≥ 5 vật như vậy gom một `ignore_region`; vật ≥ 20 px gán từng cái dù đông (ảnh ví dụ img_018, img_024: `validate_labels` cảnh báo 16 box < 20 px).
  - **CAL-04** (câu hỏi mở 5): kính chắn gió và khung mui thuộc mask xe; người nhìn thấy bên trong là `person` riêng, bỏ khỏi mask xe (ảnh ví dụ img_013, img_017 ở Part 5).
  - **CAL-01**: ghi chú review: box `car`/`truck` có mui hoặc 3 bánh phải kiểm lại lớp (img_013, img_004).
  - **CAL-03**: ghi chú quy trình review: với mỗi box máy phải hỏi "có vật thật không" trước khi giữ.
- Câu hỏi mở chưa chốt: 1 (xe đạp điện), 3 (hàng nhô ra), 4 (ngưỡng 20% người trong xe có mui), 6 (box chồng ảnh đêm), 7 (conf tối thiểu). Lý do: lô ảnh không có đủ ca ví dụ để quyết; tiếp tục dùng `can_review` khi gặp.
- Nguyên nhân gốc, tách theo loại:
  - **Guideline mơ hồ** (phải sửa chữ trong v1.1): CAL-02, CAL-04.
  - **Máy sai** (guideline đã rõ, không đổi chữ): CAL-01. YOLO26 không có lớp `three_wheeler` nên gọi tuk-tuk là `car`/`truck`; lỗi sai lớp này không sửa được bằng đổi ngưỡng, phải ghép YOLOE.
  - **Tôi sai khi duyệt** (không đổi quy tắc gán nhãn, chỉ đổi quy trình): CAL-03. Bỏ bước 2 (Thừa) của checklist.

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` + 5 ảnh ngẫu nhiên, seed 42): img_003, img_006, img_014, img_031, img_035; 110 đối tượng. Mẫu không có ảnh `can_review` nào vì tôi không tick `can_review` box nào khi duyệt lô batch. Người kiểm: chính tôi (**hạn chế**: tôi vừa là người duyệt, dễ bỏ sót cùng kiểu lỗi đã bỏ sót lúc review).
- Lỗi theo **đối tượng**: 7/110 = 6.4% · theo **ảnh**: 4/5 = 80.0% (ngưỡng Release ≤ 2% theo ảnh).
- Top lỗi:
  - `thua` 3 lỗi: img_003 (box `car` 0.76 trên mặt đường / thùng gỗ xe ba gác), img_014 (2 box `three_wheeler` 0.33 và 0.63 trên gánh hàng rong).
  - `sai_vi_tri_kich_thuoc` 2 lỗi: img_031 (box `person` 0.79 và `bicycle` 0.45 không phủ hết vật).
  - `bo_sot` 2 lỗi: img_035 (xe máy trên vỉa hè và cụm xe phía xa chưa có box hoặc `ignore_region`).
  - Lỗi `thua` đứng đầu (3/7), khớp với Part 4: tôi để lọt cả 3/3 bẫy `extra` và chỉ xoá 17/434 box máy. QC xác nhận bước "Thừa" của checklist bị bỏ qua có hệ thống. Box `car` 0.76 ở img_003 cũng cho thấy điểm cao không có nghĩa là đúng.
- **Giới hạn mẫu nhỏ:** 4 ảnh lỗi trong 5 ảnh mẫu, cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là 96%, cận dưới khoảng 38%. Mẫu 5 ảnh không đủ để chứng minh lô đạt ≤ 2%, kể cả khi 0 lỗi (0/5 thì cận trên vẫn khoảng 43%). Nhưng ở đây ngay cả cận dưới (~38%) cũng cao hơn 2% rất nhiều, nên kết luận "không đạt" là chắc chắn dù mẫu nhỏ.
- Script kiểm được: tên file, lớp có trong taxonomy, box trong ảnh, đủ ảnh, accepted/edited/deleted/added, tỉ lệ bắt bẫy, tỉ lệ lỗi theo phiếu. Script **không** kiểm được: box có ôm sát vật thật không, lớp có đúng thực tế không, vật cả tôi lẫn máy cùng bỏ sót (ngoài gold), tôi có thật sự theo checklist không. Phiếu QC cũng chỉ đúng bằng mắt người kiểm.

**Quyết định: Rework**, ngày 2026-11-18. Lý do (đi theo Hold → Rework → Release):
1. Không Hold: các ca gặp trong lô đã có quy tắc (câu hỏi mở 2 và 5 đã chốt trong v1.1; các câu còn lại không gặp ca nào trong lô), và không có ảnh hỏng.
2. Rework vì: ảnh có lỗi 4/5 = 80.0% > 2%; bắt bẫy 5/12 = 41.7% < 80% (lô duyệt chưa đáng tin); lỗi `thua` lặp lại ở 2/5 ảnh mẫu và trùng với kiểu bẫy lọt.

Việc cần làm tiếp:
- Duyệt lại cả 24 ảnh, lần này với mỗi box máy phải hỏi "có vật thật không" (bước 2, Thừa), áp dụng 3 quy tắc ở mục 4.
- Sửa các lỗi QC đã thấy (xoá box #18 img_003, #169 #170 img_014; kéo lại #509 #510 img_031; vẽ thêm vật sót và `ignore_region` img_035), export `submission/labels/qc_fixed.json`.
- Chạy lại `build_release.py` và `qc_sample.py --force`, ghi kết quả vòng 2 vào báo cáo.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không: tôi gán tay 4 ảnh gold trước khi chạy bất kỳ model nào lên ảnh gold và không xem pre-label. Trong lúc gán tôi có hỏi AI để tóm tắt quy tắc của guideline v1.0 (bảng lớp, quy tắc 20 px, người ngồi trên xe), nhưng không đưa ảnh gold cho AI và AI không gán thay. >>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Windows, không có GPU (torch 2.15.0+cpu), Python 3.12.6 trong venv; CVAT chạy bằng Docker trên máy, phiên bản <<v2.xx>>
- Dùng cache `data/cache/` thay vì tự chạy? <<Không: tự chạy YOLO26 và YOLOE trên cả 36 ảnh (r08, r09)>>
- Lỗi / bug script đã gặp (nếu có): <<`work/runs.csv` đánh trùng mã `r03` cho hai lượt chạy khác nhau (lượt thử YOLOE smoke test và lượt `t_y26`); trong `model_log.csv` tôi dùng `r03` cho `t_y26`.
  `validate_labels.py` cảnh báo 16 box cao < 20 px trong `batch_reviewed.json` (chủ yếu img_018, img_024). Đây là cảnh báo, không phải lỗi, nhưng là nguồn cho CAL-02.>>
- Một điều bạn làm khác lần sau: <<Ở mỗi ảnh, sau khi tự đếm vật, tôi sẽ duyệt **từng box máy** để hỏi "box này có vật thật không" (bước Thừa) thay vì chỉ tìm vật thiếu, vì tôi để lọt 3/3 bẫy `extra` và QC tìm ra 3/7 lỗi là box thừa. Ngoài ra sẽ tải YOLO26m (chỉ ~2 s/ảnh trên CPU) để tăng recall `person` / `motorbike`, vì YOLO26n chỉ đạt recall tối đa 56% / 47% trên gold>>
