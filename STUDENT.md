# Thông tin học viên

- Họ và tên: Nguyễn Tiến Thịnh
- MSHV: 2A202603354
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: thinhnguyen83205
- Repo bài nộp (URL): https://github.com/thinhnguyen83205/K5-L23-DAY08-2A2026001234-NguyenTienThinh-AIPrelabel
- Commit chốt bài (hash): điền sau khi commit cuối
- Máy: Windows, không có GPU (CPU only, torch 2.15.0+cpu), Python 3.12.6
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 393 / 15 / 17 (mẫu số: 434 box máy); added: 123 (mẫu số: 533 đối tượng)
- So với gold: precision / recall của **máy** 81.0% (47/58) / 62.7% (47/75) (mẫu số recall: 75 đối tượng gold)
- Bắt bẫy: 5/12 = 41.7%
- Thời gian: gán tay X phút/ảnh; duyệt có pre-label Y phút/ảnh
- Lỗi QC: 80.0% theo ảnh (4/5), 6.4% theo đối tượng (7/110)
- **Quyết định:** Rework — lý do một dòng: ảnh lỗi 4/5 = 80% > 2% và bắt bẫy 5/12 = 41.7% < 80%, chủ yếu do bỏ sót bước kiểm box thừa