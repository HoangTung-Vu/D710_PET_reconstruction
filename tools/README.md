# tools

Công cụ chạy tay, **không** nằm trên đường `d710 exam → osem → export`. Không có
gì ở đây được pipeline import; xoá một file ở đây không làm hỏng tái tạo.

## Đang dùng

| file | dùng khi nào |
|---|---|
| `compare_vendor.py` | so ảnh của ta với bản BQML của GE trên cùng ca; `--json` ghi `calib_<đường>.json` |
| `calib_k.py` | gộp các `calib_*.json` thành **hai** hằng số `K` để dán vào `utils/scanner.py` |
| `dicom_suv.py` | thư mục PET DICOM bất kỳ → NIfTI SUV |
| `compare_suv.py` | so hai khối SUV |
| `ct_nifti.py` | CT DICOM → NIfTI HU, trên đúng affine của `export` |
| `migrate_out.sh` | chuyển cây đầu ra cũ sang `$D710_OUT` (dry-run mặc định) |

`../run_all_ok.sh`, `../export_all_ok.sh`, `../rerun_lm_ok.sh` gọi bốn cái đầu.

## Chẩn đoán — giữ lại vì có chỗ trỏ tới

| file | vì sao còn |
|---|---|
| `tof_direction.py` | đo chiều trục TOF; `utils/terms.py` nhắc tên nó trong thông báo lỗi |
| `tof_profile.py` | đo profile TOF của scatter; `utils/terms.py` in đúng lệnh này khi cần |
| `lm_frame.py` | đo hệ quy chiếu list-mode; `tests/test_lm_geom.py` dẫn nó làm nguồn của khung đã chốt. **Mù với sai số góc dùng chung** — 10,04° từng lọt qua nó |

## Phép đo một lần, giữ làm bằng chứng

Không ai gọi, và **cố ý giữ**: chúng là số liệu đằng sau hai quyết định kiến
trúc, chứ không phải mã chết cần dọn.

| file | đã chứng minh điều gì |
|---|---|
| `projector_bench.py` | chi phí projector — vì sao subset không mua được gì với parallelproj |
| `pytomo_lm_probe.py` | PyTomography chạy được list-mode D710 → cơ sở để chọn nó |

⚠ `pytomo_lm_probe.py` dựng hình học **riêng** và đã lệch khỏi `lm/geom.py`:
`(cos, sin)` thay vì `(sin, −cos)`, `R_MM` trần thay vì `R_EFF_MM`, và không đảo
trục TOF. Docstring của nó nói lấy hằng số từ `utils/scanner.py` — chỉ đúng với
*hằng số*, không đúng với hình học. Đừng đọc nó như bản tham chiếu.
