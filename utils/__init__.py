"""Hạ tầng dùng chung — mọi thứ KHÔNG thuộc về một thuật toán tái tạo cụ thể.

| module | nội dung |
|---|---|
| `paths` | `$D710_OUT/<ca>/...`; chỗ duy nhất biết cây đầu ra |
| `attenuation` | CT DICOM -> mu-map -> hệ số suy giảm |
| `geometry` | quy ước chỉ số bin D710 -> STIR |
| `terms` | nạp các số hạng của một bed từ Interfile |
| `attn` | `af` theo bed, có cache trên đĩa (`work/bed<n>/attn.hs`) |
| `sirf_env` | chdir vào scratch + giữ `MessageRedirector` sống |
| `quant` | count/voxel -> Bq/mL -> SUV; hằng số `K` |
| `export` | ghi NIfTI / DICOM |
| `plots` | hình cho notebook |

Thuật toán ở nơi khác: `osem/` là một, thuật toán sau (FBP, MLEM, deep prior)
tạo package riêng cùng cấp và dùng lại đúng `utils/` này. Nên **không có gì
mang tính OSEM được đặt vào đây** — nếu một hàm chỉ có nghĩa với OSEM, chỗ của
nó là `osem/`.
"""
