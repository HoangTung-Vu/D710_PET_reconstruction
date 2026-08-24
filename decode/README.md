# decode — giải mã raw RDF của GE bằng chính mã của GE

Đọc raw của Discovery 710 ra sinogram Interfile + singles + PETSIRD list-mode.
Không tự cài đặt lại codec nào: cả hai codec của GE đều **chưa đảo được**, và
không cần đảo, vì binary của hãng chạy được ngoài console.

```bash
# một lần: image được bàn giao nguyên con, KHÔNG dựng lại
docker load -i d710_full.tar
export D710_OUT=~/UET/d710_out

d710 decode --raw <petRDFS/.../DIR> --case nema
d710 decode --raw <petRDFS/.../DIR> --lists <petLists/.../DIR> --listmode --case nema
```

Host chỉ cần **docker**. Không conda, không i386 multiarch, không cần checkout
`petsw/` — tất cả nằm trong image.

`decode_in.sh` ở đây là vòng lặp per-bed chạy **bên trong** container; `d710`
lo phần mount. Nó là file riêng chứ không bake vào image, nên sửa vòng lặp
không phải dựng lại 7 GB.

Ra ở `$D710_OUT/<ca>/decoded/`.

## Ra cái gì

| file | nội dung |
|---|---|
| `bed<n>.hs` / `.s` | sinogram prompt, 288 view × 553 plane × 381 u |
| `bed<n>.json` | header RDF đã parse (bed, table position, prompts, TOF, liều…) |
| `bed<n>.singles.npy` | singles theo tinh thể, 576 × 24 |
| `bed<n>.prd` | list-mode PETSIRD, chỉ với `--listmode` |
| `bed<n>.convert.log` | log giải mã, **chứa dòng MATCH** |

Bước tiếp theo là `d710 estimate` (randoms, scatter, norm, dead time) rồi
`d710 tostir`. `d710 exam` chạy cả ba.

## Hai đường giải mã, hai cơ chế khác nhau

Cùng một image, nhưng bên trong là hai cách chạy mã của GE khác hẳn nhau.

**Sinogram — `librdf.so.0`.** Codec entropy per-row (`RDF_RIVN_4BIT_V1`) chống
được ba đợt tấn công thống kê. Nhưng `/usr/PET/lib/linux2/librdf.so.0` là ELF
**32-bit x86 có đủ DWARF** và chạy nguyên xi. Python 64-bit không `dlopen` được
.so 32-bit — hai ABI không ở chung một process — nên `native/rdfx.c` là một
binary `-m32` nhỏ làm việc đó và ghi mảng thô ra file; `gerdf/vendor.py` chạy nó
rồi `memmap` kết quả.

**List-mode — `unglepl`.** Codec GLEPL nén `LIST*.BLF` cũng không đảo, cũng
không cần: `unglepl` của console chạy ở đây ~200 MB/s. Nó là binary i386 và cần
sáu thư viện **chưa bao giờ được copy khỏi console** — nhưng đóng bao symbol cho
thấy trong sáu cái đó chỉ đúng **một symbol** bị gọi thật: `getcfg`. Nên
`native/stub/` có năm stub rỗng chỉ mang SONAME và một `libreadcfg.so` thật.

## Vì sao trong container lại đơn giản hơn trên host

Binary console mở file cấu hình bằng **đường dẫn tuyệt đối** `/usr/PET/…`,
`/usr/g/…`. Trên host chúng không tồn tại, nên `native/petsw_run.sh` phải dựng
overlay ghi được lên `/usr` rồi bind cây `petsw/` vào, trong một mount namespace
không cần root.

Trong image thì `Dockerfile` đã `COPY` cây đó vào **đúng hai đường dẫn ấy**, nên
toàn bộ trò namespace là thừa — và không chỉ thừa: container không có
`CAP_SYS_ADMIN`, `unshare` sẽ trả `Operation not permitted` và mọi binary console
thành bất khả dụng. `PETSW_ROOT=/` trong image báo cho `petsw_run.sh` biết để
`exec` thẳng, chỉ đặt `LD_LIBRARY_PATH`.

`stub/` phải đứng **trước** mọi thư mục có bản 64-bit trùng tên: `/vendorlib`
mang `libmsghand`/`libcupipc`/`libeventmgr` cho `pet_recon` x86-64, và một loader
32-bit chạm vào đó chỉ tốn công loại sai ELFCLASS.

## Dữ liệu KHÔNG được copy vào container

`d710 decode` bind-mount:

```
<raw>    -> /raw    read-only
<lists>  -> /lists  read-only
<out>    -> /out    ghi được
decode/  -> /decode read-only   (vòng lặp per-bed, sửa không cần build lại)
```

Ba lý do, không phải một:

1. Một exam là hàng chục GB. Copy vào nghĩa là nhân đôi, và bản sao nằm trong
   lớp ghi của container.
2. Dữ liệu acquisition gốc **không tái tạo được**. `rdfx` mở file với
   `accessMode 0` vì mode 1 và 2 đều `O_RDWR`; mount read-only là lớp chặn thứ
   hai cho cùng một điều.
3. Chạy với `--user $(id -u):$(id -g)`, nếu không mọi thứ dưới `--out` trở về
   thuộc quyền root — container chỉ có mỗi user root và bind mount truyền thẳng
   uid của host.

Ngoại lệ duy nhất: GLEPL ghi bản giải nén **cạnh output**, không cạnh input
(`/out/.gerdf_lm`, cỡ bằng file `.BLF`), rồi bị xoá sau mỗi bed.

## Kiểm đúng/sai bằng gì

`convert` **từ chối in MATCH** trừ khi tổng đếm giải mã bằng `totalPrompts` trong
header, và `decode_in.sh` dừng nếu không thấy MATCH. Nên dòng giải mã cũng chính
là dòng kiểm đếm — không có chuyện file cụt trông vẫn như dữ liệu đúng.

Đo thật trên exam 8 bed:

```
bed1  26,114,669  bed3  32,977,313  bed5  44,113,570  bed7  49,041,434
bed2  30,764,762  bed4  41,057,162  bed6  41,568,323  bed8  72,059,155
                                            8/8 MATCH, 38 giây cả exam
```

List-mode bed 1: 93.3 MB `.BLF` → 162.8 MB sau `unglepl` → 26,114,944 event
PETSIRD trong 59 giây. Lệch 275 event so với sinogram là pre-roll; bỏ bằng
`--drop-preroll`.

## Bẫy đã gặp

* **`-Wl,--export-dynamic` không phải tuỳ chọn.** `librdf.so.0` under-linked và
  bind `ErrLog` ngược về nơi nạp nó (ta định nghĩa trong `rdfx.c` thay vì
  `dlopen` cả `libErr.so.0`). Thiếu cờ đó thì `ErrLog` không vào `.dynsym` và
  `dlopen(RTLD_NOW)` chết với `undefined symbol: ErrLog`. Nhánh multilib của
  `build.sh` từng thiếu — không ai thấy vì host không có multilib, container thì
  có. Dòng cuối của `Dockerfile` giờ là smoke test chặn đúng lỗi này.
* **`fopen64`, không phải `fopen`.** `rdfx` là 32-bit nên `FILE*` mang offset
  32-bit và chết câm ở 2 GiB. Một dump TOF đầy đủ là 3.34 GB: `fwrite` bắt đầu
  fail nhưng biến đếm trong RAM vẫn khớp header.
* **`SINO*` 2D không giải mã được ở đây.** Scan norm/cal là mảng 2D không nén
  (`data_type` khác 7); `convert` báo lỗi rõ ràng thay vì đoán.
