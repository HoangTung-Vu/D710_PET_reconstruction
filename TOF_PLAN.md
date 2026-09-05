# TOF: kế hoạch, và trả lời hai câu hỏi trước khi bắt tay

> **TRẠNG THÁI 2026-08-27 (cuối ngày) — kế hoạch này đã chạy xong, và §0.2 sai.**
>
> * §1 (dựng lại image) và §2 (`--tof-mash`) : **xong**, image có 89.2459 / 675.
> * §3 scatter : **xong, nhưng không theo đường file này đoán.** Không phải đảo
>   `GetScatterViewDataTof`. Chỉ cần `IgJobReq.reconMethod = 3`, và
>   `CalcScatterEstimateTof` — thứ §0.2 dưới đây gạt đi là "không phải thứ cần"
>   — chính là thứ cần. Xem `TOF_SCATTER_REVERSE.md`.
> * §0.1 (`S` lặp lại) và §0.3 (`randoms / n_tof`) : **vẫn đúng**, và đã chạy.
> * Phần "ĐÃ ĐO 2026-08-27" trong §3 vẫn đúng như một **phép đo** và vẫn còn
>   trong code làm đường lui (`terms.scatter_tof_profile`); nó chỉ thôi là kế
>   hoạch chính. Đo thêm được: profile toàn cục sai 20.6 % L1 so với phân bố
>   thật của GE.
>
> Phần còn lại giữ nguyên, kể cả chỗ sai, vì lập luận dẫn tới chỗ sai đáng đọc.
>
> **BỔ SUNG 2026-08-29 — có một lỗi thứ hai mà cả kế hoạch này lẫn
> `TOF_SCATTER_REVERSE.md` bản đầu đều không thấy: CHIỀU của trục TOF bị soi
> gương.** Không phải scatter, không phải randoms, không phải `S` — cả ba đều
> đúng như §0 nói. Trục *chỉ số* mới sai: GE đánh số theo `deltaTime` tăng dần,
> STIR đọc index 0 là timing position âm nhất, và một timing position là **độ
> dời có dấu dọc LOR**. Mọi ảnh TOF trước ngày đó đặt hoạt độ sang nửa sai của
> LOR. Đã sửa, có cửa kiểm; xem `TOF_SCATTER_REVERSE.md` §8.
>
> Bài học đáng ghi: §0.1 và §0.3 chứng minh *giá trị* của từng số hạng là đúng,
> và đúng thật. Không câu nào trong đó nói gì về **thứ tự** của trục — mà mọi số
> hạng lại đi qua cùng một trục, nên chúng nhất quán với nhau kể cả khi cùng
> sai. Đó là lý do bước 6 dưới đây (đo lại vs GE) phải làm, không bỏ được.

Viết 2026-08-27, sau khi so bản dựng của mình với VPFXS của GE trên cùng ca
`20260810_FDG26081008`. Mọi khẳng định dưới đây có nguồn: hoặc chữ ký symbol
trong `custom_tool/petsw/usr/PET/release/petig/pet_recon` (có DWARF đầy đủ),
hoặc private tag DICOM của chính series GE dựng, hoặc `vendor/README.md` §3a.

Lấy lại danh sách symbol bất cứ lúc nào:

```bash
nm -C --defined-only custom_tool/petsw/usr/PET/release/petig/pet_recon > /tmp/syms.txt
```

---

## 0. Hai câu hỏi

### 0.1 `sens` — norm và dead time thì sao? **Không cần gì cả.**

Ba mẩu bằng chứng, độc lập nhau:

```
CNorm3d::ApplyNormalization (float*, int, float*, bool)
CDeadtime3d::ApplyDeadtime  (float*, int, float*, int, float*, float*, float*, bool)
```

Mỗi thứ **đúng một class, một hàm apply, không có bản `...Tof`** — trong khi
scatter thì có (xem 0.2). Không phải GE quên: norm là hiệu suất của một *cặp
tinh thể*, dead time là *phân suất sống* của một block trong một frame. Cả hai
không phụ thuộc photon đến sớm hay muộn.

```
COsemTofMain::GetAttnViewData (unsigned long view, float* out)
```

Getter attenuation **bên trong** vòng TOF OSEM: không một tham số TOF nào. Nó
lấy nguyên buffer non-TOF ra dùng.

```
OclMultAttnNormDt::OclMultAttnNormDt (..., OsemTofGpu::COsemTofValues const&, ...)
```

Kernel GPU nhân `attn × norm × dt` **chạy trong vòng TOF** (nhận
`COsemTofValues`), nhưng thứ nó nhân vào là dữ liệu non-TOF. Đó đúng nghĩa đen
là "tính `normdt` một lần, phát lại cho mọi bin TOF".

**Kết luận:** `S = normdt × af` giữ nguyên, chỉ lặp lại trên trục TOF.

```python
S_tof = np.repeat(A["sensitivity"], n_tof, axis=0)   # trục TOF là axis 0
```

Đây là **chính xác**, không phải xấp xỉ, và STIR hợp tác: kernel TOF của
`ProjMatrixByBin` chuẩn hoá sao cho `Σ_t k_t = 1` dọc LOR, nên
`Σ_t S·(G_t x) = S·(G x)` — đúng bằng mô hình non-TOF hiện tại.

### 0.2 "Chỉ cần chọn đúng nhánh là được chứ gì?" — **Không, và tin tốt hơn thế.**

Xếp bốn getter của `COsemTofMain` cạnh nhau thì kiến trúc lộ ra ngay:

| getter | chữ ký | số bản |
|---|---|---|
| attenuation | `GetAttnViewData(view, out)` | 1 |
| randoms | `GetRandomsViewData(view, out)` | 1 |
| scatter | `GetScatterViewDataNonTof(view, out)` | 2 |
| scatter | `GetScatterViewDataTof(ulong, ulong, ulong, const float* in, float* out)` | ↑ |

`GetScatterViewDataTof` nhận **một buffer input** cộng ba tham số chỉ số. Hàm
nhận input rồi ghi output không phải là "một mô hình scatter khác" — nó là một
**phép biến đổi**. non-TOF vào, TOF ra.

Nghĩa là: **GE cũng tính scatter non-TOF** — đúng như `m_pScatter` mình dump
được, `axes: view(288) x v(553) x u(381)` — rồi **trải ra lúc dùng**, từng view
một, ngay trong vòng OSEM.

Khẳng định thêm bởi cặp này:

```
CCorrDataMem::DeallocateBuffsPreOsemTof(bool)
CRawDataMem::DeallocateBuffsPreOsemTof(bool)
```

Buffer của tầng prep được **giải phóng TRƯỚC tầng TOF OSEM**. Hai pha tách bạch,
và pha prep là non-TOF *by design* — không phải do `estimate.py` gọi nhầm.

Nên `CalcScatterEstimateTof` / `calcScatter3dTOF` **không phải thứ cần**. Chúng
là một cấu hình khác (SSS tự chạy TOF-aware). Đường VPFXS đi qua
`GetScatterViewDataTof`.

**Thứ thiếu không phải một lần chạy SSS khác. Là đúng một hàm, chữ ký sạch.**
Đó là tin tốt: phạm vi nhỏ hơn hẳn so với "tự mô hình hoá TOF scatter".

Bảng trọng số nằm ở `COsemTofValues::SetupTofWeights()` và `SetupTofBinDisc()`.

> **↑ Ba đoạn trên SAI.** `CalcScatterEstimateTof` đúng là thứ cần, và nó
> **là** "một lần chạy SSS khác" — chính xác hơn: một pass TOF thêm vào sau 5
> vòng SSS non-TOF, trong cùng `CScatterFully3dModel` mà `extract.gdb` đang lái.
> Cái đã dẫn tới kết luận sai: chỉ nhìn `COsemTofMain` (tầng OSEM) rồi suy ra
> tầng prep không thể sinh TOF. `CCorrDataMem::m_pScatterTOF` — 288 view ×
> 151360 B — được cấp phát ở **mọi** lần chạy, kể cả những lần đã kết luận là
> "prep là non-TOF by design"; log của chính chúng nói vậy.
>
> `GetScatterViewDataTof` vẫn tồn tại và vẫn là hàm nội suy bảng này lúc chạy
> OSEM — nhưng nó không còn chặn đường. Xem `TOF_SCATTER_REVERSE.md`.

### 0.3 randoms

`GetRandomsViewData(view, out)` — một bản, không tham số TOF. Chia đều:

* 55 bin × 89.2459 ps = **4.909 ns**, đúng bằng cửa sổ trùng phùng
  (`pos/negCoincidenceWindow 37/37`, `delayWindowOffset 32`);
* accidental coincidence không có tương quan thời gian.

Nên `randoms / n_tof` là **vật lý đúng**, không phải xấp xỉ tiện tay.

---

## 1. Dựng lại image — làm trước, độc lập với mọi thứ khác

`custom_tool/gerdf/interfile.py` đã sửa trên host nhưng **được bake vào image**
(`/opt/custom_tool`), không mount. Hai hằng số đang sai trong image:

| | cũ | mới | nguồn |
|---|---|---|---|
| `TOF_BIN_SIZE_PS` | 89 | **89.2459** | `coincTimingPrecision = 0.089246 ns` |
| `TOF_RESOLUTION_PS` | 550 | **675** | `TIMING_RESOLUTION` trong `sharcAp.cfg.XR` |

550 là mặc định **Discovery 690** dựng sẵn của STIR, một cái thay thế chung
chung. Đây là D710. Hẹp hơn 23%, và đó là bề rộng kernel STIR tích chập dọc
LOR — không phải chuyện thẩm mỹ.

### Điều kiện đã đủ

* image `d710:full` = `sha256:78b6d173d7eb`, tạo 2026-08-24, 2.5 GB;
* **có đủ build history** (`docker history` thấy từng lệnh `COPY`), nên cache
  ăn tới tận layer đổi;
* `custom_tool/petsw` có mặt, 18 GB; `.dockerignore` cắt context còn ~3.6 GB.

### Lệnh — chạy từ **PROJECT ROOT**, không phải `D710/`

```bash
docker tag d710:full d710:pre-tof                    # đường lui
docker build -t d710:full -f D710/Dockerfile .
```

Chỉ `COPY custom_tool/gerdf` (`Dockerfile:127`) trở đi chạy lại, cộng
`RUN build.sh` (`Dockerfile:136`). Vài giây.

### Kiểm

```bash
docker run --rm d710:full python3 -c \
  'from gerdf.interfile import TOF_BIN_SIZE_PS as B, TOF_RESOLUTION_PS as R; print(B, R)'
# phải in:  89.2459 675

docker run --rm d710:full python3 /opt/custom_tool/ge_rdf_tool.py selftest
```

### Đừng làm thế này

```dockerfile
FROM d710:full
COPY custom_tool/gerdf /opt/custom_tool/gerdf     # ← KHÔNG
```

Nhanh hơn thật, nhưng Dockerfile sẽ thôi mô tả image — đúng cái thất bại mà
`Dockerfile:33-40` kể lại về `bake.sh` và `docker commit`: ảnh chạy được nhưng
không ai dựng lại được vì Dockerfile không nói nó chứa gì.

---

## 2. `--tof-mash` trong gerdf CLI

**Bắt buộc, không phải tối ưu** — và con số quyết định là RAM, không phải đĩa.
OSEM giữ prompts + background + sensitivity cùng lúc, float32.

55 = 5 × 11, nên chỉ có bốn lựa chọn:

| mash | bin | đĩa/bed | RAM cho OSEM | TOF res. hiệu dụng | lợi ích SNR còn lại |
|---|---|---|---|---|---|
| 1 | 55 | 6.7 GB | **40 GB** | 681 ps = 102 mm | 100 % |
| **5** | **11** | **1.33 GB** | **8 GB** | **809 ps = 121 mm** | **92 %** |
| 11 | 5 | 0.61 GB | 3.6 GB | 1191 ps = 179 mm | 76 % |
| 55 | 1 | 0.12 GB | 0.7 GB | — | non-TOF |

Resolution hiệu dụng = quadrature của **675 ps** (`TIMING_RESOLUTION`) với bề
rộng bin; lợi ích SNR ~ `sqrt(D/dx)`.

Máy này có **30 GB** (20 GB available), nên **giữ cả 55 bin không chạy được** —
đó là tường cứng, không phải sở thích. Gộp 11 thì thừa chỗ nhưng bin rộng 982 ps
> 675 ps của máy nên mất 24 %. **Gộp 5 là cái tốt nhất còn vừa: mất 8 %.**

`d710` mặc định `--tof-mash 5`; `gerdf convert` để mặc định `1` — nó là công cụ,
giữ nguyên dữ liệu, chỗ đặt chính sách là wrapper.

Hạ tầng đã sẵn: `interfile.py:make_header(..., num_tof_bins, tof_mashing)` nhận
tham số và phát đúng key của STIR 6.4 —
`TOF mashing factor := 5` cạnh `Maximum number of (unmashed) TOF time bins := 55`.
Chỉ CLI là all-or-nothing (`cli.py:987`, `cli.py:1013`). Thêm `--tof-mash N`
vào cùng chỗ `--collapse-tof` đang đứng, rồi truyền xuống.

Cửa kiểm có sẵn: `convert` từ chối báo `MATCH` nếu tổng đếm giải mã khác header
prompts. Sau mash tổng phải **không đổi**.

---

## 3. Lắp background TOF trong `terms.py`

`utils/terms.py:load()` hiện **raise** khi prompts có TOF mà các số hạng thì
không — guard đó thêm vào lúc đặt TOF thành mặc định, để tránh numpy broadcast
im lặng. Thay nó bằng phần lắp ráp.

Ba số hạng, ba việc khác nhau. **Đừng gộp vào một hàm** — độ chắc chắn của
chúng khác nhau xa:

| số hạng | cách | chắc đến đâu |
|---|---|---|
| `S` = norm × dt × attn | `repeat` trên trục TOF | **chắc** — GE làm y hệt, xem §0.1 |
| randoms | `/ n_tof` | **chắc** — cửa sổ 4.909 ns, xem §0.3 |
| scatter | ~~đảo `GetScatterViewDataTof`~~ → `reconMethod = 3` | **xong** — kernel của chính GE |

### Cách rẻ nhất để đọc `GetScatterViewDataTof`

Đừng đọc disassembly. `vendor/` đã có sẵn hạ tầng gọi thẳng hàm của vendor
trong gdb (`README.md` §3a, cùng đường đã dùng cho `CPrep3d::DoTask`). Nạp một
scatter view non-TOF đã biết, gọi hàm, đọc buffer output — profile lộ ra ngay.

Ba `unsigned long` nhiều khả năng là `(view, nOffset, nCount)` hoặc
`(nu, nv, ntof)`; gọi thử với giá trị đã biết là phân biệt được ngay, rẻ hơn
đọc mã máy.

### ĐÃ ĐO 2026-08-27: trải phẳng KHÔNG chấp nhận được, và có đường thay thế

`tools/tof_profile.py` kiểm trục TOF bằng chính dữ liệu, không gọi hàm nào của
vendor. Nền tảng là một sự thật hình học: **LOR càng xa tâm càng ít đi qua bệnh
nhân**, nên cắt sinogram theo bán kính là tách được ba thành phần.

Chạy trên 5 view của bed 1 (243.480 đếm):

| | kết quả |
|---|---|
| trục TOF | đỉnh bin 29/55, nửa chiều cao bin 19..36 **liền mạch** → đơn điệu, **gộp bin liền kề là đúng** |
| randoms, \|u\|>280 mm | CoV **0.0404** vs sàn Poisson **0.0393** → **phẳng đúng bằng sai số đo** |
| randoms, \|u\| 200–280 mm | CoV 0.108 vs sàn 0.048 → chưa phẳng, scatter còn tới đó |
| scatter (vành 140–210 mm trừ nền randoms) | đỉnh bin 28, **max/mean 4.61**, gần như bằng 0 ngoài bin 15–40 |
| nếu trải scatter phẳng | sai **4.6x** ở đỉnh; **38/55 bin (69 %)** có prompts < background → **true rate ÂM** |

Nên `randoms / n_tof` **đã được xác nhận bằng đo**, còn `scatter / n_tof` thì
hỏng nặng — không phải sai số nhỏ đổi lấy tiện lợi.

**Đường thay thế không cần vendor.** Cái bảng trên không chỉ bác bỏ trải phẳng,
nó còn **đo được luôn profile TOF của scatter** — chính là thứ
`GetScatterViewDataTof` lẽ ra sinh ra. `--save prof.npy` ghi profile đã chuẩn
hoá. Vùng lấy nó (`CalcSinoTails`, `SCAT_TAILFIT_ANGLE_WINDOW 31`) đúng là vùng
GE dùng để chuẩn scatter.

Hạn chế phải nói rõ: đây là **một** profile toàn cục, trong khi phân bố TOF thật
của scatter đổi theo LOR (bán kính, và chỗ photon bị tán xạ dọc LOR). Tốt hơn
phẳng rất nhiều, vẫn kém `GetScatterViewDataTof` làm theo từng view. Thứ tự đúng
vẫn là: phẳng (chạy được) → profile đo được (dùng được) → vendor (đúng).

**Đã đo được hạn chế đó, sau khi có bản của GE:** tâm profile chạy từ bin 21.4
tới 31.4 theo `(view, u)` — trải 10 bin = 890 ps = 134 mm — và một profile toàn
cục lệch **20.6 %** (khoảng cách L1, trọng số theo số đếm). Đường "vendor
(đúng)" hoá ra rẻ hơn cả ba, nên nó là mặc định; profile đo được còn lại làm
đường lui cho bed nào estimate trước khi có `reconMethod = 3`.

---

## 4. Thứ tự, và điểm dừng an toàn

1. ~~**Dựng lại image** (§1)~~ — **xong**, image có 89.2459 / 675.
2. ~~**`--tof-mash`** (§2)~~ — **xong**, `d710` mặc định mash 5.
3. ~~**Decode một bed** `--tof --tof-mash 5`, kiểm `MATCH`~~ — **xong** trên ped
   bed 1: 55 → 11 bin, 1.33 GB, `MATCH` 18.759.294 prompts.
4. ~~**scatter chia đều tạm**~~ — **bỏ qua được**, vì bước 5 hoá ra rẻ hơn.
5. ~~**Đọc `GetScatterViewDataTof`**~~ → **`IgJobReq.reconMethod = 3`**, xong;
   `TOF_SCATTER_REVERSE.md`. Kiểm được ngay ở tổng theo từng bin TOF: với
   scatter của GE thì **0/11 bin có true rate âm**, còn chia đều thì 69 % âm.
6. **Đo lại vs GE**: CoV, corr, bias — và thứ đáng nhìn nhất là **vùng nhiễu
   thấp** (gan, não), vì đó là chỗ TOF phải thắng rõ nhất. **Chưa làm.**

Ở mọi bước, `d710 decode --no-tof` vẫn cho đúng đường cũ, và guard trong
`terms.py` chặn chạy nhầm cấu hình lẫn lộn.

---

## 5. Không nằm trong plan này

* **PSF (SharpIR)** — LUT xuyên tâm 47.6 KB có sẵn
  (`systemConfig/local/psfLUT.XR`), kernel trục đúng nghĩa đen là
  `[1/6, 2/3, 1/6]`. Việc riêng, làm sau TOF.
* **2×24 thay cho 3×12** — GE dùng 2 iteration × 24 subset. Miễn phí, nhưng đổi
  cùng lúc với TOF thì không biết cái nào gây ra thay đổi gì.
* **Cánh tay thấp 30%** ở r>175 mm, và **lệch 2.7 mm theo y** — hai việc độc
  lập, xem so sánh cặp với GE.
