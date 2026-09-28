# Bài toán tái tạo ảnh PET trên GE Discovery 710 — lời giải toán học

Tài liệu này chỉ chứa **toán**: mô hình thuận, hàm hợp lý, dẫn xuất thuật toán,
chứng minh các tính chất được dùng, và các hằng số hình học của máy. Mọi chi
tiết cài đặt (định dạng file, API, bộ nhớ, CLI) đều được bỏ qua.

Hai thuật toán được giải đầy đủ:

| | dữ liệu | mô hình | thuật toán |
|---|---|---|---|
| **A** | sinogram, không TOF | $\bar y = S\,(Gx) + b$ | OSEM (§5) |
| **B** | danh sách sự kiện, có TOF | $\bar y_{l,t} = w_l (G_t x)_l + b_{l,t}$ | LM-TOF-OSEM (§7) |

Cả hai cùng một ảnh, cùng một lưới, cùng bốn số hạng hiệu chỉnh; khác nhau ở
**không gian dữ liệu** (bin hay sự kiện) và ở việc **có mô hình hoá thời gian
bay hay không**. §7.3 chứng minh một kết quả làm hai đường gặp nhau: ảnh
sensitivity của bài toán list-mode có TOF **bằng đúng** ảnh sensitivity không
TOF.

---

## 1. Ký hiệu

### 1.1 Tập chỉ số

| ký hiệu | nghĩa | giá trị D710 |
|---|---|---|
| $N_r$ | số vòng tinh thể (ring) | $24$ |
| $N_d$ | số tinh thể trên một vòng | $576$ |
| $N_c = N_r N_d$ | tổng số tinh thể | $13\,824$ |
| $c \in \{0,\dots,N_c-1\}$ | chỉ số tinh thể | |
| $l = (c_1,c_2)$ | một **LOR** = một cặp tinh thể | $N_L = N_r^2 N_v N_u = 63\,203\,328$ |
| $v \in \{0,\dots,N_v-1\}$ | góc chiếu (view) | $N_v = 288$ |
| $u \in \{0,\dots,N_u-1\}$ | bin xuyên tâm (tangential) | $N_u = 381$ |
| $p \in \{0,\dots,N_p-1\}$ | mặt phẳng sinogram (axial × segment) | $N_p = 553$ |
| $i = (p,v,u)$ | một **bin** sinogram | $N_b = N_p N_v N_u = 60\,679\,584$ |
| $t \in \{0,\dots,N_T-1\}$ | bin thời gian bay (TOF) | $N_T \in \{1,5,11,55\}$ |
| $j$ | voxel ảnh | $N_x = 47\cdot337^2 = 5\,337\,743$ |
| $e \in \{1,\dots,N_E\}$ | sự kiện trong list-mode | $1.9\!-\!8.7\times10^7$ / bed |
| $m \in \{1,\dots,M\}$ | tập con (subset) của OSEM | $M = 24$ |
| $k$ | chỉ số vòng lặp (subiteration) | |

### 1.2 Đại lượng

| ký hiệu | nghĩa | miền |
|---|---|---|
| $x_j \ge 0$ | hoạt độ tích luỹ trong voxel $j$ (count/voxel) | $\mathbb R_+^{N_x}$ |
| $y_i \in \mathbb N$ | số **prompt** đếm được ở bin $i$ | dữ liệu |
| $\bar y_i$ | kỳ vọng của $y_i$ | |
| $G_{ij}$ | ma trận hình học (xác suất/độ dài) | $\mathbb R_+^{N_b\times N_x}$ |
| $\eta_i$ | hiệu suất cặp tinh thể (normalisation) | $(0,\infty)$ |
| $\delta_i$ | phân suất sống (live fraction, dead time) | $(0,1]$ |
| $\alpha_i$ | xác suất sống sót do suy giảm (attenuation factor) | $(0,1]$ |
| $S_i = \eta_i \delta_i \alpha_i$ | **sensitivity** của bin $i$ | $(0,\infty)$ |
| $r_i$ | kỳ vọng trùng phùng ngẫu nhiên (randoms) | $\mathbb R_+$ |
| $s_i$ | kỳ vọng tán xạ (scatter) | $\mathbb R_+$ |
| $b_i = r_i + s_i$ | nền cộng (background) | $\mathbb R_+$ |
| $\sigma_j = \sum_i S_i G_{ij}$ | **ảnh sensitivity** | $\mathbb R_+^{N_x}$ |
| $\mu(\mathbf z)$ | hệ số suy giảm tuyến tính tại 511 keV | mm$^{-1}$ |
| $\mathfrak m_i \in \{1,2\}$ | số cặp ring gộp vào bin $i$ (span-2) | |
| $w_l$ | sensitivity của **một** LOR $l$ | |
| $a_{l,t}$ | nền cộng đã chia trọng số, $a = b/w$ | |
| $h_t(\ell)$ | nhân TOF của bin $t$ tại vị trí $\ell$ dọc LOR | $[0,1]$ |

Quy ước: $G$ là toán tử tuyến tính, $G^{\mathsf T}$ là chuyển vị (back-projection);
$\odot$ và phân số giữa hai vector là phép nhân/chia theo phần tử;
$\langle\cdot,\cdot\rangle$ là tích vô hướng Euclid.

### 1.3 Hằng số máy (dùng trong mọi công thức số bên dưới)

$$
\begin{aligned}
&R = 405.10\ \text{mm}, \quad
 d_{\mathrm{DOI}} = 8.4\ \text{mm}, \quad
 R_{\mathrm{eff}} = R + d_{\mathrm{DOI}} = 413.50\ \text{mm},\\
&\Delta_{\mathrm{ring}} = 6.5399994\ \text{mm}, \quad
 \Delta_z = \tfrac12\Delta_{\mathrm{ring}} = 3.2699997\ \text{mm}, \quad
 \Delta_x = \Delta_y = 2.1306\ \text{mm},\\
&c = 0.299792458\ \text{mm/ps}, \quad
 \tau_{\mathrm{LSB}} = 89.2459\ \text{ps}, \quad
 \tau_{\mathrm{FWHM}} = 675\ \text{ps},\\
&\mu_{\text{nước}} = 0.0093\ \text{mm}^{-1}, \quad
 \mu_{\text{xương}} = 0.0166\ \text{mm}^{-1}.
\end{aligned}
$$

---

## 2. Mô hình thuận

### 2.1 Từ vật lý đến tích phân đường

Gọi $f(\mathbf z) \ge 0$ là mật độ hoạt độ (Bq/mL) tại điểm $\mathbf z\in\mathbb R^3$.
Mỗi phân rã $\beta^+$ sinh hai photon 511 keV bay ngược chiều nhau. Nếu cả hai
được ghi nhận trong cửa sổ trùng phùng, cặp tinh thể $(c_1,c_2)$ xác định một
đường **LOR** $\mathcal L_l$. Bỏ qua acollinearity và độ dài quãng chạy
positron, số photon-cặp phát ra dọc $\mathcal L_l$ trong thời gian thu $T$ tỉ lệ
với tích phân đường

$$
[\mathcal X f](l) \;=\; \int_{\mathcal L_l} f(\mathbf z)\, \mathrm d\ell .
\tag{2.1}
$$

**Suy giảm.** Một cặp photon dọc $\mathcal L_l$ chỉ tới được cả hai đầu nếu
không photon nào bị hấp thụ. Với hai xác suất độc lập và đường đi hợp lại đúng
bằng toàn bộ $\mathcal L_l$ (bất kể điểm phát nằm ở đâu — đây chính là lý do
attenuation trong PET là hệ số **của LOR** chứ không của điểm):

$$
\alpha_l \;=\;
\exp\!\Big(-\!\!\int_{\mathcal L_l^{(1)}}\!\!\mu\Big)\cdot
\exp\!\Big(-\!\!\int_{\mathcal L_l^{(2)}}\!\!\mu\Big)
\;=\;
\exp\!\Big(-\!\!\int_{\mathcal L_l}\!\mu(\mathbf z)\,\mathrm d\ell\Big).
\tag{2.2}
$$

$\mu$ được suy từ CT bằng phép biến đổi song tuyến Carney

$$
\mu(\mathrm{HU}) =
\begin{cases}
\mu_w\left(1 + \dfrac{\mathrm{HU}}{1000}\right), & \mathrm{HU}\le 0,\\[2mm]
\mu_w + \mathrm{HU}\,\dfrac{\mu_b-\mu_w}{1000\,\beta_{\mathrm{kVp}}}, & \mathrm{HU} > 0,
\end{cases}
\qquad \beta_{120}=0.837 .
\tag{2.3}
$$

**Hiệu suất và dead time.** Mỗi cặp tinh thể có hiệu suất riêng $\eta_l$
(hình học vi mô + độ nhạy tinh thể + hiệu ứng khối); mỗi khối detector có phân
suất sống $\delta_l \in (0,1]$ giảm khi tốc độ singles tăng. Cả hai **không phụ
thuộc thời điểm đến của photon** (xem §6.4).

Gộp lại, kỳ vọng số **trues** đếm được trên LOR $l$ là
$S_l\,[\mathcal X f](l)$ với $S_l=\eta_l\delta_l\alpha_l$.

### 2.2 Rời rạc hoá

Khai triển $f$ trên cơ sở voxel $\{\chi_j\}$, $f \approx \sum_j x_j \chi_j$.
Tuyến tính của (2.1) cho

$$
[\mathcal X f](l) \approx \sum_j G_{lj} x_j, \qquad
G_{lj} = \int_{\mathcal L_l} \chi_j(\mathbf z)\,\mathrm d\ell ,
\tag{2.4}
$$

tức $G_{lj}$ là độ dài giao của tia với voxel $j$ (Siddon), hoặc trọng số nội
suy tuyến tính của Joseph khi lấy mẫu $\mathcal L_l$ theo bước đều — hai đường
tính dùng hai biến thể này, và chúng chỉ khác nhau ở mô hình lấy mẫu, không ở
mô hình thống kê bên dưới. Khi tia được lấy mẫu theo bước $\Delta$ dọc trục
trội, $G$ tích luỹ **theo bước voxel**; do đó $G$ — và mọi hằng số hiệu chuẩn
suy ra từ nó (§9.4) — **phụ thuộc kích thước voxel**.

Ngoài ra, mô hình độ phân giải (PSF) được đưa vào bằng một toán tử làm trơn
$B$ trong **không gian ảnh**:

$$
G \;\longrightarrow\; G B, \qquad
B = \text{Gauss 3D, FWHM } 6.4\ \text{mm}.
\tag{2.5}
$$

$B$ đối xứng ($B=B^{\mathsf T}$) nên chuyển vị của mô hình là $B G^{\mathsf T}$;
mọi công thức dưới đây giữ nguyên khi thay $G$ bởi $GB$.

### 2.3 Hai số hạng cộng

Ngoài trues, máy còn đếm:

* **Randoms** $r_i$ — hai photon từ hai phân rã khác nhau rơi vào cùng cửa sổ
  trùng phùng $2\tau$. Với tốc độ singles $S_1, S_2$ trên hai detector,
  $r = 2\tau S_1 S_2$: **không** liên quan đến $x$ theo cách tuyến tính-hình
  học, và về thống kê là một nguồn cộng độc lập.
* **Scatter** $s_i$ — cặp photon thật nhưng ít nhất một photon đã tán xạ
  Compton, nên LOR ghi được không đi qua điểm phát. Ước lượng bằng mô phỏng
  tán xạ đơn (SSS) trên $\mu$-map.

Cả hai được ước lượng **trong miền số đếm đo được**, tức đã bao gồm $\eta,
\delta$ (chúng là kỳ vọng của các coincidence *thực sự ghi được*). Đây là lý do
toán học khiến chúng **không** đi qua $S$:

$$
\boxed{\;\bar y \;=\; S \odot (G x) \;+\; b\;},\qquad b = r + s .
\tag{2.6}
$$

(Nếu, ngược lại, một số hạng cộng được cho trong miền *phát xạ* thì nó phải
nằm trong ngoặc: $\bar y = S\odot(Gx + a) + b$. Ở đây $a\equiv 0$.)

### 2.4 Mô hình thống kê

> **Mệnh đề 2.1.** Với $x$ cố định, các biến đếm $\{y_i\}$ độc lập và
> $y_i \sim \mathrm{Poisson}(\bar y_i)$ với $\bar y_i$ cho bởi (2.6).

*Chứng minh (phác).* Phân rã phóng xạ là quá trình Poisson không thuần nhất
trên $\mathbb R^3\times[0,T]$. Ghi nhận một cặp photon là một phép **thinning
độc lập** (mỗi phân rã được giữ với xác suất phụ thuộc vị trí, qua hình học,
suy giảm và hiệu suất) rồi **ánh xạ** vào không gian bin. Thinning độc lập và
ánh xạ đo được của một quá trình Poisson lại cho quá trình Poisson, và các bin
rời nhau cho các biến độc lập; cường độ của bin $i$ chính là $S_i (Gx)_i$.
Randoms là một quá trình Poisson độc lập cường độ $r_i$ (xấp xỉ chuẩn cho
trùng phùng ngẫu nhiên), scatter là ảnh của một phép thinning khác, cường độ
$s_i$. Tổng hai quá trình Poisson độc lập là Poisson với cường độ cộng. $\square$

Hai chỗ mệnh đề này là **xấp xỉ**, và cần nhớ khi đọc kết quả:
(i) dead time làm các biến đếm bớt Poisson (phương sai nhỏ hơn trung bình);
(ii) $r$ và $s$ là *ước lượng*, không phải hằng số đã biết, nên phương sai của
chúng không được lan truyền.

### 2.5 Hình học chỉ số bin

**Xuyên tâm.** Bin $(v,u)$ ứng với cặp detector

$$
d_1 = \Big(v + \big\lfloor \tfrac{\hat u}{2}\big\rfloor\Big) \bmod N_d, \qquad
d_2 = \Big(v - \big\lceil \tfrac{\hat u}{2}\big\rceil + N_v\Big) \bmod N_d,
\qquad \hat u = u - \big\lfloor \tfrac{N_u}{2}\big\rfloor .
\tag{2.7}
$$

Vì $d_1 - d_2 = \hat u + N_v = \hat u + N_d/2$, và dây cung nối hai điểm ở góc
$2\pi d/N_d$ trên đường tròn bán kính $R_{\mathrm{eff}}$ cách tâm một khoảng
$R_{\mathrm{eff}}\lvert\cos\frac{\phi_1-\phi_2}{2}\rvert$, ta được **khoảng
cách xuyên tâm** của bin:

$$
s(\hat u) \;=\; R_{\mathrm{eff}}\,
\Big|\cos\Big(\tfrac{\pi \hat u}{N_d} + \tfrac{\pi}{2}\Big)\Big|
\;=\; R_{\mathrm{eff}}\,\sin\!\Big(\tfrac{\pi |\hat u|}{N_d}\Big).
\tag{2.8}
$$

Ba hệ quả trực tiếp:

1. Lưới xuyên tâm **không đều** (dữ liệu không arc-corrected): bước
   $\dfrac{\mathrm ds}{\mathrm d\hat u} = \dfrac{\pi R_{\mathrm{eff}}}{N_d}
   \cos\dfrac{\pi \hat u}{N_d}$, bằng $2.2553$ mm ở tâm và co lại ở rìa.
2. **Bán kính FOV ngang** là $s$ tại $\hat u_{\max} = (N_u-1)/2 = 190$:
   $$
   \rho_{\max} = R_{\mathrm{eff}}\sin\!\Big(\frac{\pi (N_u-1)}{2N_d}\Big) = 355.8\ \text{mm}.
   \tag{2.9}
   $$
   Ngoài bán kính này **không LOR nào cắt voxel**, nên $\sigma_j = 0$ ở đó —
   xem §5.3.
3. Lưới ảnh vuông cạnh $337\times2.1306 = 718.0$ mm có góc ở bán kính
   $506$ mm $\gg \rho_{\max}$.

**Dọc trục (span-2).** Bin trục được đánh chỉ số bằng tổng vòng
$z = \rho_1+\rho_2$ và hiệu vòng $\delta = \rho_1-\rho_2$; segment $0$ gom
$\delta\in\{-1,0,+1\}$. Điều kiện $\rho_{1,2}=\frac{z\mp\delta}{2}\in\mathbb Z$ cho

$$
\mathfrak m(z) \;=\; \#\{\delta : z-\delta \text{ chẵn}, \ 0\le \tfrac{z-\delta}{2},\ \tfrac{z+\delta}{2} < N_r\}
= \begin{cases} 1, & z \text{ chẵn } (\delta = 0),\\ 2, & z \text{ lẻ } (\delta = \pm1).\end{cases}
\tag{2.10}
$$

Segment $0$ có $2N_r-1 = 47$ mặt phẳng — đúng bằng số mặt phẳng ảnh của một bed.

> **Mệnh đề 2.2 (tính nhất quán của span-2).** Nếu $\mathfrak m_i$ LOR gộp
> vào bin $i$ có cùng hình chiếu $g$ và cùng sensitivity trên mỗi LOR
> $\bar w_i$, thì (2.6) vẫn đúng nguyên dạng với
> $$S_i = \mathfrak m_i\,\bar w_i, \qquad y_i = \textstyle\sum_{l\in i} y_l, \qquad b_i = \sum_{l\in i} b_l,$$
> trong khi $G$ chỉ bắn **một** tia đại diện.

*Chứng minh.* $\bar y_i = \sum_{l \in i}\big[\bar w_i g + b_l\big]
= (\mathfrak m_i \bar w_i)\,g + b_i = S_i (Gx)_i + b_i$. $\square$

Hệ quả thực dụng: $y$, $b$ và $S$ **cùng** mang thừa số $\mathfrak m_i$, nên
mô hình bin tự nhất quán và **không được** nhân $\mathfrak m$ thêm lần nữa.
Ngược lại, khi tách bin về từng LOR (list-mode, §7.4) thì phải **chia** cho
$\mathfrak m_i$.

---

## 3. Ước lượng hợp lý cực đại và MLEM

Đặt $A = \mathrm{diag}(S)\,G$, tức $A_{ij} = S_i G_{ij}$, và
$\bar y(x) = Ax + b$.

### 3.1 Hàm mục tiêu

$$
L(x) \;=\; \log \prod_i \frac{\bar y_i^{\,y_i} e^{-\bar y_i}}{y_i!}
\;\overset{c}{=}\; \sum_i \Big[\, y_i \log\big((Ax)_i + b_i\big) - (Ax)_i \,\Big],
\tag{3.1}
$$

(bỏ hằng số không phụ thuộc $x$). Bài toán:

$$
\hat x \;=\; \arg\max_{x \ge 0} L(x).
\tag{3.2}
$$

> **Mệnh đề 3.1.** $L$ lõm trên $\{x: Ax+b > 0\}$.

*Chứng minh.* $\nabla^2 L = -A^{\mathsf T}\mathrm{diag}\!\big(y_i/\bar y_i^2\big)A$.
Với mọi $d$: $d^{\mathsf T}\nabla^2L\,d = -\sum_i \frac{y_i}{\bar y_i^2}(Ad)_i^2 \le 0$
vì $y_i \ge 0$. $\square$

Gradient và điều kiện dừng KKT:

$$
\frac{\partial L}{\partial x_j} = \sum_i A_{ij}\frac{y_i}{\bar y_i} - \sigma_j,
\qquad \sigma_j \equiv \sum_i A_{ij} = \sum_i S_i G_{ij},
\tag{3.3}
$$

$$
x_j \ge 0, \qquad
x_j\Big(\sum_i A_{ij}\frac{y_i}{\bar y_i} - \sigma_j\Big) = 0, \qquad
\sum_i A_{ij}\frac{y_i}{\bar y_i} - \sigma_j \le 0 .
\tag{3.4}
$$

$\sigma$ là **ảnh sensitivity**: back-projection của toàn bộ hệ số nhân. Nó
xuất hiện lại ở §9.2 làm trọng số ghép bed, và đó không phải trùng hợp (§9.2
chứng minh nó là nghịch đảo phương sai).

### 3.2 Bổ đề multinomial

> **Bổ đề 3.2.** Cho $N_1,\dots,N_m$ độc lập, $N_q\sim\mathrm{Poisson}(\mu_q)$,
> $N=\sum_q N_q$. Khi đó
> $\ (N_1,\dots,N_m)\,\big|\,\{N=n\} \sim \mathrm{Multinomial}\big(n; \mu_q/\textstyle\sum_{q'}\mu_{q'}\big).$

*Chứng minh.* Với $\sum n_q = n$, $\mu = \sum\mu_q$:
$$
\Pr[N_q=n_q \,\forall q \mid N=n]
= \frac{\prod_q e^{-\mu_q}\mu_q^{n_q}/n_q!}{e^{-\mu}\mu^{n}/n!}
= \frac{n!}{\prod_q n_q!}\prod_q\Big(\frac{\mu_q}{\mu}\Big)^{n_q}. \qquad\square
$$

Hệ quả: $\mathbb E[N_q \mid N=n] = n\,\mu_q/\mu$.

### 3.3 EM → MLEM

**Dữ liệu đầy đủ.** Cho $n_{ij}$ = số count ở bin $i$ phát ra từ voxel $j$, và
$n_{i0}$ = số count nền ở bin $i$; giả thiết chúng độc lập,
$n_{ij}\sim\mathrm{Poisson}(A_{ij}x_j)$, $n_{i0}\sim\mathrm{Poisson}(b_i)$, và
dữ liệu quan sát là $y_i = \sum_j n_{ij} + n_{i0}$. Log-likelihood đầy đủ:

$$
L_c(x) \overset{c}{=} \sum_{i,j}\big[\,n_{ij}\log(A_{ij}x_j) - A_{ij}x_j\,\big].
\tag{3.5}
$$

**E-step.** Theo Bổ đề 3.2 áp cho $m = N_x+1$ nguồn Poisson của bin $i$:

$$
\bar n_{ij}^{(k)} \;=\; \mathbb E\big[n_{ij}\,\big|\,y, x^{(k)}\big]
\;=\; y_i\,\frac{A_{ij}x_j^{(k)}}{\bar y_i^{(k)}},
\qquad \bar y_i^{(k)} = (Ax^{(k)})_i + b_i .
\tag{3.6}
$$

**M-step.** Thay (3.6) vào (3.5) và cho đạo hàm theo $x_j$ bằng 0:

$$
\frac{\partial}{\partial x_j}\sum_i\big[\bar n_{ij}^{(k)}\log x_j - A_{ij}x_j\big]
= \frac{1}{x_j}\sum_i \bar n_{ij}^{(k)} - \sigma_j = 0 ,
$$

$$
\boxed{\;
x_j^{(k+1)} \;=\; \frac{x_j^{(k)}}{\sigma_j}\sum_i A_{ij}\,
\frac{y_i}{(Ax^{(k)})_i + b_i}\;}
\tag{3.7}
$$

Đây là **MLEM** (Shepp–Vardi, mở rộng cho nền cộng). Viết theo toán tử:

$$
x^{(k+1)} = \frac{x^{(k)}}{\sigma}\odot A^{\mathsf T}\!\left[\frac{y}{Ax^{(k)}+b}\right],
\qquad
\sigma = A^{\mathsf T}\mathbf 1 .
\tag{3.7'}
$$

### 3.4 Đơn điệu tăng (không cần lý thuyết EM tổng quát)

> **Định lý 3.3.** Với $x^{(k)} > 0$ và $\bar y^{(k)} > 0$, dãy (3.7) thoả
> $L(x^{(k+1)}) \ge L(x^{(k)})$, đẳng thức chỉ khi $x^{(k+1)} = x^{(k)}$.

*Chứng minh.* Đặt, với $i$ cố định,
$$
\alpha_{ij} = \frac{A_{ij}x_j^{(k)}}{\bar y_i^{(k)}},\qquad
\beta_i = \frac{b_i}{\bar y_i^{(k)}},\qquad
\sum_j \alpha_{ij} + \beta_i = 1,\ \ \alpha,\beta\ge 0 .
$$
Vì $\log$ lõm, bất đẳng thức Jensen cho mọi $x\ge0$:
$$
\log\Big(\sum_j A_{ij}x_j + b_i\Big)
= \log\Big(\sum_j \alpha_{ij}\frac{A_{ij}x_j}{\alpha_{ij}} + \beta_i\frac{b_i}{\beta_i}\Big)
\;\ge\; \sum_j \alpha_{ij}\log\frac{A_{ij}x_j}{\alpha_{ij}} + \beta_i\log\frac{b_i}{\beta_i}.
$$
Nhân $y_i$, cộng theo $i$, trừ $\sum_i (Ax)_i$, ta được hàm **minorant**
$$
\varphi(x\,|\,x^{(k)}) = \sum_j\Big[\Big(\sum_i y_i\alpha_{ij}\Big)\log x_j - \sigma_j x_j\Big] + \text{const},
$$
thoả $\varphi(x|x^{(k)}) \le L(x)$ với mọi $x\ge0$ và
$\varphi(x^{(k)}|x^{(k)}) = L(x^{(k)})$ (Jensen thành đẳng thức tại $x=x^{(k)}$).
$\varphi$ **tách theo $j$** và lõm ngặt theo $\log x_j$; cực đại duy nhất của
nó đạt tại đúng (3.7). Do đó
$$
L(x^{(k+1)}) \;\ge\; \varphi(x^{(k+1)}|x^{(k)}) \;\ge\; \varphi(x^{(k)}|x^{(k)}) \;=\; L(x^{(k)}). \qquad\square
$$

Vì $L$ lõm và tập chấp nhận được lồi, mọi điểm tụ của dãy thoả KKT (3.4), tức
là nghiệm ML.

### 3.5 Hai bất biến của phép cập nhật

> **Mệnh đề 3.4 (bảo toàn số đếm).**
> $$\sum_j \sigma_j x_j^{(k+1)} \;=\; \sum_i y_i\,\frac{(Ax^{(k)})_i}{(Ax^{(k)})_i + b_i}.$$
> Riêng khi $b\equiv 0$: $\ \sum_j \sigma_j x_j^{(k+1)} = \sum_i y_i$ với mọi $k$.

*Chứng minh.* Nhân (3.7) với $\sigma_j$ rồi cộng theo $j$, đổi thứ tự tổng:
$\sum_j x^{(k)}_j\sum_i A_{ij}\frac{y_i}{\bar y_i^{(k)}}
= \sum_i \frac{y_i}{\bar y_i^{(k)}}(Ax^{(k)})_i$. $\square$

Nói cách khác, MLEM **bảo toàn tổng số đếm có trọng số**; mọi count bị đẩy ra
ngoài vùng có nghĩa là count bị **lấy mất** khỏi bệnh nhân, chứ không biến mất
vô hại (đây là lý do §5.3 tồn tại).

> **Mệnh đề 3.5 (bất biến không âm và bất biến 0).** Nếu $x^{(k)}\ge0$ thì
> $x^{(k+1)}\ge0$. Hơn nữa $x_j^{(k)} = 0 \Rightarrow x_j^{(k')} = 0$ với mọi
> $k' > k$.

*Chứng minh.* (3.7) là tích của các đại lượng không âm; thừa số $x_j^{(k)}$
đứng ngoài nên $0$ là điểm bất động của toạ độ $j$. $\square$

Đây là công cụ **áp đặt giá đỡ (support)** duy nhất cần dùng: đặt $x^{(0)}_j=0$
ở đâu, ảnh sẽ bằng 0 ở đó mãi mãi — cắt sau khi tái tạo thì đã muộn, vì count
đã bị bơm ra ngoài (Mệnh đề 3.4).

---

## 4. OSEM

### 4.1 Định nghĩa

Chia tập bin thành $M$ tập con rời nhau $\mathcal S_1,\dots,\mathcal S_M$,
$\bigcup_m \mathcal S_m = \{1..N_b\}$. Đặt

$$
\sigma^{(m)}_j = \sum_{i\in\mathcal S_m} A_{ij}, \qquad
\sum_{m} \sigma^{(m)} = \sigma .
\tag{4.1}
$$

Một **subiteration** OSEM là (3.7) giới hạn trên $\mathcal S_m$:

$$
\boxed{\;
x^{(k+1)}_j = \frac{x^{(k)}_j}{\sigma^{(m_k)}_j}
\sum_{i\in \mathcal S_{m_k}} A_{ij}\,\frac{y_i}{(Ax^{(k)})_i + b_i}\;}
\qquad m_k = (k \bmod M) + 1 .
\tag{4.2}
$$

Một **iteration** = $M$ subiteration liên tiếp, quét hết mọi tập con.
Cấu hình dùng ở đây: $M = 24$, $n_{\mathrm{it}} = 2$, tức $48$ subiteration.

### 4.2 Điều kiện cân bằng tập con

> **Mệnh đề 4.1.** Nếu $\sigma^{(m)} = \frac1M \sigma$ với mọi $m$ (tập con
> *cân bằng*) và dữ liệu **nhất quán** (tồn tại $x^\ast\ge0$ với
> $Ax^\ast + b = y$), thì $x^\ast$ là điểm bất động của mọi subiteration (4.2).

*Chứng minh.* Thay $x^{(k)} = x^\ast$: tỉ số $y_i/\bar y_i = 1$, nên tổng
trong (4.2) bằng $\sigma^{(m)}_j$, và thương bằng 1. $\square$

Với dữ liệu **không** nhất quán (trường hợp thực, vì có nhiễu), các subiteration
có các điểm bất động khác nhau và OSEM đi vào **chu trình giới hạn** quanh
nghiệm ML thay vì hội tụ. Định lý 3.3 **không** còn đúng: OSEM không đơn điệu.

Cân bằng đạt được thế nào ở hai đường tính:

* **Sinogram (§5):** $\mathcal S_m = \{(p,v,u): v \equiv m \ (\mathrm{mod}\ M)\}$
  — các view *xen kẽ*. $M = 24 \mid N_v = 288$, mỗi tập con 12 view. Cân bằng
  chỉ **xấp xỉ**, dựa vào đối xứng quay của máy: $\sigma^{(m)}$ khác nhau giữa
  các $m$ đúng bằng mức bất đối xứng góc của đối tượng.
* **List-mode (§7.5):** $\mathcal S_m$ = các sự kiện có chỉ số $\equiv m$,
  còn ảnh sensitivity **không phụ thuộc sự kiện**, nên
  $\sigma^{(m)} = \kappa_m \sigma$ với $\kappa_m = |\mathcal S_m|/N_E$ —
  cân bằng **chính xác**, không cần đối xứng nào.

### 4.3 Dừng sớm là chính quy hoá

$L$ lõm nhưng bài toán ML là **ill-posed**: khi $k\to\infty$, MLEM/OSEM hội tụ
về nghiệm ML của *một* mẫu nhiễu, và phương sai voxel tăng không giới hạn
(nhiễu "muối tiêu" ở vùng độ nhạy thấp). Ba biện pháp, dùng theo thứ tự này:

1. **Dừng sớm** ở $n_{\mathrm{it}}\cdot M$ subiteration (bù trừ bias–variance).
2. **Lọc hậu kỳ** (§9.3): tuyến tính, tách khỏi vòng lặp, nên không phá vỡ
   Mệnh đề 3.4 trong lúc lặp.
3. **Prior** khi cần (§7.6): thay OSEM bằng BSREM với thế
   Relative Difference.

Với $n_{\mathrm{it}} = 2, M = 24$, mức "đã lặp" tương đương $\approx 48$ vòng
MLEM về tốc độ hội tụ, nhưng **không** về nhiễu — đây là lý do bộ lọc hậu kỳ
không phải trang trí mà là nửa còn lại của thuật toán.

---

## 5. Thuật toán A — OSEM non-TOF trên sinogram

### 5.1 Bài toán

Cho một bed:

* dữ liệu $y \in \mathbb N^{N_b}$, $N_b = 60\,679\,584$ bin (không TOF, $N_T=1$);
* $S = \eta\,\delta\,\alpha \in \mathbb R_+^{N_b}$ — tích ba sensitivity, trong
  đó $\eta\delta$ (norm × dead time) đến từ hiệu chuẩn máy và $\alpha$ từ CT
  qua (2.2)–(2.3);
* $b = r + s \in \mathbb R_+^{N_b}$;
* ẩn $x \in \mathbb R_+^{47\times337\times337}$.

Giải (3.2) với $A = \mathrm{diag}(S)\,G$ bằng (4.2).

### 5.2 Ảnh sensitivity

$$
\sigma^{(m)} = G^{\mathsf T}\!\big[\,\mathbf 1_{\mathcal S_m}\odot S\,\big],
\qquad
\sigma = \sum_{m=1}^{M}\sigma^{(m)} = G^{\mathsf T} S .
\tag{5.1}
$$

Điểm cốt tử về **thứ tự**: $\sigma$ phải được tính **với $S$ đã gắn vào**.
Nếu tính $\sigma$ từ $G^{\mathsf T}\mathbf 1$ rồi mới nhân $S$ vào tử số, phép
cập nhật (4.2) trở thành

$$
x_j \frac{\sum_i S_i G_{ij}\,y_i/\bar y_i}{\sum_i G_{ij}}
$$

— vẫn là một phép lặp nhân hợp lệ nhưng **không phải** MLEM của mô hình (2.6):
nó chỉ đánh trọng số lại, và nghiệm bất động của nó không thoả (3.4). Sai lệch
này không sinh lỗi, chỉ sinh một ảnh sai *một cách hợp lý*.

### 5.3 Giá đỡ của ảnh (FOV tròn trong lưới vuông)

Định nghĩa giá đỡ

$$
\Omega = \Big\{ j : \ \rho_j \le \rho_{\max} \Big\},
\qquad \rho_j = \sqrt{x_j^2+y_j^2},\quad \rho_{\max} = 355.8\ \text{mm} \ \text{(2.9)} .
\tag{5.2}
$$

Với $j \notin \Omega$ ta có $G_{ij} = 0\ \forall i$, do đó $\sigma_j = 0$ và
(4.2) là $0/0$. Về mặt số học thương này không phải "nhỏ" mà là **vô nghĩa**.

**Cách xử lý đúng theo toán:** dùng Mệnh đề 3.5 — đặt

$$
x^{(0)}_j = \mathbf 1[j\in\Omega]
\tag{5.3}
$$

thì $x^{(k)}_j = 0$ cho mọi $k$ và mọi $j\notin\Omega$, và theo Mệnh đề 3.4
toàn bộ số đếm ở lại trong $\Omega$. Cắt *sau* khi lặp thì đã mất count (đo
trên dữ liệu thật: $34\%$ số đếm của một bed rơi ra ngoài $\rho_{\max}$ khi
không áp giá đỡ, với đỉnh giả cao gấp $52$ lần đỉnh thật).

### 5.4 Thuật toán

$$
\begin{array}{ll}
\textbf{Vào:} & y,\ S = \eta\delta\alpha,\ b = r+s,\ M,\ n_{\mathrm{it}} \\
1: & x \leftarrow \mathbf 1_\Omega \\
2: & \sigma^{(m)} \leftarrow G^{\mathsf T}[\mathbf 1_{\mathcal S_m}\odot S], \quad m=1..M \\
3: & \textbf{for } n = 1..n_{\mathrm{it}} \textbf{ for } m = 1..M: \\
4: & \qquad \bar y \leftarrow S \odot (Gx) + b \\
5: & \qquad x \leftarrow \dfrac{x}{\sigma^{(m)}} \odot
      G^{\mathsf T}\!\Big[\mathbf 1_{\mathcal S_m}\odot S \odot \dfrac{y}{\bar y}\Big] \\
6: & \textbf{Ra: } x \ (\text{count/voxel}),\ \sigma \ (\text{dùng ở §9.2})
\end{array}
$$

Bước 4–5 chỉ thực hiện trên các bin thuộc $\mathcal S_m$.

### 5.5 Chi phí

Gọi $C_{\mathrm{fp}}$ là chi phí một lần chiếu thuận toàn bộ sinogram. Nếu
projector hỗ trợ chiếu **đúng tập con**, chi phí một iteration là
$2C_{\mathrm{fp}}$ (một thuận + một ngược, chia đều cho $M$ tập con), và tổng

$$
C_{\mathrm{sino}} \;=\; 2\,n_{\mathrm{it}}\,C_{\mathrm{fp}} .
\tag{5.4}
$$

Nếu projector **luôn chiếu toàn bộ sinogram** cho mỗi tập con (trường hợp thực
tế của backend matrix-free dùng ở đây), chi phí thành

$$
C_{\mathrm{sino}} \;=\; 2\,n_{\mathrm{it}}\,M\,C_{\mathrm{fp}},
\tag{5.5}
$$

tức **tăng tuyến tính theo số tập con** — chia tập con không còn mua được gì
về thời gian, chỉ mua tốc độ hội tụ. Và $C_{\mathrm{fp}} \propto N_b \cdot N_T$:
mở TOF trên sinogram **nhân** chi phí lên $N_T$ lần ($60.7$ M bin $\to$ $303$ M
bin ở $N_T = 5$). Đây là lý do toán học vì sao đường sinogram chạy không TOF,
và đường TOF phải là list-mode (§7.7 cho công thức đối chiếu).

---

## 6. Thời gian bay (TOF)

### 6.1 Từ chênh lệch thời gian đến vị trí

Gọi $\ell$ là toạ độ có dấu dọc LOR, gốc ở trung điểm, chiều dương hướng về
detector 1. Nếu điểm huỷ cặp ở $\ell$, photon tới detector 1 sớm hơn:

$$
\Delta t = t_1 - t_2 = -\frac{2\ell}{c}
\quad\Longleftrightarrow\quad
\ell = -\frac{c\,\Delta t}{2}.
\tag{6.1}
$$

Dấu trừ trong (6.1) là toàn bộ nội dung của phép đảo trục TOF giữa quy ước của
máy và quy ước "vị trí có dấu dọc LOR": chỉ số TOF thô $g$ của máy đổi thành
chỉ số $t$ theo

$$
t \;=\; \Big\lfloor \frac{(N^{\mathrm{raw}}_T - 1) - (g + \lfloor N^{\mathrm{raw}}_T/2\rfloor)}{\ \varphi\ } \Big\rfloor,
\qquad \varphi = \frac{N^{\mathrm{raw}}_T}{N_T},
\tag{6.2}
$$

với $N_T^{\mathrm{raw}} = 55$, $g\in[-27,+27]$, và $\varphi$ là hệ số gộp
(mash). Phép đảo và phép gộp **giao hoán** vì $\varphi \mid N_T^{\mathrm{raw}}$.

Vị trí trung tâm của bin $t$, dưới dạng có dấu:

$$
\ell_t = \Big(t - \frac{N_T-1}{2}\Big)\,\Delta_{\mathrm{TOF}} + o,
\qquad
\Delta_{\mathrm{TOF}} = \frac{c\,\tau_{\mathrm{LSB}}}{2}\varphi,
\tag{6.3}
$$

$o = 0$ khi $N_T$ lẻ, $o = -\Delta_{\mathrm{TOF}}/2$ khi $N_T$ chẵn. Toàn dải

$$
D_{\mathrm{TOF}} = N_T^{\mathrm{raw}}\frac{c\,\tau_{\mathrm{LSB}}}{2} = 735.8\ \text{mm}
\quad(\text{không đổi theo mash}),
\qquad \Delta_{\mathrm{TOF}}\big|_{N_T=55} = 13.38\ \text{mm}.
$$

Độ phân giải thời gian $\tau_{\mathrm{FWHM}} = 675$ ps thành độ phân giải
không gian

$$
\mathrm{FWHM}_\ell = \frac{c\,\tau_{\mathrm{FWHM}}}{2} = 101.2\ \text{mm},
\qquad
\varsigma = \frac{\mathrm{FWHM}_\ell}{2\sqrt{2\ln 2}} = 42.97\ \text{mm}.
\tag{6.4}
$$

### 6.2 Nhân TOF và phân hoạch đơn vị

Xác suất một cặp phát tại $\ell$ rơi vào bin thời gian $t$ là tích phân của
nhân Gauss trên bề rộng bin:

$$
h_t(\ell) \;=\; \int_{\ell_t - \Delta_{\mathrm{TOF}}/2}^{\ell_t + \Delta_{\mathrm{TOF}}/2}
\frac{1}{\sqrt{2\pi}\varsigma}\exp\!\Big(-\frac{(\xi-\ell)^2}{2\varsigma^2}\Big)\mathrm d\xi
\;=\;
\frac12\left[
\mathrm{erf}\!\Big(\frac{\ell_t^{+}-\ell}{\varsigma\sqrt2}\Big) -
\mathrm{erf}\!\Big(\frac{\ell_t^{-}-\ell}{\varsigma\sqrt2}\Big)\right].
\tag{6.5}
$$

> **Định lý 6.1 (phân hoạch đơn vị).** Với các bin liền kề phủ kín
> $[-D_{\mathrm{TOF}}/2, +D_{\mathrm{TOF}}/2]$,
> $$
> \sum_{t=0}^{N_T-1} h_t(\ell) =
> \frac12\left[\mathrm{erf}\Big(\frac{D_{\mathrm{TOF}}/2-\ell}{\varsigma\sqrt2}\Big)
> - \mathrm{erf}\Big(\frac{-D_{\mathrm{TOF}}/2-\ell}{\varsigma\sqrt2}\Big)\right]
> \;=\; 1 - O\!\left(e^{-\left(\frac{D_{\mathrm{TOF}}/2}{\varsigma\sqrt2}\right)^2}\right).
> $$

*Chứng minh.* Các cận trong (6.5) khớp nhau ($\ell_t^{+} = \ell_{t+1}^{-}$)
nên tổng **thu gọn kiểu telescoping**, chỉ còn hai cận ngoài cùng. Ở đây
$D_{\mathrm{TOF}}/2 = 367.9$ mm $= 8.56\,\varsigma$, nên với $|\ell|$ trong
FOV, $\mathrm{erf}(\pm 6.05) = \pm(1 - 10^{-17})$: tổng bằng $1$ trong độ chính
xác số thực. $\square$

*Chú ý:* khi nhân bị **cắt** ở $\pm n_\varsigma \varsigma$ ($n_\varsigma = 3$
trong đường list-mode) để tiết kiệm tính toán, tổng còn $\ge 0.9973$ thay vì 1;
đó là xấp xỉ duy nhất mà Định lý 6.1 phải nhượng bộ.

### 6.3 Ma trận hệ thống có TOF

$$
(G_t)_{lj} \;=\; \int_{\mathcal L_l} \chi_j(\mathbf z)\,h_t(\ell(\mathbf z))\,\mathrm d\ell .
\tag{6.6}
$$

> **Hệ quả 6.2.** $\displaystyle\sum_{t=0}^{N_T-1} G_t = G$ (chính xác, theo
> Định lý 6.1).

Ba hệ quả *vật lý–toán học* rút ra ngay, và cả ba đều là **đẳng thức**, không
phải xấp xỉ:

1. **$S$ được lặp lại y nguyên trên mọi bin TOF.** $\eta$ (hiệu suất cặp tinh
   thể), $\delta$ (phân suất sống của khối) và $\alpha$ (xác suất sống sót của
   *cặp* photon dọc LOR, xem (2.2)) đều không phụ thuộc thời điểm đến. Do đó
   $$
   \sum_t S\odot(G_t x) = S \odot (Gx),
   $$
   tức mô hình TOF **thu về đúng** mô hình không TOF khi cộng dồn trục thời
   gian. Nếu $S$ bị chia cho $N_T$ (hoặc bị nhân thêm một hàm của $t$), đẳng
   thức này gãy và ảnh sai một hằng số.
2. **Randoms trải đều theo TOF:** $r_{l,t} = r_l/N_T$. Lý do: trùng phùng ngẫu
   nhiên là hai singles **không tương quan thời gian**, nên phân bố của
   $\Delta t$ là đều trên cửa sổ trùng phùng; và $N_T^{\mathrm{raw}}
   \tau_{\mathrm{LSB}} = 55\times89.2459 = 4.909$ ns **đúng bằng** cửa sổ đó.
3. **Scatter thì không đều:** $s_{l,t} = s_l\,w_t(l)$ với
   $\sum_t w_t(l) = 1$. Chuẩn hoá này bảo toàn tổng:
   $$
   \sum_t b_{l,t} = \sum_t\Big(\frac{r_l}{N_T} + s_l w_t(l)\Big) = r_l + s_l = b_l .
   \tag{6.7}
   $$
   Trải scatter **đều** theo TOF là sai định lượng chứ không chỉ về hình dạng:
   trên dữ liệu thật, phân bố đều làm $69\%$ số bin TOF có tốc độ true âm
   ($y_{l,t} - b_{l,t} < 0$), còn phân bố đúng làm $0\%$.

### 6.4 Chiều của LOR và phép đối xứng chỉ số

$\ell$ có **dấu**, nên nó chỉ có nghĩa khi LOR đã được định hướng. Với một bin
sinogram, chiều chuẩn là $(d_1,d_2)$ của (2.7); một sự kiện ghi theo thứ tự
ngược lại phải đổi chỉ số

$$
t \;\longmapsto\; (N_T - 1) - t .
\tag{6.8}
$$

Với list-mode, cặp tinh thể được đưa vào **theo đúng thứ tự ghi được**, nên
chiều LOR là chiều của chính sự kiện và chỉ cần **một dấu toàn cục**
$\epsilon\in\{\pm1\}$ cho cả tập dữ liệu. Hai khung quy chiếu này không thể suy
ra nhau, và chọn nhầm sẽ đặt hoạt độ sang **nửa sai** của mọi LOR — trong khi
mọi bất biến đếm được (tổng count, kích thước dữ liệu, $\sum y \ge \sum r$)
đều **không đổi**.

### 6.5 TOF mua được gì (ước lượng phương sai cổ điển)

Không TOF, một sự kiện chỉ nói "ở đâu đó trên dây cung dài $D$"; có TOF, nó
nói "ở đâu đó trong đoạn $\mathrm{FWHM}_\ell$". Theo lập luận Budinger, độ lợi
phương sai xấp xỉ

$$
g_{\mathrm{TOF}} \;\approx\; \frac{D}{\mathrm{FWHM}_\ell}
\;=\; \frac{2D}{c\,\tau_{\mathrm{FWHM}}} ,
\tag{6.9}
$$

với $D$ là đường kính đối tượng. Với $D = 300$ mm và $\mathrm{FWHM}_\ell =
101.2$ mm: $g_{\mathrm{TOF}} \approx 3$ — tương đương tăng gấp ba số đếm hiệu
dụng ở cùng liều.

---

## 7. Thuật toán B — LM-TOF-OSEM

### 7.1 Hàm hợp lý dạng list-mode

Không gian dữ liệu bây giờ là tập ô $(l,t)$, $l$ chạy trên **mọi LOR hợp lệ**
($N_L = 63\,203\,328$) và $t$ trên $N_T = 55$ bin thời gian. Mô hình (2.6) viết
lại ở mức **từng LOR**:

$$
\bar y_{l,t} \;=\; w_l\,(G_t x)_l \;+\; b_{l,t},
\qquad w_l = \eta_l\delta_l\alpha_l .
\tag{7.1}
$$

> **Định lý 7.1 (log-likelihood list-mode).** Cho $N_E$ sự kiện, sự kiện $e$
> rơi vào ô $(l_e,t_e)$. Khi đó, bỏ các hằng số không phụ thuộc $x$,
> $$
> \boxed{\;
> L(x) \;=\; \sum_{e=1}^{N_E}\log\Big[(G_{t_e}x)_{l_e} + a_{l_e,t_e}\Big]
> \;-\; \big\langle x,\; G^{\mathsf T} w\big\rangle \;}
> \qquad a_{l,t} \equiv \frac{b_{l,t}}{w_l}.
> \tag{7.2}
> $$

*Chứng minh.* Từ (3.1) áp trên các ô:
$L = \sum_{l,t}\big[y_{l,t}\log\bar y_{l,t} - \bar y_{l,t}\big] + \text{const}$.

*Số hạng thứ nhất.* $y_{l,t}$ là số sự kiện rơi vào ô đó, nên
$\sum_{l,t} y_{l,t}\log\bar y_{l,t} = \sum_e \log \bar y_{l_e,t_e}$
(mỗi sự kiện đóng góp đúng một lần; ô rỗng đóng góp $0$). Tách
$$
\log \bar y_{l,t} = \log\Big(w_l\big[(G_tx)_l + a_{l,t}\big]\Big)
= \underbrace{\log w_l}_{\text{hằng số theo } x} + \log\big[(G_tx)_l + a_{l,t}\big].
$$

*Số hạng thứ hai.* Cộng trên **toàn bộ** ô, kể cả ô không có sự kiện:
$$
\sum_{l,t}\bar y_{l,t}
= \sum_l w_l \sum_t (G_t x)_l + \underbrace{\sum_{l,t} b_{l,t}}_{\text{hằng số}}
\overset{\text{HQ 6.2}}{=} \sum_l w_l (Gx)_l + \text{const}
= \langle x, G^{\mathsf T}w\rangle + \text{const}. \qquad\square
$$

Ba điều đọc ra từ (7.2), và cả ba đều là **hệ quả**, không phải lựa chọn:

* **Trọng số $w$ không chia hình chiếu thuận.** Nó xuất hiện đúng hai chỗ: (i)
  như một hằng số cộng trong log (bị loại), (ii) trong ảnh sensitivity. Muốn
  nền cộng vào đúng chỗ, nó phải tới **đã chia sẵn** cho $w$:
  $a = b/w$. Kiểm tra ngược: $w\cdot a = b$, đúng bằng nền của LOR đó.
* **Ảnh sensitivity phải lấy trên MỌI LOR**, không chỉ các LOR có sự kiện —
  số hạng thứ hai của (7.2) là tổng trên toàn bộ ô. Đây là điểm khác biệt bản
  chất giữa list-mode và "sinogram thưa".
* **Ảnh sensitivity KHÔNG có TOF.** Xem §7.3.

### 7.2 Từ bin sang LOR: chia bội số span-2

Các đại lượng đo/ước lượng nằm ở mức **bin**; một sự kiện là **một LOR**. Theo
Mệnh đề 2.2 với giả thiết chia đều giữa $\mathfrak m_i$ cặp ring:

$$
w_l = \frac{(\eta\delta)_{i(l)}}{\mathfrak m_{i(l)}}\cdot \alpha_{i(l)},
\qquad
r_l = \frac{r_{i(l)}}{\mathfrak m_{i(l)}},
\qquad
s_l = \frac{s_{i(l)}}{\mathfrak m_{i(l)}} .
\tag{7.3}
$$

$\alpha$ **không** chia: nó là xác suất sống sót của **một** LOR (2.2), không
phải một tổng số đếm hay một sensitivity của bin. Đây là phép nghịch đảo chính
xác của Mệnh đề 2.2 — ở đường sinogram thì cấm nhân thêm $\mathfrak m$, ở
đường list-mode thì bắt buộc chia. Giả thiết duy nhất được thêm: hai cặp ring
của một bin lẻ segment 0 có **cùng** sensitivity (không có thông tin nào phân
biệt chúng).

Kết hợp (7.3) với (6.7):

$$
a_{l,t} \;=\; \frac{1}{w_l}\Big(\frac{r_l}{N_T} + s_l\,w_t(l)\Big).
\tag{7.4}
$$

### 7.3 Định lý về ảnh sensitivity

> **Định lý 7.2.** Trong bài toán list-mode có TOF, ảnh sensitivity
> $$
> \sigma \;=\; \nabla_x \Big[\textstyle\sum_{l,t}\bar y_{l,t}\Big] \;=\; G^{\mathsf T} w
> $$
> **không** phụ thuộc TOF: nó là back-projection **không TOF** của trọng số
> LOR, lấy trên toàn bộ $N_L$ LOR hợp lệ.

*Chứng minh.* Trực tiếp từ Định lý 7.1, số hạng thứ hai. Bản chất là Hệ quả
6.2: $\sum_t G_t = G$. $\square$

Đây là kết quả làm hai thuật toán gặp nhau: $\sigma$ của thuật toán B là
**cùng một đại lượng toán học** với $\sigma$ của thuật toán A (5.1), chỉ khác
lưới lấy tổng (LOR thay vì bin) — và do (2.10) hai tổng đó bằng nhau:

$$
\sum_{l} w_l G_{lj} \;=\; \sum_i \mathfrak m_i \cdot \frac{S_i}{\mathfrak m_i} G_{ij} \;=\; \sum_i S_i G_{ij}.
$$

Về mặt tính toán, Định lý 7.2 nói rằng chi phí ảnh sensitivity **không** nhân
lên theo $N_T$ — một back-projection không TOF trên $63.2$ M LOR, một lần duy
nhất cho cả lần chạy.

### 7.4 Gradient và bước cập nhật

Từ (7.2), với $H_t \equiv G_t$ ký hiệu projector hình học có TOF:

$$
\nabla L(x) \;=\; \sum_{e}\frac{(G_{t_e})^{\mathsf T}_{\;\cdot\,l_e}}{(G_{t_e}x)_{l_e} + a_{e}} \;-\; \sigma
\;=\; \mathcal H^{\mathsf T}\!\left[\frac{\mathbf 1}{\mathcal H x + a}\right] - \sigma ,
\tag{7.5}
$$

trong đó $\mathcal H$ là toán tử "chiếu lên đúng $N_E$ ô có sự kiện" và
$\mathbf 1$ là vector $N_E$ số 1 — dạng list-mode của $y/\bar y$ với $y\equiv1$
trên các ô đã ghi.

Áp phép **tiền điều kiện nhân** $C(x) = \mathrm{diag}(x)/\sigma$ (đúng bộ
tiền điều kiện biến gradient ascent thành EM, so sánh (3.7')):

$$
x \;\leftarrow\; x + \frac{x}{\sigma}\odot\Big(\mathcal H^{\mathsf T}\Big[\frac{\mathbf 1}{\mathcal H x + a}\Big] - \sigma\Big)
\;=\; \frac{x}{\sigma}\odot \mathcal H^{\mathsf T}\!\left[\frac{\mathbf 1}{\mathcal H x + a}\right].
\tag{7.6}
$$

Số hạng $-\sigma$ triệt tiêu với $+x$: **cập nhật gradient có tiền điều kiện
$\equiv$ cập nhật nhân của EM**. Đó là lý do (7.6) và (3.7) là *cùng một thuật
toán* viết ở hai dạng.

### 7.5 Tập con list-mode

Chia sự kiện xen kẽ: $\mathcal S_m = \{e : e \equiv m \ (\mathrm{mod}\ M)\}$,
$\kappa_m = |\mathcal S_m|/N_E$. Số hạng thứ hai của (7.2) là **tổng trên toàn
bộ ô**, không phụ thuộc sự kiện, nên khi giới hạn vào tập con nó phải được
**co lại theo tỉ lệ** $\kappa_m$:

$$
\boxed{\;
x \;\leftarrow\; \frac{x}{\kappa_m\,\sigma}\odot
\mathcal H_m^{\mathsf T}\!\left[\frac{\mathbf 1}{\mathcal H_m x + a_m}\right]\;}
\tag{7.7}
$$

> **Mệnh đề 7.3.** Các tập con của (7.7) **cân bằng chính xác** theo nghĩa
> Mệnh đề 4.1: $\sigma^{(m)} = \kappa_m \sigma$ với $\sum_m \kappa_m = 1$.

*Chứng minh.* Hiển nhiên từ định nghĩa; $\sigma$ không phụ thuộc tập sự kiện,
còn $\kappa_m$ là phân suất sự kiện. $\square$

Đối lập với §4.2: ở đường sinogram, cân bằng phụ thuộc đối xứng góc của đối
tượng; ở đây nó là một đồng nhất thức.

### 7.6 Biến thể chính quy hoá (BSREM)

Khi cần khống chế nhiễu bằng prior thay vì bằng dừng sớm, thay (7.7) bằng

$$
x \;\leftarrow\; x + \frac{\vartheta(n)\,x}{\kappa_m\,\sigma}\odot
\Big(\nabla L_m(x) - \beta\,\nabla V(x)\Big),
\tag{7.8}
$$

với $\vartheta(n)$ là dãy nới lỏng ($\vartheta\equiv1$ mặc định) và $V$ là thế
**Relative Difference**:

$$
V(x) = \sum_{j}\sum_{j'\in\mathcal N_j} \omega_{jj'}\,
\frac{(x_j-x_{j'})^2}{x_j + x_{j'} + \gamma\,|x_j - x_{j'}|},
\qquad \gamma = 1,
\tag{7.9}
$$

$\omega_{jj'}$ là nghịch đảo khoảng cách Euclid giữa hai voxel lân cận. Tính
chất của (7.9): với $|x_j - x_{j'}|$ nhỏ nó xử sự như thế bậc hai (làm trơn),
với hiệu lớn mẫu số tăng tuyến tính nên **hình phạt bão hoà** — biên được giữ.
Đạo hàm:

$$
\frac{\partial V}{\partial x_j} = \sum_{j'\in\mathcal N_j}\omega_{jj'}\,
\frac{(x_j - x_{j'})\big(\gamma|x_j - x_{j'}| + x_j + 3x_{j'}\big)}
{\big(x_j + x_{j'} + \gamma|x_j-x_{j'}|\big)^2}.
\tag{7.10}
$$

$\beta = 0$ đưa (7.8) về đúng (7.7).

### 7.7 Thuật toán

$$
\begin{array}{ll}
\textbf{Vào:} & \{(c_1,c_2,g)_e\}_{e=1}^{N_E},\ \eta\delta,\ \alpha,\ r,\ s,\ M,\ n_{\mathrm{it}},\ N_T,\ \epsilon \\
1: & \text{ánh xạ mỗi sự kiện: } (c_1,c_2)\mapsto l_e, \ g\mapsto t_e \text{ theo } (6.2),\ \epsilon \\
2: & w_l,\ a_{l,t} \leftarrow (7.3),(7.4) \\
3: & \sigma \leftarrow G^{\mathsf T} w \quad\text{trên toàn bộ } N_L \text{ LOR (không TOF, Định lý 7.2)} \\
4: & x \leftarrow \mathbf 1_{\Omega\cap\Omega_z} \\
5: & \textbf{for } n=1..n_{\mathrm{it}} \textbf{ for } m=1..M: \\
6: & \qquad p \leftarrow \mathcal H_m x + a_m \quad (\text{chiếu Joseph có TOF, cắt } \pm3\varsigma) \\
7: & \qquad x \leftarrow \dfrac{x}{\kappa_m \sigma}\odot \mathcal H_m^{\mathsf T}\!\big[\mathbf 1/p\big] \\
8: & \textbf{Ra: } x,\ \sigma
\end{array}
$$

**Giá đỡ** ở bước 4 gồm hai phần: đĩa ngang $\Omega$ của (5.2), và dải trục

$$
\Omega_z = \Big\{ j : \ |z_j| \le \max_\rho |z_\rho| + \tfrac{\Delta_z}{2}\Big\},
\tag{7.11}
$$

tức mặt phẳng ảnh nằm trong FOV khi **tâm** của nó cách vòng ngoài cùng không
quá nửa mặt phẳng (làm tròn, không phải cắt trên/dưới). Lưới ở đây đặt **đúng**
lên dải vòng — 47 mặt phẳng $\Delta_z$ so với 24 vòng $2\Delta_z$ — nên
$z_{\max}$ rơi *chính xác* lên biên mặt phẳng 46, và một phép cắt kiểu
$\lfloor\cdot\rfloor$ sẽ xoá hẳn mặt phẳng đó khỏi giá đỡ; theo Mệnh đề 3.5 nó
sẽ bằng 0 vĩnh viễn, và theo Mệnh đề 3.4 số đếm của nó bị dồn sang mặt phẳng
bên cạnh.

**Chi phí.** Mỗi sự kiện được chiếu một lần mỗi iteration, và với TOF tia bị
cắt còn $\pm3\varsigma$ quanh $\ell_{t_e}$ thay vì cả dây cung:

$$
C_{\mathrm{LM}} \;\approx\;
\underbrace{N_L\,\frac{\bar D}{\Delta_x}}_{\text{sensitivity, 1 lần}}
\;+\;
2\,n_{\mathrm{it}}\,N_E\,\frac{\min(6\varsigma,\ \bar D)}{\Delta_x},
\tag{7.12}
$$

với $\bar D$ là chiều dài dây cung trung bình. So với (5.5): chi phí list-mode
**tỉ lệ với số sự kiện thật sự đếm được** ($10^7$–$10^8$) chứ không với số ô
của không gian dữ liệu ($N_L N_T \approx 3.5\times10^9$), và TOF **làm giảm**
chi phí (tia ngắn hơn) thay vì nhân nó lên.

---

## 8. Hai thuật toán, đối chiếu toán học

| | A — sinogram non-TOF | B — list-mode TOF |
|---|---|---|
| không gian dữ liệu | $N_b = 6.07\times10^7$ bin | $N_E$ sự kiện trong $N_L N_T = 3.5\times10^9$ ô |
| mô hình | $\bar y = S\odot(Gx)+b$ | $\bar y_{l,t} = w_l(G_tx)_l + b_{l,t}$ |
| hàm mục tiêu | $\sum_i [y_i\log\bar y_i - \bar y_i]$ | $\sum_e \log[(G_{t_e}x)_{l_e}+a_e] - \langle x,\sigma\rangle$ |
| ảnh sensitivity | $G^{\mathsf T}S$ | $G^{\mathsf T}w$ — **bằng nhau** (§7.3) |
| bội số span-2 | mang sẵn trong $S,y,b$ (M.đ. 2.2) | chia ra: $w = S/\mathfrak m$ (§7.2) |
| tập con | view xen kẽ, cân bằng **xấp xỉ** | sự kiện xen kẽ, cân bằng **chính xác** |
| nền cộng | $b$, ngoài $S$ | $a = b/w$, trong ngoặc log |
| cập nhật | (4.2) | (7.7) |
| chi phí | (5.5): $\propto n_{\mathrm{it}} M N_b N_T$ | (7.12): $\propto n_{\mathrm{it}} N_E$ |
| TOF | **nhân** chi phí lên $N_T$ | **giảm** chi phí (tia ngắn) |
| phương sai | tham chiếu | $\approx / g_{\mathrm{TOF}}$, (6.9) |

Cả hai cho ra cùng một đại lượng vật lý: $x$ tính bằng **count/voxel**, chưa
hiệu chuẩn tuyệt đối. Do $G$ tích luỹ theo bước voxel và hai đường dùng hai
projector khác nhau, hằng số hiệu chuẩn $K$ ở §9.4 **phải đo riêng cho từng
đường** (và cho từng kích thước voxel).

---

## 9. Hậu xử lý định lượng

### 9.1 Hiệu chỉnh phân rã

Hoạt độ suy giảm theo $A(t) = A_0 e^{-\lambda(t-t_{\mathrm{inj}})}$,
$\lambda = \ln 2/T_{1/2}$. Một bed bắt đầu ở $t_n$, kéo dài $T_n$, nên số đếm
của nó tỉ lệ với

$$
\int_{t_n}^{t_n+T_n} A_0 e^{-\lambda(t - t_{\mathrm{inj}})}\,\mathrm dt
= A_0\,e^{-\lambda \Delta_n}\,\frac{1 - e^{-\lambda T_n}}{\lambda},
\qquad \Delta_n = t_n - t_{\mathrm{inj}} .
\tag{9.1}
$$

Ảnh tái tạo tỉ lệ với **hoạt độ trung bình trong khung**, tức (9.1) chia $T_n$.
Hệ số đưa bed $n$ về thời điểm tiêm:

$$
\boxed{\;f_n = \Big[e^{-\lambda\Delta_n}\,\frac{1-e^{-\lambda T_n}}{\lambda T_n}\Big]^{-1}\;}
\tag{9.2}
$$

Bỏ thừa số thứ hai (dùng hoạt độ tức thời tại đầu khung) sai $\sim0.5\%$ ở đây
— nhưng sai **khác nhau theo từng bed**, nên nó không tan vào hằng số $K$ mà
sống sót thành một **gradient dọc trục** trong ảnh toàn thân.

Nếu muốn quy chiếu về **đầu chuỗi chụp** $t_0$ (quy ước của nhà sản xuất) thay
vì về thời điểm tiêm, nhân thêm

$$
f_{\mathrm{start}} = e^{-\lambda(t_0 - t_{\mathrm{inj}})} ,
\tag{9.3}
$$

thừa số $\frac{1-e^{-\lambda T}}{\lambda T}$ **triệt tiêu** vì cả hai quy ước
cùng lấy trung bình trên khung. Trên các ca ở đây $f_{\mathrm{start}}^{-1} \in
[1.41, 1.65]$ — không phải chi tiết nhỏ.

### 9.2 Ghép bed: trọng số nghịch phương sai

Bàn dịch $124.26$ mm giữa hai bed liên tiếp $=$ đúng $38$ mặt phẳng, trong khi
một bed dài $47$ mặt phẳng $\Rightarrow$ chồng lấn $9$ mặt phẳng, và chúng là
mặt phẳng $38..46$ của bed dưới gặp $0..8$ của bed trên: **hai đầu yếu nhất
gặp nhau**.

> **Mệnh đề 9.1.** Nếu $\hat x^{(1)}_j, \hat x^{(2)}_j$ là hai ước lượng độc
> lập, không chệch của cùng voxel $j$ với $\mathrm{Var}(\hat x^{(n)}_j)
> \approx x_j/\sigma^{(n)}_j$, thì tổ hợp tuyến tính không chệch có phương sai
> nhỏ nhất là
> $$
> \hat x_j = \frac{\sum_n \sigma^{(n)}_j \hat x^{(n)}_j}{\sum_n \sigma^{(n)}_j}.
> $$

*Chứng minh.* Với $\hat x = \sum_n c_n \hat x^{(n)}$, $\sum_n c_n = 1$, ta có
$\mathrm{Var} = \sum_n c_n^2 V_n$ ($V_n$ = phương sai thành phần). Nhân tử
Lagrange: $2c_nV_n = \text{const} \Rightarrow c_n \propto 1/V_n \propto
\sigma^{(n)}_j$. $\square$

Xấp xỉ $\mathrm{Var}(\hat x_j)\approx x_j/\sigma_j$ đến từ thông tin Fisher
của mô hình Poisson: với voxel $j$ tách rời,

$$
\mathcal I_{jj} = \sum_i \frac{A_{ij}^2}{\bar y_i}
\;\approx\; \sum_i \frac{A_{ij}^2}{A_{ij}x_j} = \frac{\sigma_j}{x_j}
\quad\Longrightarrow\quad
\mathrm{Var} \approx \mathcal I_{jj}^{-1} = \frac{x_j}{\sigma_j}.
\tag{9.4}
$$

Vậy trọng số ghép **là chính $\sigma$**, ảnh sensitivity mà cả hai thuật toán
đã tính sẵn (5.1)/(7.2) — và nó là trọng số **theo từng voxel**, 3D, không
phải một hàm chỉ của toạ độ trục. Công thức ghép, sau khi đã hiệu chỉnh phân
rã (9.2):

$$
x^{\mathrm{WB}}_{j} \;=\;
\frac{\sum_{n} \sigma^{(n)}_{j}\, f_n\, x^{(n)}_{j}}
     {\sum_{n} \sigma^{(n)}_{j}} ,
\tag{9.5}
$$

tổng chạy trên các bed phủ voxel $j$; mặt phẳng ảnh của bed $n$ đặt tại
$z = z_n^{\mathrm{table}} + p\,\Delta_z$.

### 9.3 Bộ lọc hậu kỳ

Hai nhân tách rời, vì lưới không đẳng hướng ($\Delta_x = 2.13$ mm,
$\Delta_z = 3.27$ mm):

$$
\text{ngang: } \ \mathcal G_{\varsigma_\perp},\quad
\varsigma_\perp = \frac{6.4\ \text{mm}}{2\sqrt{2\ln2}} = 2.718\ \text{mm};
\qquad
\text{dọc trục: } \ \frac{1}{2+\varrho}\,[\,1,\ \varrho,\ 1\,],\ \varrho = 4 .
\tag{9.6}
$$

Nhân trục $[1,4,1]/6$ có phương sai
$\sum_k p_k k^2 = 2/6 = 1/3$ mặt phẳng$^2$, tức
$\varsigma_z = 0.577\,\Delta_z = 1.89$ mm, tương đương
$\mathrm{FWHM}_z = 4.45$ mm.

Điều kiện biên **khác nhau theo trục, có lý do**: ngang thì tích chập với 0
(ngoài FOV thực sự không có hoạt độ); dọc trục thì lặp mặt phẳng biên (thể
tích kết thúc vì hết bed, không phải vì hết bệnh nhân). Lọc phải chạy trên thể
tích **đã ghép**: lọc từng bed rồi mới ghép sẽ tích chập hai đầu yếu với 0,
đúng chỗ mà (9.5) tồn tại để tránh.

### 9.4 Hiệu chuẩn tuyệt đối và SUV

$$
\text{Bq/mL} \;=\; K \cdot f_{\mathrm{start}} \cdot x_{\text{count/voxel}},
\tag{9.7}
$$

$K$ có đơn vị $(\text{Bq/mL})/(\text{count/voxel})$. $K$ **không** là hằng số
phổ quát: nó chỉ đúng cho đúng chuỗi hiệu chỉnh đã đo ra nó, và tỉ lệ nghịch
với thể tích voxel (vì $G$ tích luỹ theo bước voxel, §2.2) — dùng lại $K$ đo ở
$2.1306$ mm cho lưới $1.3672$ mm sai hệ số $(2.1306/1.3672)^2 = 2.43$ theo
hướng ngang.

Chặn trên vật lý của $K$: ảnh đã quy chiếu về thời điểm tiêm nên tổng hoạt độ
trong FOV không thể vượt liều tiêm,

$$
K \;\le\; \frac{A_{\mathrm{inj}}}{V_{\mathrm{vox}}\sum_j x_j},
\tag{9.8}
$$

đẳng thức chỉ khi 100% liều nằm trong FOV (không bao giờ đúng).

SUV theo khối lượng cơ thể và theo diện tích da:

$$
\mathrm{SUV}_{\mathrm{bw}} = \frac{\text{Bq/mL}}{A_{\mathrm{inj}}/W},
\qquad
\mathrm{SUV}_{\mathrm{bsa}} = \frac{\text{Bq/mL}\cdot \mathrm{BSA}}{A_{\mathrm{inj}}},
$$
$$
\mathrm{BSA}\,[\mathrm{m^2}] = 0.007184\,W^{0.425}[\mathrm{kg}]\,H^{0.725}[\mathrm{cm}] \quad (\text{Du Bois}).
\tag{9.9}
$$

Cả hai **tuyến tính theo $K$**: mọi sai số của $K$ đi thẳng vào SUV với hệ số 1.

---

## 10. Danh mục giả thiết

Mọi chỗ mô hình rời khỏi vật lý chính xác, gom lại một chỗ:

| # | giả thiết | ảnh hưởng |
|---|---|---|
| 1 | Poisson độc lập theo bin (Mệnh đề 2.1) | dead time làm dữ liệu **dưới tán** (sub-Poisson) |
| 2 | $r, s$ là hằng số đã biết, không phải ước lượng có phương sai | phương sai ảnh bị đánh giá thấp |
| 3 | Bỏ qua acollinearity và quãng chạy positron trong $G$ | mất phân giải, một phần được $B$ (2.5) hấp thụ |
| 4 | PSF là Gauss đẳng hướng 6.4 mm | thực tế PSF giãn theo bán kính |
| 5 | Hai cặp ring của bin lẻ segment 0 có sensitivity bằng nhau (§7.2) | không có dữ liệu nào nói khác |
| 6 | Nhân TOF cắt ở $3\varsigma$ | Định lý 6.1 chỉ còn đúng đến $0.27\%$ |
| 7 | Attenuation dùng biến đổi song tuyến Carney (2.3) | bảng chuyển đổi thật của máy là hàm bậc thang 5 đoạn |
| 8 | $w_t(l)$ (hình dạng TOF của scatter) lấy từ mô hình tán xạ đơn | phân bố thật đổi theo $(v,u)$ |
| 9 | Dừng sớm thay cho chính quy hoá (khi $\beta = 0$) | nghiệm phụ thuộc số vòng lặp |
| 10 | $\Sigma_j$ ngoài $\rho_{\max}$ đặt bằng 0 | đúng theo (2.9), nhưng cần áp qua $x^{(0)}$ (M.đ. 3.5) |

---

## Phụ lục A — đối chiếu ký hiệu với các đại lượng dữ liệu

| ký hiệu | đại lượng |
|---|---|
| $y$ | prompts — số đếm thô, **chưa** trừ bất cứ thứ gì |
| $r$ | randoms |
| $s$ | scatter (mô phỏng tán xạ đơn) |
| $b = r+s$ | background |
| $\eta\delta$ | normdt (norm × dead time) — **là một sensitivity**: sửa đúng là **chia** dữ liệu cho nó |
| $\eta$ | norm_only; $\delta = (\eta\delta)/\eta$ |
| $\alpha$ | attn — hệ số suy giảm $\in(0,1]$, **không** phải ACF $=1/\alpha$ |
| $S = \eta\delta\alpha$ | sensitivity của bin |
| $w = S/\mathfrak m$ | sensitivity của một LOR |
| $\mathfrak m$ | bội số cặp ring của bin (2.10) |
| $\sigma$ | ảnh sensitivity — cũng là trọng số ghép bed (9.5) |
| $x$ | ảnh, count/voxel, quy chiếu về thời điểm tiêm |
| $K$ | hằng số hiệu chuẩn (9.7) |

Ba bất biến phải kiểm trước khi tin bất kỳ ảnh nào (gộp **theo mặt phẳng**,
vì sinogram thô chỉ có $\sim0.06$ count/bin nên so từng bin là vô nghĩa):

$$
\text{(i)}\ \ \sum_i y_i \ge \sum_i r_i,
\qquad
\text{(ii)}\ \ \sum_i s_i \le \sum_i (y_i - r_i),
\qquad
\text{(iii)}\ \ \frac{\sum_i r_i}{\#\text{delays}} \approx 0.99 .
$$

(i) và (ii) chỉ nói một điều: **tốc độ true không thể âm**. (iii) là hai đường
độc lập cùng đo một đại lượng.
