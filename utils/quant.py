"""count/voxel -> Bq/mL -> SUV. Toàn bộ phần định lượng, và hằng số `K`.

Ảnh ra khỏi `osem/` là **count/voxel đã quy về thời điểm tiêm**. Đổi sang Bq/mL
cần đúng một số vô hướng:

    Bq/mL = K · x_(count/voxel)

⚠ **`K` chỉ đúng cho ĐÚNG chuỗi hiệu chỉnh đã đo ra nó và ĐÚNG một bước
voxel.** Projector tích luỹ theo bước voxel chứ không theo thể tích, nên hằng
số đo ở 2,1306 mm mà dùng lại ở 1,3672 mm sẽ đọc cao 1,56×.

Hai mốc, không cái nào là câu trả lời cuối:

* **WCC của chính exam.** Exam khai file WCC ở header (`wcc_cal_uid`) và file đó
  có thật trong `/usr/PET/systemConfig/cal/<uid>.3dwcc`, tag `(0019,100B)` =
  `hrActivityFactor`. Nhân `1e4` là **SUY ĐOÁN** về quy ước đơn vị của GE, chưa
  dẫn ra được.
* **Chặn trên theo liều.** Ảnh đã quy về thời điểm tiêm nên Σ(hoạt độ) ≤ liều
  đã tiêm; ép bằng nhau cho một chặn trên, vì thực tế có phần liều nằm ngoài FOV.

Cả hai đều tuyến tính với SUV, nên mọi con số SUV sai đúng bằng tỉ lệ `K` sai.
"""

from __future__ import annotations

import numpy as np

#: Quy ước đơn vị GIẢ ĐỊNH giữa `hrActivityFactor` và (Bq/mL)/(count/voxel).
#: Chưa dẫn ra được; cho 68 % liều nằm trong FOV ở ca nhi, hợp lý với scan phủ
#: 77 cm của bệnh nhi cao 118 cm (thiếu chân). Đo trên NEMA mới chốt được.
WCC_UNIT_SCALE = 1e4


def dose_bq(hdr) -> float:
    """Liều thực đã vào người: liều tiêm trừ liều còn lại trong bơm, Bq."""
    return (hdr["dose_mbq"] - hdr.get("residual_dose_mbq", 0.0)) * 1e6


def voxel_ml(vox) -> float:
    """`(z, y, x)` mm -> mL."""
    return float(vox[0] * vox[1] * vox[2]) / 1000.0


def wcc_activity_factor(case, bed: int, verbose: bool = True):
    """`hrActivityFactor` của chính máy này, hoặc `None`.

    Đọc từ sidecar `estimate.json` trước — `d710 estimate` ghi sẵn lúc nó đã có
    container trong tay. Chỉ khi sidecar không có (bed dựng bằng bản cũ) mới
    hỏi lại container.
    """
    from . import container, terms

    try:
        est = terms.meta(case, bed).get("estimate", {})
    except (OSError, ValueError):
        est = {}

    if est.get("wcc_activity_factor"):
        if verbose:
            print("WCC exam khai báo: %s" % est.get("wcc_name", "?"))
        return float(est["wcc_activity_factor"])

    uid = (est.get("rdf_header") or {}).get("wcc_cal_uid")
    if not uid:
        if verbose:
            print("không có wcc_cal_uid trong sidecar của bed %d" % bed)
        return None

    got = container.cal_tags(uid, "3dwcc",
                            [("name", 0x00191006), ("factor", 0x0019100B)])
    if not got or not got.get("factor"):
        if verbose:
            print("không đọc được %s.3dwcc trong container" % uid)
        return None
    if verbose:
        print("WCC exam khai báo: %s" % got.get("name"))
    return float(got["factor"])


def k_from_wcc(factor):
    """`hrActivityFactor` -> `K`, qua quy ước đơn vị GIẢ ĐỊNH.

    `None` vào thì `None` ra, chứ không phải `TypeError`: người gọi thường viết
    `k_from_wcc(wcc_activity_factor(...))` và cái vế trong **có thể** không tra
    ra được (ca chưa có sidecar mới, hoặc header không khai `wcc_cal_uid`). Khi
    đó đường lùi là mốc theo liều, và đó là quyết định của người gọi.
    """
    return None if factor is None else float(factor) * WCC_UNIT_SCALE


def k_from_dose(vol, vox, dose: float) -> float:
    """**Chặn trên** của `K`: giả sử 100 % liều nằm trong FOV.

    Thực tế < 100 % (scan không phủ hết người), nên `K` thật nhỏ hơn số này.
    """
    total = float(np.asarray(vol).sum(dtype=np.float64))
    return dose / (total * voxel_ml(vox))


def body_mask(vol, frac: float = 0.02, pct: float = 99.9):
    """Mặt nạ thân thô: ngưỡng theo phân vị, không theo giá trị tuyệt đối.

    Ngưỡng tuyệt đối vô nghĩa ở đây vì thang chưa hiệu chuẩn (và đổi theo `K`).
    """
    v = np.asarray(vol)
    return v > frac * np.percentile(v, pct)


def suv_bw(bqml, dose: float, weight_kg: float):
    """SUV theo cân nặng. Ngầm định mô 1 g/mL."""
    return np.asarray(bqml) / (dose / (weight_kg * 1000.0))


def bsa_m2(weight_kg: float, height_m: float) -> float:
    """Du Bois: BSA(m²) = 0,007184 · W(kg)^0,425 · H(cm)^0,725."""
    return 0.007184 * weight_kg ** 0.425 * (height_m * 100) ** 0.725


def suv_bsa(bqml, dose: float, weight_kg: float, height_m: float):
    """SUV theo diện tích da. Ca nhi lệch ít hơn so với chuẩn hoá theo cân nặng.

    Không có SUVlbm: công thức Janmahasatian cần giới tính, header RDF không có.
    """
    return np.asarray(bqml) * (bsa_m2(weight_kg, height_m) * 1e4) / dose


def suv_table(bqml, mask, hdr, out=print) -> dict:
    """SUVbw và (nếu có chiều cao) SUVbsa: trung vị, p90, p99, max trong thân.

    ⚠ SUV **tuyến tính với `K`**, và `K` đang là suy đoán — mọi con số dưới đây
    sai đúng bằng tỉ lệ `K` sai. Đổi sang mốc liều là SUV nhân ngay ~1,46×.
    Chưa dùng để kết luận lâm sàng được.
    """
    dose = dose_bq(hdr)
    w = hdr["patient_weight_kg"]
    h = hdr.get("patient_height_m") or 0.0

    got = {"SUVbw": suv_bw(bqml, dose, w)}
    if h > 0:
        # Ca nhi: chuẩn hoá theo cân nặng dễ lệch hơn người lớn.
        got["SUVbsa"] = suv_bsa(bqml, dose, w, h)
    # Không có SUVlbm: công thức Janmahasatian cần giới tính, header RDF không có.

    out(f"{w} kg   {h} m   liều thực {dose / 1e6:.1f} MBq")
    out(f"mẫu số SUVbw = {dose / (w * 1000):,.1f} Bq/mL"
        + (f"   BSA = {bsa_m2(w, h):.3f} m²" if h > 0 else ""))
    out("")
    for name, s in got.items():
        v = s[mask]
        out(f"{name:7s} trung vị {np.median(v):6.3f}   "
            f"p90 {np.percentile(v, 90):6.2f}   p99 {np.percentile(v, 99):6.2f}   "
            f"max {s.max():8.1f}")
    out("\n[cơ/mỡ ~0,5–1 | gan ~1,5–2,5 | não trẻ em cao | bàng quang >20]")
    return got


def report(vol, K: float, hdr, vox, out=print) -> dict:
    """Áp `K`, in các con số phải nhìn trước khi tin, trả `{bqml, suv, ...}`."""
    dose = dose_bq(hdr)
    vml = voxel_ml(vox)
    bqml = np.asarray(vol) * K
    suv = suv_bw(bqml, dose, hdr["patient_weight_kg"])
    mask = body_mask(vol)
    in_fov = float(bqml.sum(dtype=np.float64)) * vml

    out(f">>> K đang dùng = {K:,.2f} (Bq/mL)/(count/voxel)")
    out(f"tổng trong FOV {in_fov / 1e6:6.1f} MBq = {100 * in_fov / dose:.1f} % liều"
        "     [<100 % là đúng: scan không phủ hết người]")
    out(f"Bq/mL  max {bqml.max():>12,.0f}   trung vị thân {np.median(bqml[mask]):>10,.0f}")
    out(f"SUVbw  max {suv.max():>12.1f}   trung vị thân {np.median(suv[mask]):>10.3f}"
        "     [mô mềm ~0,5–1; não/bàng quang cao hơn nhiều]")

    return {"bqml": bqml, "suv": suv, "mask": mask, "K": float(K),
            "dose_bq": dose, "voxel_ml": vml, "mbq_in_fov": in_fov / 1e6}
