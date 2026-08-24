"""Nạp các số hạng của một bed từ Interfile, và tổng hợp chúng.

Cây `y = S·(G x) + b` đọc từ đĩa như sau:

| số hạng | file | ý nghĩa |
|---|---|---|
| `prompts` | `decoded/bed<n>.hs` | prompt thô, count |
| `randoms` | `work/bed<n>/randoms.hs` | kernel GE |
| `scatter` | `work/bed<n>/scatter.hs` | SSS của GE |
| `background` | `work/bed<n>/background.hs` | `= randoms + scatter`, chính là `b` |
| `normdt` | `work/bed<n>/normdt.hs` | norm × dead time — một **ĐỘ NHẠY** |
| `norm_only` | `work/bed<n>/norm_only.hs` | norm thuần; `deadtime = normdt/norm_only` |
| `attn` | `work/bed<n>/attn.hs` | dựng từ CT ở `utils.attn`, có cache |

`normdt` là độ nhạy chứ không phải hệ số hiệu chỉnh: **chia** dữ liệu cho nó
mới là hiệu chỉnh, và `AcquisitionSensitivityModel` nhân nó vào forward
projection. Xem docstring của `vendor/to_stir.py` cho hai phép đo chốt chiều đó.

Một bed là ~6 × 231 MB. `load()` trả cả object SIRF lẫn mảng numpy và người gọi
phải `del` khi xong — giữ cả sáu bed cùng lúc là ~9 GB.
"""

from __future__ import annotations

import json

import numpy as np

#: Số hạng ở miền count — cộng lại thì có nghĩa.
COUNT_TERMS = ("prompts", "randoms", "scatter", "background", "trues")

#: Số hạng là hệ số (không thứ nguyên) — trung bình thì có nghĩa, cộng thì không.
FACTOR_TERMS = ("norm_only", "deadtime", "normdt", "attenuation", "sensitivity")

ALL_TERMS = COUNT_TERMS + FACTOR_TERMS

#: Số plane trực tiếp (segment 0) trong sinogram 553 plane.
NSEG0 = 47

#: Cái `to_stir.py` ghi ra. `attn` do `utils.attn` thêm vào sau.
ON_DISK = ("randoms", "scatter", "background", "normdt", "norm_only")


def total(a) -> float:
    """Cộng trong float64.

    Cộng 60,7 triệu bin float32 lệch cỡ vài count — đủ để một phép so tổng với
    header trượt oan.
    """
    return float(np.asarray(a).sum(dtype=np.float64))


def load(case, n: int, af=None):
    """Mọi số hạng của bed `n`, dạng `(objs SIRF, dict mảng numpy)`.

    `af` (hệ số suy giảm, mảng numpy) là tuỳ chọn: có thì thêm `attenuation` và
    `sensitivity = normdt × attenuation`, không thì hai khoá đó vắng mặt.

    **Nhớ `del` cả hai khi xong.**
    """
    import sirf.STIR as pet

    work = case.work_bed(n)
    objs = {"prompts": pet.AcquisitionData(str(case.prompt(n)))}
    for name in ON_DISK:
        objs[name] = pet.AcquisitionData(str(work / f"{name}.hs"))

    A = {k: v.as_array() for k, v in objs.items()}
    A["trues"] = A["prompts"] - A["background"]   # âm ở từng bin là bình thường (nhiễu)
    A["deadtime"] = A["normdt"] / A["norm_only"]  # phân suất sống, < 1
    if af is not None:
        A["attenuation"] = af
        # S = norm × dead time × suy giảm. Cả ba đều là ĐỘ NHẠY nên NHÂN vào
        # nhau; `normdt` đã gộp sẵn hai thừa số đầu.
        A["sensitivity"] = A["normdt"] * af
    return objs, A


def ct_dir(case, n: int) -> str:
    """Series CT mà `d710 estimate` đã dùng cho bed này.

    Lấy từ sidecar chứ không hỏi lại: suy giảm và scatter **phải** dựng từ cùng
    một CT, và đó là CT mà kernel của GE đã thấy.
    """
    ct = meta(case, n).get("estimate", {}).get("ct")
    if not ct:
        raise SystemExit("error: sidecar của bed %d không ghi CT nào" % n)
    return ct


def bed_table(case, beds, out=print) -> None:
    """Bảng tóm tắt thu nhận, một dòng mỗi bed.

    Hai ca bệnh nhân có PHI: cố ý **không** in tên / mã BN / ngày sinh.
    """
    out(f"ca {case.name!r}: {len(beds)} bed  ->  {beds}\n")
    out(f"{'bed':>4} {'table mm':>10} {'prompts':>13} {'delays':>13} "
        f"{'giây':>5} {'kcps':>8} {'R/P':>6}")
    for n in beds:
        h = case.header(n)
        dur = h["frame_duration_ms"] / 1000
        out(f"{n:>4} {h['table_position_mm']:>10.2f} {h['prompts']:>13,} "
            f"{h['delays']:>13,} {dur:>5.0f} {h['prompts'] / dur / 1e3:>8.1f} "
            f"{h['delays'] / h['prompts']:>6.3f}")
    h0 = case.header(beds[0])
    out(f"\n{h0['dose_mbq']} MBq, {h0['patient_weight_kg']} kg, "
        f"{h0['radiopharmaceutical']}")


def collect(case, beds, af: dict, out=print):
    """Một lượt qua TẤT CẢ các bed. Trả `(proj, stats, planes)`.

    Mỗi bed nạp → rút ra thứ cần → trả RAM ngay, nên đỉnh bộ nhớ là **một** bed
    (~2,5 GB) chứ không phải sáu (~15 GB). Giữ lại lát cắt để vẽ cùng vài con số
    tổng kết; mảng đầy đủ thì bỏ — bước tái tạo sẽ nạp lại.
    """
    from . import plots

    proj, stats, planes = {}, {}, {}
    for n in beds:
        objs, A = load(case, n, af=af[n])
        planes[n] = plots.busiest_plane(A["prompts"])
        for t in ALL_TERMS:
            proj[n, t] = plots.slices(A[t][0], planes[n])
        stats[n] = {t: (total(A[t]) if t in COUNT_TERMS else float(A[t].mean()))
                    for t in ALL_TERMS}
        del A, objs
        out(f"bed {n}: xong  (plane vẽ = {planes[n]})")

    out(f"\n{'bed':>4} " + "".join(f"{t:>13}" for t in COUNT_TERMS)
        + f"{'scat.frac':>11}{'livetime':>10}")
    for n in beds:
        s = stats[n]
        sf = s["scatter"] / (s["prompts"] - s["randoms"])
        out(f"{n:>4} " + "".join(f"{s[t]:>13,.0f}" for t in COUNT_TERMS)
            + f"{sf:>11.4f}{s['deadtime']:>10.4f}")
    return proj, stats, planes


def invariant_table(case, beds, proj: dict, stats: dict, out=print) -> list:
    """In bốn bất biến cho mọi bed, trả danh sách bed có vấn đề.

    Gộp theo plane trước rồi mới so — sinogram thô dưới 1 count/bin, nên `p < r`
    đúng ở rất nhiều bin chỉ vì nhiễu Poisson.
    """
    out(f"{'bed':>4} {'Σp<Σr':>8} {'Σs>Σ(p−r)':>11} {'ΣR/delays':>11} "
        f"{'S/(T+S)':>9} {'livetime':>9} {'kcps':>8} {'bit-exact':>10}")
    bad = []
    for n in beds:
        per_plane = {t: proj[n, t]["per_plane"]
                     for t in ("prompts", "randoms", "scatter")}
        v = invariants(case, n, per_plane, stats[n])
        out(f"{n:>4} {v['frac_p_lt_r']:>7.2f}% {v['frac_s_gt_t']:>10.2f}% "
            f"{v['randoms_over_delays']:>11.5f} {v['scatter_fraction']:>9.4f} "
            f"{v['livetime']:>9.5f} {v['kcps']:>8.1f} {str(v['bit_exact']):>10}")
        if v["frac_p_lt_r"] or v["frac_s_gt_t"] or not v["bit_exact"]:
            bad.append(n)

    out("\n1. Σp ≥ Σr và 2. Σs ≤ Σ(p−r): phải là 0 % ở mọi plane — "
        "true rate âm là bất khả.")
    out("3. ΣR/delays ~0,99: randoms của GE so với delay máy đếm, hai đường độc lập.")
    out("5. livetime GIẢM khi tốc độ đếm TĂNG — đó là dấu hiệu nó là độ nhạy,")
    out("   không phải hệ số hiệu chỉnh. Nó KHÔNG phải hằng số, đừng so với một số cố định.")
    out("7. WCC: chưa áp ở đâu -> ảnh là count/voxel, CHƯA phải Bq/mL.")
    out(f"\n{'MỌI BED ĐẠT' if not bad else f'BED CÓ VẤN ĐỀ: {bad}'}")
    return bad


def meta(case, n: int) -> dict:
    """`to_stir.json` của bed `n` — gồm cả `estimate.json` lồng bên trong.

    Khoá đáng quan tâm nhất: `verified.bit_exact_vs_decoded`, do `to_stir.py`
    đặt sau khi tự chứng minh ánh xạ bin trên chính dữ liệu của bed này.
    """
    with open(case.work_bed(n) / "to_stir.json") as f:
        return json.load(f)


def summarise(case, beds, A_by_bed: dict) -> dict:
    """Một dòng số cho mỗi bed: tổng miền count, trung bình hệ số."""
    out = {}
    for n in beds:
        A = A_by_bed[n]
        out[n] = {t: (total(A[t]) if t in COUNT_TERMS else float(A[t].mean()))
                  for t in ALL_TERMS if t in A}
    return out


def invariants(case, n: int, per_plane: dict, stats: dict) -> dict:
    """Bốn bất biến phải đúng trên dữ liệu thật, gộp **theo plane**.

    Gộp theo plane chứ không theo bin: sinogram thô chạy ~0,06 count/bin, nên
    `p < r` đúng ở ~82 % số bin chỉ vì nhiễu Poisson và một khẳng định theo bin
    không nói lên điều gì.

    * `frac_p_lt_r`  — % plane có Σp < Σr. Phải là 0: true rate âm là bất khả.
    * `frac_s_gt_t`  — % plane có Σs > Σ(p−r). Phải là 0 (NEMA có ngoại lệ ở rìa
      segment, xem `tests/test_pipeline_data.py`).
    * `randoms_over_delays` — randoms của GE so với delay máy đếm, hai đường độc
      lập; ~0,99.
    * `bit_exact` — `to_stir.py` đã tự chứng minh ánh xạ bin lúc ghi.
    """
    h = case.header(n)
    P, R, S = per_plane["prompts"], per_plane["randoms"], per_plane["scatter"]
    s = stats
    return {
        "bed": n,
        "frac_p_lt_r": 100.0 * float((P < R).mean()),
        "frac_s_gt_t": 100.0 * float((S > P - R).mean()),
        "randoms_over_delays": s["randoms"] / h["delays"],
        "scatter_fraction": s["scatter"] / (s["prompts"] - s["randoms"]),
        "livetime": s["deadtime"],
        "kcps": h["prompts"] / (h["frame_duration_ms"] / 1000.0) / 1e3,
        "bit_exact": bool(meta(case, n)["verified"]["bit_exact_vs_decoded"]),
    }
