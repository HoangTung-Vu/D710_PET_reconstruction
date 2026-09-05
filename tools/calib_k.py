#!/usr/bin/env python3
"""Gộp các phép đo `K` của từng ca thành MỘT hằng số cho mỗi đường tái tạo.

    python3 -m tools.calib_k                       # mọi ca có calib_*.json
    python3 -m tools.calib_k --cases a b c         # chỉ mấy ca này

Đọc `<case>/calib_sino.json` và `<case>/calib_lm.json` do
`tools.compare_vendor --json` ghi ra, in bảng từng ca rồi kết luận hai dòng để
dán vào `utils/scanner.py`.

**Vì sao một hằng số cho cả nhóm chứ không phải mỗi ca một số.** `K` là tính
chất của MÁY và của chuỗi hiệu chỉnh, không phải của bệnh nhân. Cho mỗi ca một
`K` riêng thì ca nào cũng khớp GE hoàn hảo — và phép so sánh không còn đo được
gì nữa. Cái đáng đọc ở đây là **độ tản giữa các ca**: đó chính là sai số thật
của pipeline sau khi đã bỏ đi thang tuyệt đối.

Trung vị chứ không phải trung bình: một ca lệch hình học sẽ kéo trung bình đi,
còn trung vị thì không.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.paths import out_root

#: Ca lệch quá ngần này so với trung vị thì nêu tên, không lặng lẽ gộp vào.
OUTLIER_PCT = 15.0

#: Dưới mức này thì lệch KHÔNG phải chuyện của `K` — là hình học.
MIN_R = 0.90

PATHS = (("sino", "K_EXPORT", "d710 osem (sinogram, non-TOF)"),
         ("lm", "K_EXPORT_LM", "d710 lm recon (list-mode, TOF)"))


def load(root, cases=None) -> dict:
    """`{path: [record, ...]}` đọc từ các sidecar trên đĩa."""
    got = {p: [] for p, _, _ in PATHS}
    names = cases or sorted(d.name for d in root.iterdir() if d.is_dir())
    for name in names:
        for p, _, _ in PATHS:
            f = root / name / f"calib_{p}.json"
            if f.exists():
                with open(f) as fh:
                    got[p].append(json.load(fh))
    return got


def summarise(recs, label, out=print) -> dict | None:
    """In bảng một đường tái tạo, trả về `{K, spread_pct, n_cases, ...}`."""
    out(f"\n=== {label}")
    if not recs:
        out("  (chưa có ca nào)")
        return None

    recs = sorted(recs, key=lambda d: d["case"])
    out(f"  {'ca':<16} {'K (ls)':>12} {'K (med)':>12} {'r':>7} "
        f"{'voxel':>10} {'tản':>6}  {'hấp thu':>8}")
    for d in recs:
        out(f"  {d['case']:<16} {d['k_ls']:>12,.0f} {d['k_med']:>12,.0f} "
            f"{d['r']:>7.4f} {d['n']:>10,} {d['spread_pct']:>5.1f}% "
            f"{d['uptake_min']:>7.1f}m")

    k = np.array([d["k_ls"] for d in recs], dtype=np.float64)
    K = float(np.median(k))
    spread = float((k.max() / k.min() - 1) * 100) if len(k) > 1 else 0.0
    dev = 100 * (k / K - 1)

    out(f"\n  trung vị K = {K:,.1f} (Bq/mL)/(count/voxel)"
        f"   [{len(k)} ca, min {k.min():,.0f} max {k.max():,.0f}]")
    if len(k) > 1:
        out(f"  tản giữa các ca: {spread:.1f} %   "
            f"(lệch chuẩn {100 * k.std(ddof=1) / K:.1f} %)")
    else:
        out("  tản giữa các ca: mới một ca, chưa đo được")

    bad_r = [d["case"] for d in recs if d["r"] < MIN_R]
    if bad_r:
        out(f"  ⚠ r < {MIN_R} ở {', '.join(bad_r)} — lệch HÌNH HỌC, không phải"
            " thang. Sửa hình học trước, đừng bù bằng K.")
    far = [(d["case"], v) for d, v in zip(recs, dev) if abs(v) > OUTLIER_PCT]
    for name, v in far:
        out(f"  ⚠ {name} lệch {v:+.1f} % so với trung vị — xem lại ca này thay"
            " vì gộp vào")
    if spread > 20 and len(k) > 1:
        out("  ⚠ tản > 20 % ⇒ chưa nên chốt MỘT hằng số; còn thành phần phụ"
            " thuộc ca chưa mô hình hoá")

    return {"K": K, "spread_pct": spread, "n_cases": len(k),
            "cases": [d["case"] for d in recs],
            "r_min": float(min(d["r"] for d in recs))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="calib_k", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", help="mặc định: mọi ca trong $D710_OUT")
    ap.add_argument("--out", help="gốc đầu ra; mặc định $D710_OUT")
    args = ap.parse_args(argv)

    root = out_root(args.out)
    got = load(root, args.cases)
    if not any(got.values()):
        raise SystemExit(
            f"error: không có calib_*.json nào dưới {root}\n"
            f"  sinh ra bằng: python3 -m tools.compare_vendor --case <c> "
            f"--vendor <PT_s012...> --json {root}/<c>/calib_sino.json")

    res = {}
    for p, const, label in PATHS:
        res[p] = summarise(got[p], label)

    print("\n" + "=" * 68)
    print("dán vào D710/utils/scanner.py:\n")
    for p, const, _ in PATHS:
        s = res[p]
        if s is None:
            print(f"{const} = None      # chưa đo")
        else:
            print(f"{const} = {s['K']:,.1f}".replace(",", "_")
                  + f"      # {s['n_cases']} ca, tản {s['spread_pct']:.1f} %,"
                    f" r >= {s['r_min']:.3f}")
    if res["sino"] and res["lm"]:
        print(f"\nsinogram / list-mode = {res['sino']['K'] / res['lm']['K']:.3f}"
              "   [hai đường không cùng thang — đó là lý do có hai hằng số]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
