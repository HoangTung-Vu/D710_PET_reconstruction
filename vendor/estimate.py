#!/usr/bin/env python3
"""One command: raw sinogram and CT series to randoms, scatter, sensitivity and dead time."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

from utils import container

HERE = os.path.dirname(os.path.abspath(__file__))

OUTPUTS = ["randoms.f32", "scatter.f32", "normdt.f32", "norm_only.f32",
           "prompts.u16", "singles.i32", "dt_int.f32", "dt_mux.f32"]

TOF_OUTPUT = "scatter_tof.f32"


def raw_header(raw):
    """Table position and related fields, straight from the RDF header."""
    out = container.rdf_info(raw)
    info = {}
    for key, pat in (("table_position_mm", r"table_position_mm\s*:\s*(-?[\d.]+)"),
                     ("bed_number", r"bed_number\s*:\s*(\d+)"),
                     ("prompts", r"prompts\s*:\s*([\d,]+)"),
                     ("num_tof_bins", r"num_tof_bins\s*:\s*(\d+)"),
                     ("axial_fov_mm", r"axial_fov_mm\s*:\s*([\d.]+)"),
                     ("frame_duration_ms", r"frame_duration_ms\s*:\s*([\d,]+)")):
        m = re.search(pat, out)
        if m:
            info[key] = float(m.group(1).replace(",", ""))
    m = re.search(r"study_instance_uid\s*:\s*([\d.]+)", out)
    if m:
        info["norm_cal_uid"] = m.group(1)
    m = re.search(r"series_instance_uid\s*:\s*([\d.]+)", out)
    if m:
        info["wcc_cal_uid"] = m.group(1)
    if "table_position_mm" not in info:
        raise SystemExit(
            "error: could not read table_position_mm from %s.\n"
            "  `ge_rdf_tool.py info` said:\n%s" % (raw, out[:2000]))
    return info


def resolve_norm(norm_cal_uid, raw):
    """Find the norm scan this exam declares."""
    if not norm_cal_uid:
        return None
    got = container.cal_tags(norm_cal_uid, "3dnorm",
                             [("kind", 0x00171005), ("src", 0x00171007)])
    if got is None:
        print("   no %s.3dnorm in the image -- cannot resolve the norm "
              "automatically" % norm_cal_uid, file=sys.stderr)
        return None
    kind, src = got.get("kind") or "?", got.get("src") or ""
    print("   cal %s -> %r  %s" % (norm_cal_uid, kind, src))
    if "Normalization" not in kind or not src:
        print("   that cal is not a 3D normalisation; not using it",
              file=sys.stderr)
        return None

    tail = src.lstrip("/").split("/")
    d = os.path.dirname(os.path.abspath(raw))
    for _ in range(6):
        d = os.path.dirname(d)
        cand = os.path.join(d, *tail)
        if os.path.exists(cand):
            return cand
    local = os.path.join(HERE, "cal", "norm_DXRM3_20231020.rdf")
    record = os.path.join(HERE, "cal", norm_cal_uid + ".3dnorm")
    if os.path.exists(local) and os.path.exists(record) and \
            bundled_source(record) == src:
        print("   not in this drop; using the bundled copy %s"
              % os.path.relpath(local, HERE))
        return local
    print("   the exam declares %s but it is not in this drop, and the bundled "
          "copy is a different scan" % src, file=sys.stderr)
    return None


def bundled_source(record):
    """The (0017,1007) tag of a `.3dnorm` kept in `vendor/cal/`, or an empty string."""
    d, name = os.path.dirname(record), os.path.basename(record)
    code = ("import json,pydicom\n"
            "d=pydicom.dcmread('/cal/%s', force=True)\n"
            "print(json.dumps(str(d[0x00171007].value) "
            "if 0x00171007 in d else ''))\n" % name)
    p = container.python(["-c", code], mounts=[(d, "/cal", "ro")],
                         capture=True, check=False, verbose=False)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return ""


def write_job(dst, emission, transmission, normalization):
    """Copy the vendor's XR job, replacing only the three input paths."""
    swaps = {"inputEmissionFileName[0]": emission,
             "inputTransmissionFileName[0]": transmission,
             "normalizationSinogramFile": normalization}
    seen = set()
    with open(os.path.join(HERE, "job.gdb")) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        for field, path in swaps.items():
            if 'IgJobReq.%s"' % field in line:
                lines[i] = 'python _s("IgJobReq.%s", "%s")\n' % (field, path)
                seen.add(field)
    missing = set(swaps) - seen
    if missing:
        raise SystemExit("error: job.gdb has no line for %s" % ", ".join(missing))
    with open(dst, "w") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="emission SINO* (one bed)")
    ap.add_argument("--ct", required=True, help="CT DICOM series directory")
    ap.add_argument("--norm", help="normalisation SINO*. Normally unnecessary: "
                                   "the exam's own header names its norm cal "
                                   "and that is resolved automatically.")
    ap.add_argument("--no-auto-norm", action="store_true",
                    help="do not resolve the norm from the exam header; fall "
                         "back to GE's selftest norm unless --norm is given")
    ap.add_argument("--out", required=True, help="output directory for THIS bed")
    ap.add_argument("--no-tof", action="store_true",
                    help="run GE's plain 3D OSEM job (reconMethod 2) instead of "
                         "the TOF one, so no scatter_tof.f32 is produced")
    ap.add_argument("--table-location", type=float,
                    help="override the bed table position read from the RDF")
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--keep-going", action="store_true",
                    help="do not stop if the container exits non-zero")
    args = ap.parse_args()
    tof = not args.no_tof
    outputs = OUTPUTS + ([TOF_OUTPUT] if tof else [])

    container.ensure_image()

    raw = os.path.abspath(args.raw)
    ct = os.path.abspath(args.ct)
    out = os.path.abspath(args.out)
    for p in (raw, ct):
        if not os.path.exists(p):
            raise SystemExit("error: no such path: %s" % p)
    os.makedirs(out, exist_ok=True)

    print("== reading the raw header")
    info = raw_header(raw)
    table = args.table_location if args.table_location is not None \
        else info["table_position_mm"]
    print("   bed %s, table %.2f mm, %s prompts, %s TOF bins"
          % (info.get("bed_number"), table, info.get("prompts"),
             info.get("num_tof_bins")))

    data = os.path.join(out, "data")
    os.makedirs(data, exist_ok=True)
    shutil.copy2(raw, os.path.join(data, "emission.rdf"))

    selftest_norm = ("/usr/PET/release/petig/selftest/data/selftest_kh3d_norm.rdf")
    norm = os.path.abspath(args.norm) if args.norm else None
    if norm is None and not args.no_auto_norm:
        print("== resolving the norm the exam itself declares")
        norm = resolve_norm(info.get("norm_cal_uid"), raw)
    if norm:
        shutil.copy2(norm, os.path.join(data, "norm.rdf"))
        norm_in_container = "/data/norm.rdf"
        print("   norm: %s" % norm)
    else:
        norm_in_container = selftest_norm
        print("!! falling back to the VENDOR SELFTEST norm\n"
              "!! (%s).\n"
              "!! normdt.f32 / norm_only.f32 will then describe GE's test\n"
              "!! scanner, NOT yours.  randoms and scatter are unaffected."
              % selftest_norm, file=sys.stderr)

    print("== CT -> mu-map -> PIFA")
    container.python(
        ["/d710/vendor/ct_to_pifa.py", "/ct", "/out/data/mu.pifa",
         "--table-location", table],
        mounts=container.d710_mounts(os.path.dirname(HERE))
        + [(ct, "/ct", "ro"), (out, "/out", "rw")])
    pifa = os.path.join(data, "mu.pifa")
    if not os.path.exists(pifa):
        raise SystemExit("error: ct_to_pifa wrote no %s" % pifa)

    print("== writing the job")
    write_job(os.path.join(out, "job.gdb"), "/data/emission.rdf",
              "/data/mu.pifa", norm_in_container)

    for f in OUTPUTS + [TOF_OUTPUT]:
        for p in (os.path.join(out, f), os.path.join(out, f + ".json")):
            if os.path.exists(p):
                os.remove(p)

    print("== running GE's pet_recon (this takes a few minutes)")
    print("   scatter: %s" % ("TOF (reconMethod 3)" if tof
                              else "non-TOF (reconMethod 2)"))
    env = dict(os.environ, D710_JOB="/out/job.gdb",
               D710_TOF="1" if tof else "0")
    log = os.path.join(out, "extract.log")
    with open(log, "wb") as lf:
        rc = subprocess.run(["timeout", str(args.timeout),
                             os.path.join(HERE, "run.sh"),
                             "--out", out, "--data", data, "extract.gdb"],
                            env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    print("   container exit %d, log -> %s" % (rc, log))
    if rc != 0 and not args.keep_going:
        raise SystemExit("error: the container failed; see %s "
                         "(use --keep-going to collect partial output)" % log)

    got = [f for f in outputs if os.path.exists(os.path.join(out, f))]

    wcc = container.cal_tags(info.get("wcc_cal_uid"), "3dwcc",
                             [("name", 0x00191006), ("factor", 0x0019100B)]) \
        if info.get("wcc_cal_uid") else None

    with open(os.path.join(out, "estimate.json"), "w") as f:
        json.dump({"raw": raw, "ct": ct,
                   "norm": norm or selftest_norm,
                   "norm_source": ("--norm" if args.norm else
                                   "resolved from norm_cal_uid" if norm else
                                   "vendor selftest fallback"),
                   "mu_orientation": "DICOM LPS (measured; no flips)",
                   "table_position_mm": table, "rdf_header": info,
                   "recon_method": 3 if tof else 2,
                   "tof_scatter": tof,
                   "outputs": got, "container_exit": rc,
                   "wcc_applied": False,
                   "wcc_name": (wcc or {}).get("name"),
                   "wcc_activity_factor": (
                       float(wcc["factor"]) if wcc and wcc.get("factor") else None)},
                  f, indent=2, sort_keys=True)

    print()
    for f in got:
        print("   %s/%s" % (out, f))
    missing = [f for f in outputs if f not in got]
    if missing:
        print("\n!! missing: %s -- check %s" % (", ".join(missing), log),
              file=sys.stderr)
        return 1
    print("\nread them with:  d710 read %s/scatter.f32" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
