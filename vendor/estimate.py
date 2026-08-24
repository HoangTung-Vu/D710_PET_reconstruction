#!/usr/bin/env python3
"""One command: raw sinogram + CT DICOM -> randoms, scatter, sensitivity, dead time.

Normally driven by the CLI, which works out --out for you:

    d710 estimate --raw <petRDFS/.../DIR> --ct <CT series> --case ped --bed 6

By hand it is the same thing with the paths spelled out:

    PYTHONPATH=<D710> python3 vendor/estimate.py \\
        --raw ~/Documents/12082026/petRDFS/.../SINO0001 \\
        --ct  ~/Documents/12082026/PESI/p1/e1/s2 \\
        --out $D710_OUT/nema/vendor/bed2

Everything is estimated by GE's own `pet_recon`, running under gdb in the
`d710:full` container -- these are the vendor's kernels, not a reimplementation.

**This script needs nothing but python3 and docker.** No conda, no numpy, no
pydicom: every step that needs a library runs inside the image, which already
has them, and every vendor file it reads is already in the image too. It stays
on the host only because it drives `docker` itself.

What it does, in order:

  1. reads the raw RDF header (`ge_rdf_tool.py info`, in the container) for the
     bed's table position, which is what registers the CT to the PET bed;
  2. turns the CT series into a mu-map on GE's PIFA grid (`ct_to_pifa.py`, in
     the container);
  3. writes a job file from the vendor's own XR job template, with only the
     three input paths swapped;
  4. runs `extract.gdb` in the container with the data mounted read-only and
     --out mounted straight onto /out, so the kernels write their results in
     place -- no staging directory, and two beds can run at the same time;
  5. leaves the sinograms plus a JSON sidecar each in --out.

Outputs (all float32 unless noted, view x v x u = 288 x 553 x 381):

    randoms.f32     randoms
    scatter.f32     model-based (SSS) scatter
    normdt.f32      normalisation x dead time   <- the sensitivity term
    norm_only.f32   normalisation alone; dead time = normdt / norm_only
    prompts.u16     the emission sinogram as the vendor loaded it
    singles.i32     per-crystal singles          576 x 24
    dt_int.f32      per-block dead time          256

No well-counter (WCC) scaling is applied anywhere, so the absolute scale is
yours to calibrate -- but the exam's own WCC factor is recorded in
`estimate.json`, which is where `d710 export` picks it up.

LIST-MODE: not accepted here.  A LIST*.BLF has to be histogrammed into a SINO*
first; see `d710 decode --listmode`.  Every LIST*.BLF on disk has a matching
SINO* from the same acquisition, so pass that instead.
"""
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


def raw_header(raw):
    """table_position_mm and friends, straight from the RDF header.

    The text form of `info` rather than `--json`: every number below is read by
    a regex that was calibrated against exactly that printout, and the two
    forms are not guaranteed to round the same way.  Host and container were
    diffed on a real bed -- the only line that differs is the one echoing the
    file's own path, which nothing here reads.
    """
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
    # Header 0xEEC is the norm cal UID.  ge_rdf_tool still prints it under the
    # old, wrong label "study_instance_uid" (0xF74 / "series_instance_uid" is
    # really the WCC cal UID) -- see the d710-wcc-and-norm-cal note.
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
    """Find the norm scan THIS exam declares, instead of guessing one.

    The chain, all of it checkable:

        emission RDF header 0xEEC  = norm_cal_uid
          -> /usr/PET/systemConfig/cal/<uid>.3dnorm   (508 B of DICOM, in the image)
             (0017,1005) "PET 3D Normalization"   <- not a WCC scan
             (0017,1007) /petRDFS/<a>/<b>/<c>/SINO000n   <- the source scan
          -> that same relative path under the exam's own drop directory

    The DICOM read happens in the container, because that is where the vendor's
    calibration tree lives now.  The walk down the drop happens here, because
    the drop is on the host and only the host can see it.

    Returns the host path, or None with a printed reason.
    """
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

    # src is a console path like /petRDFS/AAA/BBB/CCC/SINO0001.  The exam drop
    # keeps the same tree, so walk up from the raw file until petRDFS matches.
    tail = src.lstrip("/").split("/")
    d = os.path.dirname(os.path.abspath(raw))
    for _ in range(6):
        d = os.path.dirname(d)
        cand = os.path.join(d, *tail)
        if os.path.exists(cand):
            return cand
    # Not in the drop -- fall back to the copy kept beside the tool, but ONLY
    # when that copy is the very scan this exam declares.  The repo's own
    # `cal/<uid>.3dnorm` names the scan the bundled `.rdf` came from, so the
    # two console paths have to be the same string.  Without this check any
    # exam whose norm is missing gets the bundled one handed to it -- another
    # calibration, possibly another scanner, and nothing downstream can tell:
    # the sinogram has the right shape and the image still looks like an image.
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
    """(0017,1007) out of a `.3dnorm` kept in `vendor/cal/`, or "".

    Read in the container for the same reason as the one above: pydicom is in
    the image, not necessarily on the host.  The file is mounted in.
    """
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
    """Copy the vendor's XR job, swapping only the three input paths.

    job.gdb carries all 524 IgJobReq fields from GE's own selftest_kh_3dir.job.
    Regenerating it from scratch would mean re-deriving every correction flag,
    so only the file names are touched and everything else stays the vendor's.
    """
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
    ap.add_argument("--table-location", type=float,
                    help="override the bed table position read from the RDF")
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--keep-going", action="store_true",
                    help="do not stop if the container exits non-zero")
    args = ap.parse_args()

    container.ensure_image()

    raw = os.path.abspath(args.raw)
    ct = os.path.abspath(args.ct)
    out = os.path.abspath(args.out)
    for p in (raw, ct):
        if not os.path.exists(p):
            raise SystemExit("error: no such path: %s" % p)
    os.makedirs(out, exist_ok=True)

    # ---------------------------------------------------------- 1. the bed
    print("== reading the raw header")
    info = raw_header(raw)
    table = args.table_location if args.table_location is not None \
        else info["table_position_mm"]
    print("   bed %s, table %.2f mm, %s prompts, %s TOF bins"
          % (info.get("bed_number"), table, info.get("prompts"),
             info.get("num_tof_bins")))

    # The container sees one directory, so everything the job names has to live
    # under it.  Copy the two inputs in rather than mounting three host trees.
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

    # -------------------------------------------------------- 2. the mu-map
    # In the container: it has numpy/scipy/pydicom, and the result was checked
    # byte-for-byte against the host's on a real bed.
    print("== CT -> mu-map -> PIFA")
    container.python(
        ["/d710/vendor/ct_to_pifa.py", "/ct", "/out/data/mu.pifa",
         "--table-location", table],
        mounts=container.d710_mounts(os.path.dirname(HERE))
        + [(ct, "/ct", "ro"), (out, "/out", "rw")])
    pifa = os.path.join(data, "mu.pifa")
    if not os.path.exists(pifa):
        raise SystemExit("error: ct_to_pifa wrote no %s" % pifa)

    # ----------------------------------------------------------- 3. the job
    print("== writing the job")
    write_job(os.path.join(out, "job.gdb"), "/data/emission.rdf",
              "/data/mu.pifa", norm_in_container)

    # --------------------------------------------------------- 4. the recon
    # --out IS /out.  extract.gdb writes the final files straight into it, so
    # there is no staging directory to collide over and nothing to move.
    for f in OUTPUTS:
        for p in (os.path.join(out, f), os.path.join(out, f + ".json")):
            if os.path.exists(p):
                os.remove(p)

    print("== running GE's pet_recon (this takes a few minutes)")
    env = dict(os.environ, D710_JOB="/out/job.gdb")
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

    # --------------------------------------------------------- 5. the sidecar
    got = [f for f in OUTPUTS if os.path.exists(os.path.join(out, f))]

    # The exam's own well-counter factor, recorded here rather than looked up
    # again at export time: this is the one moment the cal UID and a container
    # are both in hand.  Nothing is scaled by it -- `d710 export` decides that.
    wcc = container.cal_tags(info.get("wcc_cal_uid"), "3dwcc",
                             [("name", 0x00191006), ("factor", 0x0019100B)]) \
        if info.get("wcc_cal_uid") else None

    with open(os.path.join(out, "estimate.json"), "w") as f:
        json.dump({"raw": raw, "ct": ct,
                   # the norm actually used, not the one asked for: without
                   # this the sidecar claims the selftest norm on every run
                   # that resolved its own.
                   "norm": norm or selftest_norm,
                   "norm_source": ("--norm" if args.norm else
                                   "resolved from norm_cal_uid" if norm else
                                   "vendor selftest fallback"),
                   "mu_orientation": "DICOM LPS (measured; no flips)",
                   "table_position_mm": table, "rdf_header": info,
                   "outputs": got, "container_exit": rc,
                   "wcc_applied": False,
                   "wcc_name": (wcc or {}).get("name"),
                   "wcc_activity_factor": (
                       float(wcc["factor"]) if wcc and wcc.get("factor") else None)},
                  f, indent=2, sort_keys=True)

    print()
    for f in got:
        print("   %s/%s" % (out, f))
    missing = [f for f in OUTPUTS if f not in got]
    if missing:
        print("\n!! missing: %s -- check %s" % (", ".join(missing), log),
              file=sys.stderr)
        return 1
    print("\nread them with:  d710 read %s/scatter.f32" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
