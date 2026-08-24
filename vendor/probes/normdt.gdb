# Norm x deadtime, straight out of GE's kernels, with NO WCC in it.
#
# The recipe is GE's own, lifted from CAccelIntfNormDTData::ApplyNormDeadtime
# (0x4a39e0), which is what the accelerator path uses to build its NormDT
# sinogram:
#
#   GE_vfill(1.0f, work, number_v_theta * numberSamples)   <- a view of ones
#   if (IgJobReq.normalizationFlag)    CNorm3d::ApplyNormalization(work, view, scratch, false)
#   if (IgJobReq.emissionDeadTimeFlag) CDeadtime3d::ApplyDeadtime(work, view, work, 0, &f1, &f2, scratch, false)
#   ... then PadData / projTranspose / p_rev4, which are accelerator packing
#       only and are deliberately NOT reproduced here.
#
# Applying the corrections to a buffer of ones is exactly what makes the
# result the multiplicative correction itself rather than corrected data.
# Note what is absent: no wccActivityFactor, no wccSensitivityFactor, no
# m_fwccScaleFactor.  This routine never touches them.
#
# Norm data itself is loaded by setup3dEmissJob(&IgJobReq, false): with the
# bool false it skips the emission-segment loop and does only the
# normalizationFlag and attenuationFlag loads, through sharcCmp3dRemoteLoad --
# AP side, no FIFO handshake.
source /vendor/boot.gdb
source /vendor/lib.gdb
# 0x1FF makes sharcCmpOpenDataFiles dump every RDF header field -- ~50k lines
# through a pipe, which is most of a run's wall clock.  0x4 keeps the LOAD
# messages, which are the ones worth having.
set var sharcCmpDebugFlag = 0x4
set var PrintTraceFlag = 1
source /vendor/job.gdb

python
NU, NV, NVIEW = 381, 553, 288          # checked against the param struct below
n = NU * NV

# Everything up to the loads runs with the pool frozen.  Running
# sharcCmpOpenDataFiles unlocked and then re-locking is what made
# ReallocateBuffs SIGSEGV inside glibc, reproducibly; with the whole prologue
# locked it succeeds every time.
banner("open + allocate")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $cd = $ig->m_pCorrDataMem._M_ptr")
gdb.execute("set $im = $ig->m_pImageDataMem._M_ptr")
gdb.execute("set $ps = $ig->m_pParamStruct._M_ptr")
# allocation-heavy: keep the other threads frozen (see boot.gdb)
for c in ("$ig->CleanDataBuffers(0)", "$ig->InitParamStruct()",
          "$rd->DeallocateBuffs(0, $ig)", "$cd->DeallocateBuffs(0, $ig)",
          "$rd->ReallocateBuffs(0, $ig)", "$cd->ReallocateBuffs(0, $ig)",
          "$im->ReallocateBuffs(0, $ig)"):
    show(c)

banner("geometry check -- the constants above must match the live struct")
got = (int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_u") or 0),
       int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_v_theta") or 0),
       int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_phi") or 0))
say("   live (number_u, number_v_theta, number_phi) = %s, assumed %s\n"
    % (got, (NU, NV, NVIEW)))
if got != (NU, NV, NVIEW):
    NU, NV, NVIEW = got
    n = NU * NV
    say("   !! using the live values instead\n")

banner("loads -- these dispatch to the thread pool, so unlock")
with unlocked():
    show("$ig->LoadRawData()")
    show("$ig->InitLuts()")
    gdb.execute("set var *(int *) (0xf38968 + 8) = 0")   # singlesType  = 0
    show("((int (*)(void *, void *)) sharcCmp3dRemoteSinglesLoad)(&IgJobReq, (void *) 0xf38968)")
    gdb.execute("set var *(int *) (0xf38980 + 8) = 0")   # deadtimeType = 0
    show("((int (*)(void *, void *)) sharcCmp3dRemoteDeadtimeLoad)(&IgJobReq, (void *) 0xf38980)")
    show("((int (*)(void *, char)) setup3dEmissJob)(&IgJobReq, (char) 0)")

show("*$rd->m_pSingles@6")
show("*$rd->m_pIntDeadtime@4")
show("$rd->m_pNorm->m_pCyclicMemBuff->m_uiAllocElement")
dump_views("$rd->m_pNorm", "norm_raw.f32", "f4",
           {"note": "CRawDataMem::m_pNorm, as loaded from normalizationSinogramFile"})

banner("construct CNorm3d and CDeadtime3d on the live pointers")
# Both constructors are pure field stores, so they are replayed by writing the
# fields rather than by calling them.  Calling them through a cast function
# pointer is what gdb kept refusing -- and it reports that refusal as an
# ordinary command error, so the object silently stayed uninitialised malloc
# memory and Initialize() faulted.  Writing the bytes cannot fail that way.
#
#   CNorm3d      (32 B): [0]=CRawDataMem* [8]=paramStruct* [0x10]=0 [0x18]=0
#   CDeadtime3d  (56 B): [0]=CRawDataMem* [8]=paramStruct*
#                        [0x10..0x28]=0   [0x30]=1.0f
import struct as _s
inf = gdb.selected_inferior()
rd_p = int(val("$rd"))
ps_p = int(val("$ig->m_pParamStruct._M_ptr"))

gdb.execute("set $n3 = (CNorm3d *) malloc(32)")
n3 = int(val("$n3"))
inf.write_memory(n3, _s.pack("<QQQQ", rd_p, ps_p, 0, 0))

gdb.execute("set $d3 = (CDeadtime3d *) malloc(56)")
d3 = int(val("$d3"))
inf.write_memory(d3, _s.pack("<QQQQQQ", rd_p, ps_p, 0, 0, 0, 0)
                     + _s.pack("<f", 1.0) + b"\x00" * 4)

show("$n3->m_pRawDataMem")
show("$d3->m_pRawDataMem")
show("$n3->Initialize()")
show("$d3->Initialize()")

banner("norm x deadtime, all %d views" % NVIEW)
gdb.execute("set $buf = (float *) malloc(%d)" % (n * 4))
gdb.execute("set $scr = (float *) malloc(%d)" % (5 * n * 4))   # 5x, per NewReconRequest
gdb.execute("set $f1  = (float *) malloc(16)")
gdb.execute("set $f2  = (float *) malloc(16)")
buf = int(val("$buf"))
ones = _ones = __import__("struct").pack("<f", 1.0) * n
inf = gdb.selected_inferior()

import os
path = os.path.join(OUT, "normdt.f32")
ok = True
with open(path, "wb") as f:
    for v in range(NVIEW):
        inf.write_memory(buf, ones)
        gdb.execute("set var *$f1 = 0")
        gdb.execute("set var *$f2 = 0")
        r1 = ex("print $n3->ApplyNormalization($buf, %d, $scr, 0)" % v, quiet=True)
        r2 = ex("print $d3->ApplyDeadtime($buf, %d, $buf, 0, $f1, $f2, $scr, 0)" % v,
                quiet=True)
        if r1 is None or r2 is None:
            say("!! view %d failed:  norm=%s  deadtime=%s\n" % (v, r1, r2))
            ok = False
            break
        f.write(bytes(inf.read_memory(buf, n * 4)))
        if v % 48 == 0:
            say("   view %3d  norm=%s  dt=%s\n"
                % (v, (r1 or "").strip().split("= ")[-1],
                      (r2 or "").strip().split("= ")[-1]))

if ok:
    import json, array
    a = array.array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    nz = [x for x in a if x != 0.0]
    meta = dict(name="normdt.f32", axes="view(%d) x v(%d) x u(%d)" % (NVIEW, NV, NU),
                dtype="f4", elements=len(a), nonzero=len(nz),
                min_nonzero=min(nz) if nz else None, max=max(nz) if nz else None,
                wcc_applied=False,
                produced_by="CNorm3d::ApplyNormalization + CDeadtime3d::ApplyDeadtime"
                            " on a unit sinogram, per CAccelIntfNormDTData::ApplyNormDeadtime")
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    say("== normdt.f32: %d elements  nonzero=%d  min=%s max=%s\n"
        % (len(a), len(nz), meta["min_nonzero"], meta["max"]))

banner("DONE")
end
