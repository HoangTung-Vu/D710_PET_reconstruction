# extract.gdb -- drive GE's own pet_recon and write its correction sinograms
# to /out.  This is the tool; everything in probes/ is scaffolding that built it.
#
#   ./run.sh extract.gdb
#
# Outputs (each <name>.f32 has a <name>.f32.json sidecar with shape + stats):
#
#   normdt.f32     norm x dead time, 288 x 553 x 381 float32   <- the sensitivity
#   randoms.f32    randoms sinogram, same shape, from GE's own prep stage
#   scatter.f32    SSS scatter sinogram, same shape, from the mu-map
#   norm_only.f32  the same with dead time skipped, so
#                  deadtime = normdt / norm_only
#   prompts.u16    the emission sinogram as loaded, 288 x 553 x 381 uint16
#   singles.i32    per-crystal singles, 576 x 24 int32
#   dt_int.f32     per-block integral dead time, 256 float32
#   dt_mux.f32     per-block mux dead time (all zero: dt_3dmux = 0)
#
# NO WCC is applied anywhere: neither ApplyNormalization nor ApplyDeadtime
# touches wccActivityFactor / wccSensitivityFactor / m_fwccScaleFactor, and
# this script never calls anything that does.  The absolute scale is left for
# you to calibrate.
#
# Read the outputs with:  python3 read_out.py out/normdt.f32
#
# ---------------------------------------------------------------------------
# The call sequence below is CIgManager::RunCyclic's, replayed by hand.  Every
# step earned its place; see README.md section 3 for why each one is here and
# what breaks without it.
# ---------------------------------------------------------------------------
source /vendor/boot.gdb
source /vendor/lib.gdb

# 0x1FF makes sharcCmpOpenDataFiles dump every parsed RDF header field, which
# is worth having once (it is the source of PARAMS.md) but costs most of a
# run's wall clock.  0x4 keeps the LOAD messages.
set var sharcCmpDebugFlag = 0x4
set var PrintTraceFlag = 1

python
import os
# The job is sourced from here rather than by a fixed `source` line so that
# estimate.py can point it at a generated one:  D710_JOB=/out/job.gdb
JOB = os.environ.get("D710_JOB", "/vendor/job.gdb")
# --------------------------------------------------------------- 1. allocate
# Everything here runs with the processing threads frozen (boot.gdb sets
# scheduler-locking on).  Running any of it unlocked lets the pool race the
# allocator inside CCyclicMemBuffer::AllocateMem.
banner("job: %s" % JOB)
gdb.execute("source " + JOB)

banner("open data files + allocate buffers")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $cd = $ig->m_pCorrDataMem._M_ptr")
gdb.execute("set $im = $ig->m_pImageDataMem._M_ptr")

# DeallocateBuffs before ReallocateBuffs is NOT optional.  ReallocateBuffs
# opens with `if (GetNumAllocViews() != 0) skip`, and a freshly constructed
# CViewBuffer already reports 1 element of 2 bytes -- so without the
# Deallocate every Reallocate returns GRE_IG_SUCCESS having allocated nothing,
# and CRawDataLoad can then put view 0 and no other.
for c in ("$ig->CleanDataBuffers(0)", "$ig->InitParamStruct()",
          "$rd->DeallocateBuffs(0, $ig)", "$cd->DeallocateBuffs(0, $ig)",
          "$rd->ReallocateBuffs(0, $ig)", "$cd->ReallocateBuffs(0, $ig)",
          "$im->ReallocateBuffs(0, $ig)"):
    show(c)

# --------------------------------------------------------------- 2. geometry
banner("geometry, read from the live parameter struct")
NU    = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_u")       or 0)
NV    = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_v_theta") or 0)
NVIEW = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_phi")     or 0)
NTOF  = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "numTOF_bins")    or 0)
say("   number_u=%d  number_v_theta=%d  number_phi=%d  numTOF_bins=%d\n"
    % (NU, NV, NVIEW, NTOF))
if not (NU and NV and NVIEW):
    raise gdb.GdbError("geometry is zero -- InitParamStruct did not run")
n = NU * NV
GEOM = dict(number_u=NU, number_v_theta=NV, number_phi=NVIEW, numTOF_bins=NTOF,
            axes="view(%d) x v(%d) x u(%d)" % (NVIEW, NV, NU))

# ------------------------------------------------------------------ 3. loads
# These three do dispatch to the thread pool (LoadRawData runs CRawDataLoad on
# it), so the pool has to be running for them.
banner("load prompts / singles / dead time / norm")
with unlocked():
    show("$ig->LoadRawData()")
    show("$ig->InitLuts()")

    # Singles and the dead-time block are separate AP-side loads that
    # LoadRawData does not perform.  Both structs are file-scope globals whose
    # type is anonymous in the DWARF, so their fields are written through the
    # fixed addresses of the non-PIE binary; both are
    # {s32 status; s32 dragonSocketStatus; n32 type;}.
    #   sharcCmp3dRemoteSinglesLoad  needs IgJobReq.emissionRandomsFlag == 3
    #                                and singlesType == 0
    #   sharcCmp3dRemoteDeadtimeLoad needs (1 << deadtimeType) & 0xd
    gdb.execute("set var *(int *) (0xf38968 + 8) = 0")   # singlesType  = 0
    show("((int (*)(void *, void *)) sharcCmp3dRemoteSinglesLoad)(&IgJobReq, (void *) 0xf38968)")
    gdb.execute("set var *(int *) (0xf38980 + 8) = 0")   # deadtimeType = 0
    show("((int (*)(void *, void *)) sharcCmp3dRemoteDeadtimeLoad)(&IgJobReq, (void *) 0xf38980)")

    # setup3dEmissJob(jobReq, false): the false skips the emission-segment
    # loop (LoadRawData already did that) and leaves the normalizationFlag and
    # attenuationFlag loads, both through sharcCmp3dRemoteLoad -- AP side, no
    # FIFO handshake, which is what makes this reachable at all.
    show("((int (*)(void *, char)) setup3dEmissJob)(&IgJobReq, (char) 0)")

banner("what got loaded")
show("*$rd->m_pSingles@6")
show("*$rd->m_pIntDeadtime@4")
show("$rd->m_pNorm->m_pCyclicMemBuff->m_uiAllocElement")

# ------------------------------------------------------- 4. raw dumps
banner("dump the inputs")
dump_views("$rd->m_pPrompts", "prompts.u16", "u2",
           dict(GEOM, note="emission sinogram as loaded by CRawDataLoad"))
NCRYS  = int(deep("$ig->m_pParamStruct._M_ptr", "crystalsPerRing")  or 576)
NRING  = int(deep("$ig->m_pParamStruct._M_ptr", "axialCrystals")    or 24)
NBLOCK = int(deep("$ig->m_pParamStruct._M_ptr", "blocksPerSystem")  or 0)
say("   crystalsPerRing=%d axialCrystals=%d blocksPerSystem=%d\n"
    % (NCRYS, NRING, NBLOCK))

dump_expr("$rd->m_pSingles", NCRYS * NRING * 4, "singles.i32",
          {"dtype": "i4", "layout": "crystalsPerRing=%d x axialCrystals=%d"
                                    % (NCRYS, NRING),
           "note": "per-crystal singles, from rdfReadSingles"})

# The dead-time arrays are per BLOCK, not per crystal.  Dumping them as
# crystalsPerRing*axialCrystals read 13x past the end and produced values like
# 7.9e34; the plausible prefix stopped at exactly 256 = blocksPerSystem, which
# is also the granularity apCfg.avgBlockDeadtime implies.
if NBLOCK:
    for field, name in (("m_pIntDeadtime", "dt_int.f32"),
                        ("m_pMuxDeadtime", "dt_mux.f32")):
        dump_expr("$rd->" + field, NBLOCK * 4, name,
                  {"dtype": "f4", "layout": "blocksPerSystem=%d" % NBLOCK,
                   "note": "per-block dead time, from rdfReadDeadTime"})
else:
    say("!! blocksPerSystem is 0 -- skipping the dead-time dumps rather than"
        " guessing their length\n")

# ------------------------------------------------- 5. norm x dead time
# GE's own recipe, from CAccelIntfNormDTData::ApplyNormDeadtime (0x4a39e0) and
# repeated in COsem3dPrep::DoPrep:
#
#   GE_vfill(1.0f, work, number_v_theta * numberSamples)
#   if (normalizationFlag)    CNorm3d::ApplyNormalization(work, phi, scratch, false)
#   if (emissionDeadTimeFlag) CDeadtime3d::ApplyDeadtime(work, phi, work, 0,
#                                                        &f1, &f2, scratch, false)
#
# Feeding the kernels a sinogram of ones is what makes the result the
# multiplicative correction itself instead of corrected data.  The PadData /
# projTranspose / p_rev4 that follow in GE's version are accelerator packing
# and are deliberately not reproduced.
banner("build CNorm3d + CDeadtime3d")
# Both constructors are pure field stores, so they are replayed by writing the
# fields.  Calling them through a cast function pointer is fragile in gdb and
# fails as an ordinary command error, which leaves the object as uninitialised
# malloc memory and makes Initialize() fault somewhere far away.
#   CNorm3d     (32 B): [0]=CRawDataMem* [8]=paramStruct* [0x10]=0 [0x18]=0
#   CDeadtime3d (56 B): [0]=CRawDataMem* [8]=paramStruct* [0x10..0x28]=0
#                       [0x30]=1.0f
import struct as _s, os, json, array
inf  = gdb.selected_inferior()
rd_p = int(val("$rd"))
ps_p = int(val("$ig->m_pParamStruct._M_ptr"))

setv("set $n3 = (CNorm3d *) malloc(32)")
inf.write_memory(int(val("$n3")), _s.pack("<QQQQ", rd_p, ps_p, 0, 0))
setv("set $d3 = (CDeadtime3d *) malloc(56)")
inf.write_memory(int(val("$d3")),
                 _s.pack("<QQQQQQ", rd_p, ps_p, 0, 0, 0, 0)
                 + _s.pack("<f", 1.0) + b"\x00" * 4)
show("$n3->m_pRawDataMem")
show("$d3->m_pRawDataMem")
show("$n3->Initialize()")
show("$d3->Initialize()")

# scratch is 5x a view: CAccelIntfNormDTData::NewReconRequest sizes its work
# area as 5 * numberSamples * number_v_theta floats.
setv("set $buf = (float *) malloc(%d)" % (n * 4))
setv("set $scr = (float *) malloc(%d)" % (5 * n * 4))
setv("set $f1  = (float *) malloc(16)")
setv("set $f2  = (float *) malloc(16)")
buf  = int(val("$buf"))
ones = _s.pack("<f", 1.0) * n

def sweep(name, deadtime, meta):
    """Push a unit sinogram through the kernels, view by view, into /out."""
    path = os.path.join(OUT, name)
    with open(path, "wb") as f:
        for v in range(NVIEW):
            inf.write_memory(buf, ones)
            gdb.execute("set var *$f1 = 0")
            gdb.execute("set var *$f2 = 0")
            clear_df()
            r = ex("print $n3->ApplyNormalization($buf, %d, $scr, 0)" % v, quiet=True)
            if r is None:
                say("!! %s: norm failed at view %d\n" % (name, v)); return False
            if deadtime:
                clear_df()
                r = ex("print $d3->ApplyDeadtime($buf, %d, $buf, 0, $f1, $f2, $scr, 0)" % v,
                       quiet=True)
                if r is None:
                    say("!! %s: deadtime failed at view %d\n" % (name, v)); return False
            f.write(bytes(inf.read_memory(buf, n * 4)))
            if v % 96 == 0:
                say("   %s view %3d/%d\n" % (name, v, NVIEW))
    a = array.array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    nz = [x for x in a if x != 0.0]
    m = dict(GEOM); m.update(meta)
    m.update(name=name, dtype="f4", elements=len(a), nonzero=len(nz),
             min_nonzero=min(nz) if nz else None, max=max(nz) if nz else None,
             wcc_applied=False)
    with open(path + ".json", "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)
    say("== %s: %d elements  nonzero=%d  min=%s max=%s\n"
        % (name, len(a), len(nz), m["min_nonzero"], m["max"]))
    return True

banner("sweep: norm x dead time, %d views" % NVIEW)
sweep("normdt.f32", True,
      {"produced_by": "CNorm3d::ApplyNormalization + CDeadtime3d::ApplyDeadtime"
                      " on a unit sinogram"})
banner("sweep: norm only, %d views" % NVIEW)
sweep("norm_only.f32", False,
      {"produced_by": "CNorm3d::ApplyNormalization on a unit sinogram",
       "note": "deadtime factor = normdt.f32 / norm_only.f32"})

# --------------------------------------------- 6. randoms + scatter
# Both come out of GE's own per-view prep stage, COsem3dPrep::DoPrep.  Note
# CIgManager::InitRandomsFromSingles does NOT make a sinogram -- it only calls
# CalculateRFS (GE_meanvx + initRFSParams), i.e. the randoms-from-singles
# PARAMETERS, and returns SUCCESS with an all-zero buffer.
#
# Four things have to be true at once, and each was found the hard way:
#
#  1. attenuationFlag stays ON.  Turning it off to dodge the CTAC dodges the
#     SSS scatter too -- measured: attn=2 -> 97 MSCAT_* phases, attn=0 -> zero.
#     SSS needs the mu information, so this is physics, not a flag quirk.
#
#  2. TransSysGeometry must match sysGeometry.  CPetReconContextCTAC::
#     ValidateCTAC (0x43e550) compares ~20 geometry fields EX vs CTAC, and
#     TransSysGeometry is all zeros whenever no transmission RDF was opened
#     ("radialModulesPerSystem EX: 32 CTAC: 0").  Copying the 2048-byte
#     emission struct onto it makes every comparison an identity.
#
#  3. The mu-map must be in CRawDataMem::m_pAttn.  CCTAC_3D wants 47 input
#     views of 65536 B = 128*128*4 -- the PIFA payload, one mu slice per view.
#     fileStatus[1] == 0 here: the PIFA is only ever parsed for its HEADER, so
#     nothing loads the voxels.  Writing them in and marking the views valid is
#     what stops the "AreViewsAvailable not ready" spin (387 polls -> 0).
#     THIS IS ALSO THE INJECTION POINT FOR YOUR OWN MU-MAP -- see ct_to_pifa.py.
#
#  4. nThreads = 1.  CRendezvous(int)/Join() is a counted barrier (CPrep3d ctor
#     +289 builds it with nThreads) and only this one thread drives DoTask.
#     With 6 the run freezes on a Join with utime stopped dead.
banner("randoms + scatter: GE's own prep stage")
show("$ig->InitRandomsFromSingles()")
show("IgJobReq.attenuationFlag")

banner("ValidateCTAC: forge TransSysGeometry from the emission geometry")
_emis = bytes(inf.read_memory(0xf2f6a0, 2048))
inf.write_memory(0xf2fea0, _emis)
show("TransSysGeometry.radialModulesPerSystem")

banner("inject the mu-map into CRawDataMem::m_pAttn")
# The PIFA path is whatever the job points at, so a real-case PIFA built by
# ct_to_pifa.py works here with no edit to this script.
_p = val("IgJobReq.inputTransmissionFileName[0]")
PIFA = _p.string() if _p is not None else ""
say("   PIFA = %s\n" % PIFA)
nviews = num("$rd->m_pAttn->m_pCyclicMemBuff->m_uiAllocElement")
vsize  = num("$rd->m_pAttn->m_pCyclicMemBuff->m_uiElementSize")
adst   = int(val("$rd->m_pAttn->m_pCyclicMemBuff->m_pStartBuffer"))
mu = b""
try:
    with open(PIFA, "rb") as f:
        f.seek(num("transmissionCTACHeader[0].offsetToStartOfImage") or 164)
        mu = f.read(nviews * vsize)
except Exception as e:
    say("!! cannot read the PIFA: %s\n" % e)
if len(mu) == nviews * vsize:
    inf.write_memory(adst, mu)
    show("$rd->m_pAttn->InsertedNewViews(0, %d)" % nviews)
    show("$rd->m_pAttn->SetViewsAsValid(0, %d)" % nviews)
    show("$rd->m_pAttn->AreViewsAvailable(0, %d)" % nviews)
else:
    say("!! mu-map is %d bytes, need %d (%d views x %d) -- scatter will stall\n"
        % (len(mu), nviews * vsize, nviews, vsize))

banner("contexts at nThreads = 1")
for nm_, size, expr in (
        ("$ctac", 344, "((int (*)(void *, void *)) 0x441510)($ctac, $ig)"),
        ("$scat", 864, "((int (*)(void *, void *, int, char)) 0x48e070)($scat, $ig, 1, (char) 0)"),
        ("$prep", 896, "((int (*)(void *, void *, int, void *, void *)) 0x482500)"
                       "($prep, $ig, 1, $ctac, $scat)")):
    setv("set %s = (char *) malloc(%d)" % (nm_, size))
    setv("set $x = (int) memset(%s, 0, %d)" % (nm_, size))
    show(expr)
show("((int (*)(void *)) 0x482290)($prep)")    # COsem3dPrep::Initialize
show("((int (*)(void *)) 0x434de0)($prep)")    # CPrep3d::CreateTaskList

banner("DoTask -- all %d views" % NVIEW)
with unlocked():
    # Ends in a SIGSEGV inside CScatterFully3dModel::GetScatterCounts
    # (ScatterFully3dModel.cpp:5025) AFTER every view is written, so the
    # failure is expected; unwindonsignal keeps it from poisoning the run.
    show("((int (*)(void *)) 0x435e70)($prep)")

for b in ("m_pPrompts", "m_pRandoms", "m_pScatter", "m_pAttn"):
    show("$cd->%s->m_pCyclicMemBuff->m_uiCounterElement" % b)

dump_views("$cd->m_pRandoms", "randoms.f32", "f4",
           dict(GEOM, produced_by="COsem3dPrep::DoPrep",
                note="pre-estimate: finished before CreateReconContext; OSEM only reads it"))
dump_views("$cd->m_pScatter", "scatter.f32", "f4",
           dict(GEOM, produced_by="CScatterFully3dModel (SSS) via CPrep3d::DoTask",
                note="from the mu-map injected into m_pAttn"))

banner("DONE")
end
