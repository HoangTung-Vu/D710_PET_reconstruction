# tofscat.gdb -- does IgJobReq.reconMethod = 3 turn on GE's TOF scatter?
#
#   D710_JOB=/out/job.gdb ../run.sh --out <dir> --data <bed>/data probes/tofscat.gdb
#
# The claim being tested, read out of pet_recon's own DWARF and disassembly:
#
#   CIgManager::Do3dEmissionImage  (0x42a6fc)
#       cmpl $0x3, IgJobReq+0x59c        <- reconMethod
#       sete %cl
#       call CScatterFully3dModel::CScatterFully3dModel(this, ig, nThreads, cl)
#
#   CScatterFully3dModel::CScatterFully3dModel  (0x48e184)
#       movzbl 0x17(%rsp),%eax           <- that same bool
#       mov    %al, 0x2d8(%rbp)          <- m_bTOFDim
#
#   CScatterFully3dModel::CreateTaskList  gates five task types on that byte:
#       MSCAT_CREATE_IMAGE_PATHS_TOF (20)  MSCAT_CALC_SCAT_ESTIMATE_TOF (21)
#       MSCAT_COMBINE_DS_SINGLE_SCAT_SINO_TOF (22)
#       MSCAT_PHI_UPSAMPLE_TOF_SCAT (23)   MSCAT_CONVERT_PHIUP_SCAT_TO_3D (24)
#
# extract.gdb constructs that object by hand and passes (char) 0 for the bool,
# which is why every scatter this project has ever produced is non-TOF.  Here it
# is passed 1 instead, and the destination buffer -- CCorrDataMem::m_pScatterTOF,
# already allocated on every run at 288 views x 151360 B -- is dumped after
# DoTask so we can see whether GE actually wrote anything into it.
#
# The norm / dead-time sweeps of extract.gdb are left out: they are most of the
# wall clock and have nothing to do with the question.
source /vendor/boot.gdb
source /vendor/lib.gdb

set var sharcCmpDebugFlag = 0x4
set var PrintTraceFlag = 1

python
import os
JOB = os.environ.get("D710_JOB", "/vendor/job.gdb")
banner("job: %s" % JOB)
gdb.execute("source " + JOB)

# The one change under test.  Everything else in the job stays the vendor's.
banner("reconMethod 2 -> 3 (TOF)")
show("IgJobReq.reconMethod")
gdb.execute("set var IgJobReq.reconMethod = 3")
show("IgJobReq.reconMethod")

banner("open data files + allocate buffers")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $cd = $ig->m_pCorrDataMem._M_ptr")
gdb.execute("set $im = $ig->m_pImageDataMem._M_ptr")

for c in ("$ig->CleanDataBuffers(0)", "$ig->InitParamStruct()",
          "$rd->DeallocateBuffs(0, $ig)", "$cd->DeallocateBuffs(0, $ig)",
          "$rd->ReallocateBuffs(0, $ig)", "$cd->ReallocateBuffs(0, $ig)",
          "$im->ReallocateBuffs(0, $ig)"):
    show(c)

banner("geometry")
NU    = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_u")       or 0)
NV    = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_v_theta") or 0)
NVIEW = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_phi")     or 0)
NTOF  = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "numTOF_bins")    or 0)
DSNU  = int(deep("$ig->m_pParamStruct._M_ptr", "scatterData", "ds_nu")      or 0)
say("   number_u=%d number_v_theta=%d number_phi=%d numTOF_bins=%d ds_nu=%d\n"
    % (NU, NV, NVIEW, NTOF, DSNU))
GEOM = dict(number_u=NU, number_v_theta=NV, number_phi=NVIEW, numTOF_bins=NTOF,
            ds_nu=DSNU)

banner("load prompts / singles / dead time / norm")
with unlocked():
    show("$ig->LoadRawData()")
    show("$ig->InitLuts()")
    gdb.execute("set var *(int *) (0xf38968 + 8) = 0")
    show("((int (*)(void *, void *)) sharcCmp3dRemoteSinglesLoad)(&IgJobReq, (void *) 0xf38968)")
    gdb.execute("set var *(int *) (0xf38980 + 8) = 0")
    show("((int (*)(void *, void *)) sharcCmp3dRemoteDeadtimeLoad)(&IgJobReq, (void *) 0xf38980)")
    show("((int (*)(void *, char)) setup3dEmissJob)(&IgJobReq, (char) 0)")

banner("randoms + scatter: GE's own prep stage")
show("$ig->InitRandomsFromSingles()")

banner("ValidateCTAC: forge TransSysGeometry from the emission geometry")
import struct as _s
inf = gdb.selected_inferior()
_emis = bytes(inf.read_memory(0xf2f6a0, 2048))
inf.write_memory(0xf2fea0, _emis)

banner("inject the mu-map into CRawDataMem::m_pAttn")
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
else:
    say("!! mu-map is %d bytes, need %d -- scatter will stall\n"
        % (len(mu), nviews * vsize))

# ------------------------------------------------------------- the test
# Same three contexts extract.gdb builds, EXCEPT the scatter model's third
# argument.  extract.gdb passes 0; this passes 1.
banner("contexts at nThreads = 1, scatter model with bTOFDim = 1")
for nm_, size, expr in (
        ("$ctac", 344, "((int (*)(void *, void *)) 0x441510)($ctac, $ig)"),
        ("$scat", 864, "((int (*)(void *, void *, int, char)) 0x48e070)($scat, $ig, 1, (char) 1)"),
        ("$prep", 896, "((int (*)(void *, void *, int, void *, void *)) 0x482500)"
                       "($prep, $ig, 1, $ctac, $scat)")):
    setv("set %s = (char *) malloc(%d)" % (nm_, size))
    setv("set $x = (int) memset(%s, 0, %d)" % (nm_, size))
    show(expr)

# The byte the whole thing turns on.  If this reads 0 the ctor argument did not
# land where the disassembly says it does and nothing below means anything.
banner("m_bTOFDim (CScatterFully3dModel + 0x2d8)")
show("*(unsigned char *) ($scat + 0x2d8)")

show("((int (*)(void *)) 0x482290)($prep)")    # COsem3dPrep::Initialize
show("((int (*)(void *)) 0x434de0)($prep)")    # CPrep3d::CreateTaskList

# CreateTaskList has run by now, so the task list records which branch was
# taken.  m_scatTaskList is at CScatterFully3dModel + 0x168; walk it and count
# how many of the five TOF task types are in there.
banner("what CreateTaskList queued")
TASK = ["RESORT_EMISS_3D_2D", "CREATE_MU_IMG", "MASK_TABLE_MU_IMG",
        "CALC_SINO_TAILS", "SORT_SINO_TAILS_2D", "SMOOTH_TAIL_PARAMS",
        "DOWNSAMPLE_MU_IMG", "CREATE_IMAGE_PATHS", "CREATE_EMISS_IMG",
        "DOWNSAMPLE_EMIS_IMG", "CALC_SCAT_ESTIMATE",
        "COMBINE_DS_SINGLE_SCAT_SINO", "UPSAMPLE_SCAT",
        "CLEAR_BUFFS_FOR_NEXT_ITER", "SYNC_BEFORE_NEXT_ITER",
        "CLEAR_SYNC_BUFF", "CONVERT_SCAT_TO_3D", "TAILFIT_SCAT_3D",
        "REORDER_IMG_PATHS", "TAILSCALE_SCAT_3D", "CREATE_IMAGE_PATHS_TOF",
        "CALC_SCAT_ESTIMATE_TOF", "COMBINE_DS_SINGLE_SCAT_SINO_TOF",
        "PHI_UPSAMPLE_TOF_SCAT", "CONVERT_PHIUP_SCAT_TO_3D"]
try:
    # std::list<MODEL_SCAT_TASK_TYPE>: the head node's next chain, value at +16.
    head = int(val("$scat")) + 0x168
    node = int(inf.read_memory(head, 8).tobytes()[:8].hex(), 16) if False else \
           _s.unpack("<Q", bytes(inf.read_memory(head, 8)))[0]
    counts, n = {}, 0
    while node and node != head and n < 20000:
        t = _s.unpack("<I", bytes(inf.read_memory(node + 16, 4)))[0]
        counts[t] = counts.get(t, 0) + 1
        node = _s.unpack("<Q", bytes(inf.read_memory(node, 8)))[0]
        n += 1
    for t in sorted(counts):
        say("   %-38s x %d\n" % (TASK[t] if t < len(TASK) else "?%d" % t, counts[t]))
    say("   %d tasks total\n" % n)
except Exception as e:
    say("!! cannot walk m_scatTaskList: %s\n" % e)

banner("DoTask -- all %d views" % NVIEW)
with unlocked():
    show("((int (*)(void *)) 0x435e70)($prep)")

for b in ("m_pPrompts", "m_pRandoms", "m_pScatter", "m_pScatterTOF"):
    show("$cd->%s->m_pCyclicMemBuff->m_uiCounterElement" % b)

banner("dump")
dump_views("$cd->m_pScatter", "scatter.f32", "f4",
           dict(GEOM, axes="view(%d) x v(%d) x u(%d)" % (NVIEW, NV, NU),
                produced_by="CScatterFully3dModel with m_bTOFDim = 1"))
dump_views("$cd->m_pScatterTOF", "scatter_tof.f32", "f4",
           dict(GEOM, produced_by="CScatterFully3dModel with m_bTOFDim = 1",
                note="compact TOF scatter; per view ds_nu x 16 x numTOF_bins"))
dump_views("$cd->m_pRandoms", "randoms.f32", "f4",
           dict(GEOM, axes="view(%d) x v(%d) x u(%d)" % (NVIEW, NV, NU)))

banner("DONE")
end
