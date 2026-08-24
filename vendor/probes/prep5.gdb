# Randoms + scatter, by running GE's own contexts in the order the recon does.
#
# tmp/prep2.gdb got the scatter model to run in full -- MSCAT_CONVERT_SCAT_TO_3D,
# MSCAT_TAILFIT_SCAT_3D, MSCAT_TAILSCALE_SCAT_3D -- and the prep inserted 48
# views into CorrPrompts / CorrRandoms / CorrScatterPrompts.  It then spun
# forever on
#     Prep3d: AreViewsAvailable not ready Start: 0 Num: 47
#     CTAC task 6 Input: 0 47 Output: 0 47
# because CCTAC_3D is a SEPARATE context.  On the console the thread pool runs
# it alongside the prep; single-threaded here it has to be run to completion
# first, or the prep waits for views nobody is producing.
#
# Entry points, from each class's vtable (slots: Initialize, CreateTaskList,
# DoTask, GetConstStatus, Finalize):
#     CCTAC_3D              Init 0x43eef0  TaskList 0x441550  DoTask 0x441030
#     CScatterFully3dModel  Init 0x489a30  TaskList 0x48bba0  DoTask 0x48b5d0
#     COsem3dPrep           Init 0x482290  TaskList 0x434de0  DoTask 0x435e70
source /vendor/boot.gdb
source /vendor/lib.gdb
set var sharcCmpDebugFlag = 0x4
set var PrintTraceFlag = 1
source /vendor/job.gdb

python
banner("open + allocate")
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

NU    = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_u")       or 0)
NV    = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_v_theta") or 0)
NVIEW = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_phi")     or 0)
GEOM  = dict(number_u=NU, number_v_theta=NV, number_phi=NVIEW,
             axes="view(%d) x v(%d) x u(%d)" % (NVIEW, NV, NU))
say("   geometry %s\n" % (GEOM["axes"],))

banner("loads")
with unlocked():
    show("$ig->LoadRawData()")
    show("$ig->InitLuts()")
    gdb.execute("set var *(int *) (0xf38968 + 8) = 0")
    show("((int (*)(void *, void *)) sharcCmp3dRemoteSinglesLoad)(&IgJobReq, (void *) 0xf38968)")
    gdb.execute("set var *(int *) (0xf38980 + 8) = 0")
    show("((int (*)(void *, void *)) sharcCmp3dRemoteDeadtimeLoad)(&IgJobReq, (void *) 0xf38980)")
    show("((int (*)(void *, char)) setup3dEmissJob)(&IgJobReq, (char) 0)")
show("$ig->InitRandomsFromSingles()")

# The attenuation views never become available: CPrep3d waits on
#   CCTAC_3D:: AreViewsAvailable not ready Start: 0 Num: 47
# and CCTAC_3D::DoTask itself SIGSEGVs at CTAC3D.cpp:320 because the CTAC
# image set was never loaded -- that is a separate path from the PIFA that
# sharcCmpOpenDataFiles reads.  Attenuation is not one of the four things
# wanted here, so switch it off and let the prep skip the CTAC branch.
gdb.execute("set var IgJobReq.attenuationFlag = 0")
gdb.execute("set var IgJobReq.fCTACFlag = 0")
show("IgJobReq.attenuationFlag")

banner("construct the contexts")
for name, size, expr in (
        ("$ctac", 344, "((int (*)(void *, void *)) 0x441510)($ctac, $ig)"),
        ("$scat", 864, "((int (*)(void *, void *, int, char)) 0x48e070)($scat, $ig, 6, (char) 0)"),
        ("$prep", 896, "((int (*)(void *, void *, int, void *, void *)) 0x482500)"
                       "($prep, $ig, 6, $ctac, $scat)")):
    gdb.execute("set %s = (char *) malloc(%d)" % (name, size))
    gdb.execute("set $x = (int) memset(%s, 0, %d)" % (name, size))
    show(expr)

def run_context(label, obj, init, tasklist, dotask, rounds):
    banner("run %s" % label)
    show("((int (*)(void *)) %s)(%s)" % (init, obj))
    show("((int (*)(void *)) %s)(%s)" % (tasklist, obj))
    with unlocked():
        for i in range(rounds):
            r = ex("print ((int (*)(void *)) %s)(%s)" % (dotask, obj), quiet=True)
            say("   %s DoTask #%d -> %s" % (label, i, r or "!! failed\n"))
            if r is None:
                break

# DoTask stalls at view 28: the prep is a pipeline stage and blocks waiting for
# OSEM to drain its buffers, and OSEM cannot run (no GPU).  A breakpoint on
# DoPrep with `ignore` set makes gdb abandon the inferior call at a chosen view
# and hand control back -- "The program being debugged stopped while in a
# function called from GDB" is exactly what is wanted here.  Everything already
# written to the corr buffers stays written.
banner("arm a stop after the first block of views")
# DoPrep is entered once per view; the stall is inside the 29th entry (view 28),
# so fire at hit 21 = view 20, leaving views 0..19 fully written.
# $bpnum does not survive gdb.execute() from Python: it expanded to nothing, the
# ignore count was never set, and the run sailed past into the stall.  Read the
# number back out of the `break` output instead.
_out = ex("break CPrep3d::DoPrep") or ""
import re as _re
_m = _re.search(r"Breakpoint (\d+)", _out)
if not _m:
    raise gdb.GdbError("cannot read breakpoint number from %r" % _out)
ex("ignore %s 20" % _m.group(1))

run_context("COsem3dPrep", "$prep", "0x482290", "0x434de0", "0x435e70", 1)
ex("info breakpoints")

banner("what the prep produced")
for b in ("$cd->m_pPrompts", "$cd->m_pRandoms", "$cd->m_pScatter",
          "$cd->m_pNormDT", "$cd->m_pAttn"):
    show("%s->m_pCyclicMemBuff->m_uiCounterElement" % b)

dump_views("$cd->m_pRandoms", "randoms.f32", "f4",
           dict(GEOM, produced_by="COsem3dPrep::DoPrep via CPrep3d::DoTask"))
dump_views("$cd->m_pScatter", "scatter.f32", "f4",
           dict(GEOM, produced_by="CScatterFully3dModel via COsem3dPrep::DoPrep"))
dump_views("$cd->m_pNormDT", "normdt_vendor.f32", "f4",
           dict(GEOM, produced_by="COsem3dPrep::DoPrep, GE's own NormDT buffer"))
dump_views("$cd->m_pPrompts", "corr_prompts.f32", "f4",
           dict(GEOM, produced_by="COsem3dPrep::DoPrep"))

banner("DONE")
end
