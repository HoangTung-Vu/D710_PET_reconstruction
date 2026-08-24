# Run GE's per-view prep stage and take the randoms / scatter / normDT
# sinograms it writes.
#
# From tmp/prep.gdb: all three contexts construct cleanly, and COsem3dPrep's
# vtable resolves to
#     +0x10 COsem3dPrep::Initialize   0x482290
#     +0x18 CPrep3d::CreateTaskList   0x434de0
#     +0x20 CPrep3d::DoTask           0x435e70
# which is exactly what CThreadPool::RunContext drives.  Calling them straight
# avoids RunContext's std::string argument and its thread dispatch.
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

banner("construct the contexts")
for name, size, expr in (
        ("$ctac", 344, "((int (*)(void *, void *)) 0x441510)($ctac, $ig)"),
        ("$scat", 864, "((int (*)(void *, void *, int, char)) 0x48e070)($scat, $ig, 6, (char) 0)"),
        ("$prep", 896, "((int (*)(void *, void *, int, void *, void *)) 0x482500)"
                       "($prep, $ig, 6, $ctac, $scat)")):
    gdb.execute("set %s = (char *) malloc(%d)" % (name, size))
    gdb.execute("set $x = (int) memset(%s, 0, %d)" % (name, size))
    show(expr)

banner("Initialize / CreateTaskList")
show("((int (*)(void *)) 0x482290)($prep)")     # COsem3dPrep::Initialize
show("((int (*)(void *)) 0x434de0)($prep)")     # CPrep3d::CreateTaskList

banner("DoTask -- the per-view prep; needs the pool for the view semaphores")
with unlocked():
    for i in range(3):
        r = ex("print ((int (*)(void *)) 0x435e70)($prep)", quiet=True)
        say("   DoTask #%d -> %s" % (i, (r or "!! failed\n")))

banner("what the prep produced")
for b in ("$cd->m_pPrompts", "$cd->m_pRandoms", "$cd->m_pScatter",
          "$cd->m_pNormDT", "$cd->m_pDeadtimeNormWCCBuff", "$cd->m_pAttn"):
    show("%s->m_pCyclicMemBuff->m_uiAllocElement" % b)

dump_views("$cd->m_pRandoms", "randoms.f32", "f4",
           dict(GEOM, produced_by="COsem3dPrep::DoPrep via CPrep3d::DoTask"))
dump_views("$cd->m_pNormDT", "normdt_vendor.f32", "f4",
           dict(GEOM, produced_by="COsem3dPrep::DoPrep, GE's own NormDT buffer"))
dump_views("$cd->m_pScatter", "scatter.f32", "f4",
           dict(GEOM, produced_by="COsem3dPrep::DoPrep"))
dump_views("$cd->m_pPrompts", "corr_prompts.f32", "f4",
           dict(GEOM, produced_by="COsem3dPrep::DoPrep"))

banner("DONE")
end
