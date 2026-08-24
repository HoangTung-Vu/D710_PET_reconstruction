# Randoms (and scatter) by running GE's own per-view prep stage.
#
# COsem3dPrep::DoPrep(view, ...) is where the 3D randoms sinogram is actually
# written -- GetCorrRandomsBuff(view) -> Preprocess(...) ->
# SetCorrRandomsBuffValid(view).  Nothing else in the binary fills
# CCorrDataMem::m_pRandoms for the 3D path: CIgManager::InitRandomsFromSingles
# only calls CalculateRFS, which computes the randoms-from-singles PARAMETERS
# (GE_meanvx + initRFSParams) and no sinogram at all.  That is why the earlier
# InitRandomsFromSingles run returned GRE_IG_SUCCESS with an all-zero buffer.
#
# The same DoPrep also does GetCorrNormDTBuff -> GE_vfill -> ApplyNormalization
# -> ApplyDeadtime, which is independent confirmation that normdt.gdb's
# unit-sinogram recipe is the vendor's own.
#
# Do3dEmissionImage builds it as
#   COsem3dPrep(ig, m_pThreadPool->nThreads, CCTAC_3D*, CScatter*)
# so both of those have to exist first.  This script only probes how far the
# construction gets; nothing here is load-bearing yet.
source /vendor/boot.gdb
source /vendor/lib.gdb
set var sharcCmpDebugFlag = 0x4
set var PrintTraceFlag = 1
source /vendor/job.gdb

python
banner("open + allocate + load (the proven prologue)")
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
with unlocked():
    show("$ig->LoadRawData()")
    show("$ig->InitLuts()")
    gdb.execute("set var *(int *) (0xf38968 + 8) = 0")
    show("((int (*)(void *, void *)) sharcCmp3dRemoteSinglesLoad)(&IgJobReq, (void *) 0xf38968)")
    gdb.execute("set var *(int *) (0xf38980 + 8) = 0")
    show("((int (*)(void *, void *)) sharcCmp3dRemoteDeadtimeLoad)(&IgJobReq, (void *) 0xf38980)")
    show("((int (*)(void *, char)) setup3dEmissJob)(&IgJobReq, (char) 0)")

banner("randoms-from-singles parameters")
show("$ig->InitRandomsFromSingles()")

banner("thread count the prep context is built with")
show("$ig->GetNumOfProcessingThreads()")

banner("CCTAC_3D(ig)")
gdb.execute("set $ctac = (char *) malloc(344)")
gdb.execute("set $x = (int) memset($ctac, 0, 344)")
show("((int (*)(void *, void *)) 0x441510)($ctac, $ig)")

banner("CScatterFully3dModel(ig, nthreads, false)")
gdb.execute("set $scat = (char *) malloc(864)")
gdb.execute("set $x = (int) memset($scat, 0, 864)")
show("((int (*)(void *, void *, int, char)) 0x48e070)($scat, $ig, 6, (char) 0)")

banner("COsem3dPrep(ig, nthreads, ctac, scatter)")
gdb.execute("set $prep = (char *) malloc(896)")
gdb.execute("set $x = (int) memset($prep, 0, 896)")
show("((int (*)(void *, void *, int, void *, void *)) 0x482500)($prep, $ig, 6, $ctac, $scat)")

banner("drive the context by hand: Initialize / CreateTaskList / DoTask")
# CContext is polymorphic; go through the vtable so the COsem3dPrep overrides
# run.  Slot order comes from CContext's declaration:
#   Initialize, CreateTaskList, DoTask, GetConstStatus, Finalize
show("*(void **) $prep")
ex("info symbol *(void **) $prep")
ex("x/8a *(void **) $prep")

banner("DONE")
end
