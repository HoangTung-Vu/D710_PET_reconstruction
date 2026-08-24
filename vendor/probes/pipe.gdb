# The sequence CIgManager::RunCyclic actually follows, replayed by hand.
#
# The step that was missing all along is DeallocateBuffs.  ReallocateBuffs
# opens with `if (GetNumAllocViews() != 0) skip`, and a freshly constructed
# CViewBuffer already reports 1 element of 2 bytes -- so without a Deallocate
# first, every Reallocate returns GRE_IG_SUCCESS having allocated nothing.
# That is why CRawDataLoad could put view 0 and nothing else.
source /vendor/boot.gdb
source /vendor/lib.gdb
set var sharcCmpDebugFlag = 0x1FF
set var PrintTraceFlag = 1
source /vendor/job.gdb

python
banner("openDataFiles")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $cd = $ig->m_pCorrDataMem._M_ptr")
gdb.execute("set $im = $ig->m_pImageDataMem._M_ptr")

banner("CleanDataBuffers / InitParamStruct")
show("$ig->CleanDataBuffers(0)")
show("$ig->InitParamStruct()")

banner("DeallocateBuffs  <-- the step that was missing")
show("$rd->DeallocateBuffs(0, $ig)")
show("$cd->DeallocateBuffs(0, $ig)")
show("$rd->m_pPrompts->m_pCyclicMemBuff->m_uiAllocElement")

banner("ReallocateBuffs")
show("$rd->ReallocateBuffs(0, $ig)")
show("$cd->ReallocateBuffs(0, $ig)")
show("$im->ReallocateBuffs(0, $ig)")

def cyc(b):
    n  = num("%s->m_pCyclicMemBuff->m_uiAllocElement" % b)
    sz = num("%s->m_pCyclicMemBuff->m_uiElementSize" % b)
    st = val("%s->m_pCyclicMemBuff->m_pStartBuffer" % b)
    gdb.write("   %-28s views=%-6d viewSize=%-10d total=%-12d start=%s\n"
              % (b.split("->")[-1], n, sz, n * sz, st))

banner("view geometry after realloc")
for b in ("$rd->m_pPrompts", "$rd->m_pDelays", "$rd->m_pNorm", "$rd->m_pAttn",
          "$cd->m_pPrompts", "$cd->m_pRandoms", "$cd->m_pScatter",
          "$cd->m_pNormDT", "$cd->m_pDeadtimeNormWCCBuff"):
    cyc(b)

banner("LoadRawData")
show("$ig->LoadRawData()")
show("$ig->InitLuts()")

banner("DONE")
end
