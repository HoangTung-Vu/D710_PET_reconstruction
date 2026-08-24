# Where do the correction results actually live, and what has to be allocated
# before LoadRawData() stops segfaulting?
source /vendor/boot.gdb
source /vendor/lib.gdb
set var sharcCmpDebugFlag = 0
source /vendor/job.gdb

python
banner("openDataFiles")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $cd = $ig->m_pCorrDataMem._M_ptr")

banner("param struct -- how to reach it")
for e in ("$ig->m_pParamStruct",
          "$ig->m_pParamStruct._M_ptr",
          "*$ig->m_pParamStruct._M_ptr"):
    gdb.write("   whatis %-40s %s" % (e, ex("whatis " + e, quiet=True) or "  <fail>\n"))
show("$ig->m_pParamStruct._M_ptr")

banner("CCorrDataMem layout  -> where randoms/scatter/norm land")
ex("ptype /o CCorrDataMem")

banner("CCyclicMemBuffer / CViewInterface  -> how to read a CViewBuffer")
ex("ptype /o CCyclicMemBuffer")
ex("ptype CViewInterface")

banner("pointers before any alloc")
for f in ("m_pPrompts", "m_pDelays", "m_pSingles", "m_pNorm", "m_pAttn",
          "m_pDeadtimeStruct", "m_pIntDeadtime", "m_pMuxDeadtime",
          "m_uiPromptsBinSize", "m_uiBufferSize"):
    show("$rd->" + f)
show("$cd")

banner("Initialize")
show("$rd->Initialize()")

banner("ReallocateBuffs(0)")
show("$rd->ReallocateBuffs(0, $ig)")
for f in ("m_pPrompts", "m_pDelays", "m_pSingles", "m_pNorm", "m_pAttn",
          "m_uiPromptsBinSize", "m_uiBufferSize"):
    show("$rd->" + f)

banner("ReallocateBuffs(1)")
show("$rd->ReallocateBuffs(1, $ig)")
for f in ("m_pPrompts", "m_pDelays", "m_pSingles", "m_pNorm", "m_pAttn",
          "m_uiPromptsBinSize", "m_uiBufferSize"):
    show("$rd->" + f)

banner("what ReallocateBuffs reads -- source, if the CU has it")
ex("list CRawDataMem::ReallocateBuffs")
ex("info line CRawDataMem::ReallocateBuffs")

banner("DONE")
end
