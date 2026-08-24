# Why does CRawDataMem::ReallocateBuffs SIGSEGV inside libc?
# unwindonsignal off so gdb stays in the faulting frame and can be asked.
source /vendor/boot.gdb
source /vendor/lib.gdb
set unwindonsignal off
set var sharcCmpDebugFlag = 0x4
set var PrintTraceFlag = 1
source /vendor/job.gdb

python
banner("setup")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $cd = $ig->m_pCorrDataMem._M_ptr")
show("$ig->CleanDataBuffers(0)")
show("$ig->InitParamStruct()")
show("$rd->DeallocateBuffs(0, $ig)")
show("$cd->DeallocateBuffs(0, $ig)")   # <- the suspect

banner("state going into ReallocateBuffs")
show("$rd->m_pPrompts")
show("$rd->m_pPrompts->m_pCyclicMemBuff")
show("$rd->m_pPrompts->m_pCyclicMemBuff->m_uiAllocElement")
show("$rd->m_pPrompts->m_bValidData")
show("$ig->m_pParamStruct._M_ptr")
#show("emissionSorterData")

banner("the call")
show("$rd->ReallocateBuffs(0, $ig)")

banner("where did it die")
ex("bt 15")
ex("info registers rip rax rdi rsi rdx rcx")
ex("x/4i $pc")
ex("info threads")
banner("DONE")
end
