# Exact prologue of tmp/normdt.gdb, but stopping in the fault so it can be read.
source /vendor/boot.gdb
source /vendor/lib.gdb
set unwindonsignal off
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
gdb.execute("set $ps = $ig->m_pParamStruct._M_ptr")
for c in ("$ig->CleanDataBuffers(0)", "$ig->InitParamStruct()",
          "$rd->DeallocateBuffs(0, $ig)", "$cd->DeallocateBuffs(0, $ig)",
          "$rd->ReallocateBuffs(0, $ig)"):
    show(c)
banner("post-mortem")
ex("bt 20")
ex("info registers rip rsp rbp rdi rsi rdx rcx rax")
ex("x/6i $pc")
ex("info threads")
ex("p $_siginfo._sifields._sigfault.si_addr")
banner("DONE")
end
