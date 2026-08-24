# LoadRawData() no longer segfaults but returns GRE_IG_ERROR, and the reason
# goes to Ermes, which is dead here.  Catch it at the source instead:
# ReconProcessingException's ctor is handed the file and line of the failure.
source /vendor/boot.gdb
source /vendor/lib.gdb
set var sharcCmpDebugFlag = 0x1FF
# PrintTraceFlag gates the puts()/printf() trace all through cpc* and sharc*.
set var PrintTraceFlag = 1
source /vendor/job.gdb

# ReconProcessingException(ERROR_MESSAGES, char const* file, int line, char const* msg)
break ReconProcessingException::ReconProcessingException
commands
  silent
  printf "\n!!! ReconProcessingException  err=%d  %s:%d  msg=%s\n", $esi, $rdx, $ecx, $r8
  bt 12
  continue
end

python
banner("openDataFiles")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $cd = $ig->m_pCorrDataMem._M_ptr")
gdb.execute("set $im = $ig->m_pImageDataMem._M_ptr")
show("$ig->InitParamStruct()")
show("$rd->ReallocateBuffs(0, $ig)")
show("$cd->ReallocateBuffs(0, $ig)")
show("$im->ReallocateBuffs(0, $ig)")

banner("LoadRawData -- with tracing on")
show("$ig->LoadRawData()")

banner("DONE")
end
