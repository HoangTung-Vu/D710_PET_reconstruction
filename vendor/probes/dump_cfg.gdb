source /vendor/boot.gdb
source /vendor/job.gdb
print ((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)
set logging file /out/apcfg.txt
set logging overwrite on
set logging redirect on
set logging enabled on
echo === apCfg ===\n
print apCfg
echo === sysGeometry ===\n
print sysGeometry
echo === CNorm3d/CDeadtime3d owner search ===\n
print m_pRawDataMem
set logging enabled off
