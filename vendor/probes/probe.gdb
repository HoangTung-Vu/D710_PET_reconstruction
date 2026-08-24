source /vendor/boot.gdb
set var sharcCmpDebugFlag = 0x84
source /vendor/job.gdb
echo \n=== openDataFiles ===\n
print ((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)
echo \n=== state before load ===\n
print m_pRawDataMem
print apCfg.norm3dGeometryPeriod
print sysGeometry.radialCrystalsPerBlock
echo \n=== AP-side remote load (no FIFO handshake) ===\n
set $ld = (char *) malloc(262144)
print (void *) memset($ld, 0, 262144)
print ((int (*)(void *, void *)) sharcCmp3dRemoteLoad)(&IgJobReq, $ld)
echo \n=== state after load ===\n
print m_pRawDataMem
