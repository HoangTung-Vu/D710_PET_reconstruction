source /vendor/boot.gdb
set var sharcCmpDebugFlag = 0x1FF
source /vendor/job.gdb
print ((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)
set $ig = (CIgManager *) $ig
set $rd = $ig->m_pRawDataMem._M_ptr
print $rd
echo \n=== CRawDataMem::Initialize ===\n
print $rd->Initialize()
echo \n=== ReallocateBuffs ===\n
print $rd->ReallocateBuffs(0, $ig)
echo \n=== LoadRawData ===\n
print $ig->LoadRawData()
echo \n=== InitRandomsFromSingles ===\n
print $ig->InitRandomsFromSingles()
echo \n=== Do3dNormProcessing ===\n
print $ig->Do3dNormProcessing()
echo \n=== DONE ===\n
