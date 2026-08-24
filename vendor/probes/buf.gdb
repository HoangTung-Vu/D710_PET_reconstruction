source /vendor/boot.gdb
set var sharcCmpDebugFlag = 0
source /vendor/job.gdb
print ((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)
set $ig = (CIgManager *) $ig
set $rd = $ig->m_pRawDataMem._M_ptr
print $rd->Initialize()
echo \n=== buffs BEFORE ===\n
print $rd->GetPromptsBuff()
echo \n=== ReallocateBuffs(false) ===\n
print $rd->ReallocateBuffs(0, $ig)
print $rd->GetPromptsBuff()
echo \n=== ReallocateBuffs(true) ===\n
print $rd->ReallocateBuffs(1, $ig)
print $rd->GetPromptsBuff()
echo \n=== geometry the alloc depends on ===\n
print $ig->m_pRawDataMem._M_ptr
