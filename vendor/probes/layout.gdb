# Step 1 of §3b: find the prompts buffer field and see who is supposed to fill
# it.  GetPromptsBuff() is inlined, so gdb cannot call it; read the field.
source /vendor/boot.gdb
set var sharcCmpDebugFlag = 0
source /vendor/job.gdb

set logging file /out/layout.txt
set logging overwrite on
set logging redirect on
set logging enabled on

echo === openDataFiles ===\n
print ((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)

set $ig = (CIgManager *) $ig
set $rd = $ig->m_pRawDataMem._M_ptr
echo === CRawDataMem layout ===\n
ptype /o CRawDataMem
echo === CIgManager layout ===\n
ptype /o CIgManager
echo === GRE_IG_PARAMETER_STRUCT ===\n
ptype /o GRE_IG_PARAMETER_STRUCT

echo === fields BEFORE Initialize ===\n
print *$rd
echo === Initialize ===\n
print $rd->Initialize()
print *$rd
echo === ReallocateBuffs(0) ===\n
print $rd->ReallocateBuffs(0, $ig)
print *$rd
echo === ReallocateBuffs(1) ===\n
print $rd->ReallocateBuffs(1, $ig)
print *$rd

set logging enabled off
echo \n=== DONE ===\n
