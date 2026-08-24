# Methods + live buffer state.  GetPromptsBuff() is inlined so it cannot be
# called; m_pPrompts is read directly (offset 0 of CRawDataMem).
source /vendor/boot.gdb
set var sharcCmpDebugFlag = 0
source /vendor/job.gdb

set logging file /out/explore.txt
set logging overwrite on
set logging redirect on
set logging enabled on

echo === openDataFiles ===\n
print ((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)
set $ig = (CIgManager *) $ig
set $rd = $ig->m_pRawDataMem._M_ptr
set $ps = $ig->m_pParamStruct._M_ptr

echo \n=== METHODS CRawDataMem ===\n
ptype CRawDataMem
echo \n=== METHODS CIgManager ===\n
ptype CIgManager
echo \n=== CViewBuffer ===\n
ptype /o CViewBuffer

echo \n=== rawData geometry (drives buffer size) ===\n
print $ps->rawData
print $ps->completeData
print $ps->crystalsPerRing
print $ps->numberMajorRings
print $ps->numberProjections
print $ps->numberSamples
print $ps->axialCrystals
print $ps->blocksPerRing
print $ps->blocksPerSystem

echo \n=== buffers BEFORE Initialize ===\n
print $rd->m_pPrompts
print $rd->m_pDelays
print $rd->m_pSingles
print $rd->m_pNorm
print $rd->m_pDeadtimeStruct
print $rd->m_uiPromptsBinSize
print $rd->m_uiBufferSize

echo \n=== Initialize ===\n
print $rd->Initialize()
print $rd->m_pPrompts
print $rd->m_pSingles
print $rd->m_uiBufferSize

echo \n=== ReallocateBuffs(0) ===\n
print $rd->ReallocateBuffs(0, $ig)
print $rd->m_pPrompts
print $rd->m_pDelays
print $rd->m_pSingles
print $rd->m_pNorm
print $rd->m_pAttn
print $rd->m_uiPromptsBinSize
print $rd->m_uiBufferSize

echo \n=== ReallocateBuffs(1) ===\n
print $rd->ReallocateBuffs(1, $ig)
print $rd->m_pPrompts
print $rd->m_pDelays
print $rd->m_pSingles
print $rd->m_pNorm
print $rd->m_pAttn
print $rd->m_uiPromptsBinSize
print $rd->m_uiBufferSize

set logging enabled off
echo \n=== DONE ===\n
