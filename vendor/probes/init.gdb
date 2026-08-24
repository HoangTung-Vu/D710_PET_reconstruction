# m_uiBufferSize stayed 0 through both ReallocateBuffs flags, so the flag was
# never the problem: the geometry the allocator sizes from is still zero.
# GRE_IG_PARAMETER_STRUCT is filled by CIgManager::InitParamStruct(), which
# nothing has called -- openDataFiles only fills apCfg/sysGeometry.
source /vendor/boot.gdb
source /vendor/lib.gdb
set var sharcCmpDebugFlag = 0x84
source /vendor/job.gdb

python
banner("openDataFiles")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $ps = $ig->m_pParamStruct._M_ptr")

GEOM = ("rawData", "crystalsPerRing", "numberMajorRings", "numberProjections",
        "numberSamples", "axialCrystals", "blocksPerRing", "blocksPerSystem")

banner("param struct BEFORE InitParamStruct")
for f in GEOM:
    show("$ps->" + f)

banner("InitParamStruct")
show("$ig->InitParamStruct()")
for f in GEOM:
    show("$ps->" + f)
show("$ps->completeData")
show("$ps->numTOFbins" )

banner("InitLuts")
show("$ig->InitLuts()")
show("$rd->m_pLutX1")
show("$rd->m_pLutZ1")

banner("ReallocateBuffs after the geometry exists")
show("$rd->Initialize()")
show("$rd->ReallocateBuffs(0, $ig)")
show("$rd->m_uiBufferSize")
show("$rd->m_uiPromptsBinSize")
show("$rd->m_pPrompts")
show("$rd->m_pSingles")
show("*$rd->m_pPrompts")

banner("CCorrDataMem -- destination of randoms/scatter")
ex("ptype /o CCorrDataMem")

banner("CViewInterface / CCyclicMemBuffer")
ex("ptype CViewInterface")
ex("ptype /o CCyclicMemBuffer")

banner("DONE")
end
