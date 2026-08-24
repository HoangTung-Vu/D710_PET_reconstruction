# Does LoadRawData() survive now that the AP globals are published?
# Sequence copied from CIgManager::RunCyclic:
#   InitParamStruct -> ReallocateBuffs(raw, corr, image; all with bool=false)
#   -> LoadRawData -> DoProcessing.
source /vendor/boot.gdb
source /vendor/lib.gdb
set var sharcCmpDebugFlag = 0x4
source /vendor/job.gdb

python
banner("AP globals (set by the CPCCommThread stub)")
show("*(void **) 0xf38948")
show("*(void **) 0xf38950")
show("*(void **) 0xf38958")

banner("openDataFiles")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $cd = $ig->m_pCorrDataMem._M_ptr")
gdb.execute("set $im = $ig->m_pImageDataMem._M_ptr")

banner("InitParamStruct")
show("$ig->InitParamStruct()")
for f in ("number_u", "rr_number_u", "number_theta", "number_v_theta",
          "number_phi", "numTOF_bins"):
    showdeep("$ig->m_pParamStruct._M_ptr", "rawData", f)
for f in ("crystalsPerRing", "numberMajorRings", "numberProjections",
          "numberSamples", "axialCrystals", "blocksPerRing"):
    showdeep("$ig->m_pParamStruct._M_ptr", f)

banner("ReallocateBuffs -- raw, corr, image (bool=false, as RunCyclic does)")
show("$rd->ReallocateBuffs(0, $ig)")
show("$cd->ReallocateBuffs(0, $ig)")
show("$im->ReallocateBuffs(0, $ig)")
show("$cd->m_uiBufferSize")
show("$cd->m_pRandoms")
show("$cd->m_pScatter")
show("$cd->m_pDeadtimeNormWCCBuff")
show("$cd->m_pNormDT")
show("$cd->m_pGeomFactors")

banner("LoadRawData")
show("$ig->LoadRawData()")
show("$rd->m_pSingles")
show("$rd->m_pPrompts")
show("*$rd->m_pSingles@16")

banner("DONE")
end
