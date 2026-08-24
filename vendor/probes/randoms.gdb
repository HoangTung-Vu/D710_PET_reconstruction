# Randoms, from GE's own kernel.
# InitRandomsFromSingles() -> CalculateRFS() -> CCorrDataMem::m_pRandoms.
source /vendor/boot.gdb
source /vendor/lib.gdb
set var sharcCmpDebugFlag = 0x1FF
set var PrintTraceFlag = 1
source /vendor/job.gdb

python
banner("setup")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $cd = $ig->m_pCorrDataMem._M_ptr")
gdb.execute("set $im = $ig->m_pImageDataMem._M_ptr")
for c in ("$ig->CleanDataBuffers(0)", "$ig->InitParamStruct()",
          "$rd->DeallocateBuffs(0, $ig)", "$cd->DeallocateBuffs(0, $ig)",
          "$rd->ReallocateBuffs(0, $ig)", "$cd->ReallocateBuffs(0, $ig)",
          "$im->ReallocateBuffs(0, $ig)",
          "$ig->LoadRawData()", "$ig->InitLuts()"):
    show(c)

banner("singles that randoms are derived from")
show("$rd->m_pSingles")
show("*$rd->m_pSingles@24")
dump_expr("$rd->m_pSingles", 576 * 24 * 4, "singles.i32",
          {"dtype": "i4", "layout": "crystalsPerRing=576 x axialCrystals=24",
           "note": "per-crystal singles, loaded by LoadRawData"})

banner("InitRandomsFromSingles")
show("$ig->InitRandomsFromSingles()")

banner("dump randoms")
dump_views("$cd->m_pRandoms", "randoms.f32", "f4",
           {"axes": "view(288) x v(553) x u(381)",
            "produced_by": "CIgManager::InitRandomsFromSingles -> CalculateRFS"})

banner("vendor's own dumps (CWD = /out)")
show("$ig->WriteCorrData()")
show("$ig->WriteRawData()")

banner("DONE")
end
