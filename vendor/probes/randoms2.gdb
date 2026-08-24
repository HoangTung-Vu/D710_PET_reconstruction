# LoadRawData() loads the prompts views and nothing else.  Singles and the
# deadtime block have their own AP-side entry points, and both are handshake
# free -- the CPC normally pokes them over the FIFO, but the functions
# themselves only read IgJobReq and the AP globals:
#
#   sharcCmp3dRemoteSinglesLoad (jobReq, &loadSinglesStruct)
#       requires  jobReq->emissionRandomsFlag == 3   (already 3 in this job)
#                 loadSinglesStruct.singlesType == 0
#       -> rdfReadSingles(emissionFile[i], m_pRawDataMem->m_pSingles)
#
#   sharcCmp3dRemoteDeadtimeLoad(jobReq, &loadDeadtimeStruct)
#       requires  (1 << deadtimeType) & 0xd  -> type in {0,2,3}
#       -> rdfReadDeadTime(...) into m_pIntDeadtime / m_pMuxDeadtime, and
#          fills m_pDeadtimeStruct from sysGeometry
#
# Both structs are file-scope globals whose type is anonymous in the DWARF, so
# gdb cannot name their fields; they are written through their fixed addresses
# (non-PIE binary):  loadSinglesStruct 0xf38968, loadDeadtimeStruct 0xf38980,
# both {s32 status; s32 dragonSocketStatus; n32 type;}.
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

banner("fileStatus -- how many RDFs sharcCmpOpenDataFiles left open")
show("*(int *) 0xf38be0")
show("*(int *) 0xf38be4")
show("IgJobReq.emissionRandomsFlag")

banner("singles load")
gdb.execute("set var *(int *) (0xf38968 + 8) = 0")   # singlesType = 0
show("((int (*)(void *, void *)) sharcCmp3dRemoteSinglesLoad)(&IgJobReq, (void *) 0xf38968)")
show("*(int *) 0xf38968")            # singlesLoadStatus
show("*$rd->m_pSingles@12")

banner("deadtime load")
gdb.execute("set var *(int *) (0xf38980 + 8) = 0")   # deadtimeType = 0
show("((int (*)(void *, void *)) sharcCmp3dRemoteDeadtimeLoad)(&IgJobReq, (void *) 0xf38980)")
show("*(int *) 0xf38980")
show("$rd->m_pIntDeadtime")
show("*$rd->m_pIntDeadtime@8")
show("$rd->m_pDeadtimeStruct->intCorrFactor3d")
show("$rd->m_pDeadtimeStruct->muxCorrFactor3d")
show("$rd->m_pDeadtimeStruct->timingCorrFactor3d")
show("$rd->m_pDeadtimeStruct->livetimeFactorStatus")

banner("InitRandomsFromSingles")
show("$ig->InitRandomsFromSingles()")

banner("dumps")
dump_expr("$rd->m_pSingles", 576 * 24 * 4, "singles.i32",
          {"dtype": "i4", "layout": "576 crystals/ring x 24 rings"})
dump_views("$cd->m_pRandoms", "randoms.f32", "f4",
           {"axes": "view(288) x v(553) x u(381)",
            "produced_by": "CIgManager::InitRandomsFromSingles -> CalculateRFS"})
dump_views("$rd->m_pNorm", "norm_raw.f32", "f4",
           {"note": "as loaded from normalizationSinogramFile"})

banner("DONE")
end
