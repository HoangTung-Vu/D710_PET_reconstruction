# "can't reserve view N" comes from CCyclicMemBuffer::RequestPutDataNotAligned
# taking its error branch at 0x416eb0, which is entered when
#     m_uiFirstElementVal + m_uiCounterElement > requestedIndex
# Measure those three numbers instead of reasoning about them.
source /vendor/boot.gdb
source /vendor/lib.gdb
set var sharcCmpDebugFlag = 0
set var PrintTraceFlag = 0
source /vendor/job.gdb

# error branch of RequestPutDataNotAligned; $rbx = CCyclicMemBuffer*, $ebp = idx
break *0x416eb0
commands
  silent
  printf "REJECT buf=%s idx=%u first=%u counter=%u alloc=%u elemSize=%u\n", \
     ((CCyclicMemBuffer *)$rbx)->m_strName, $ebp, \
     ((CCyclicMemBuffer *)$rbx)->m_uiFirstElementVal, \
     ((CCyclicMemBuffer *)$rbx)->m_uiCounterElement, \
     ((CCyclicMemBuffer *)$rbx)->m_uiAllocElement, \
     ((CCyclicMemBuffer *)$rbx)->m_uiElementSize
  continue
end
disable 4

python
banner("setup")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $cd = $ig->m_pCorrDataMem._M_ptr")
gdb.execute("set $im = $ig->m_pImageDataMem._M_ptr")
show("$ig->InitParamStruct()")
show("$rd->ReallocateBuffs(0, $ig)")
show("$cd->ReallocateBuffs(0, $ig)")
show("$im->ReallocateBuffs(0, $ig)")

def cyc(label):
    banner("cyclic state: " + label)
    for b in ("$rd->m_pPrompts", "$rd->m_pDelays", "$cd->m_pRandoms"):
        for f in ("m_strName", "m_uiAllocElement", "m_uiElementSize",
                  "m_uiFirstElementVal", "m_uiCounterElement", "m_iConstStatus"):
            show("%s->m_pCyclicMemBuff->%s" % (b, f))
        show("%s->m_bAreAllViewsInvalid" % b)
        gdb.write("   ---\n")

cyc("after ReallocateBuffs")

banner("CleanDataBuffers(0) -- RunCyclic calls this before the alloc round")
show("$ig->CleanDataBuffers(0)")
cyc("after CleanDataBuffers(0)")

banner("LoadRawData with the reject probe armed")
gdb.execute("enable 4")
show("$ig->LoadRawData()")
gdb.execute("disable 4")
cyc("after LoadRawData")

banner("DONE")
end
