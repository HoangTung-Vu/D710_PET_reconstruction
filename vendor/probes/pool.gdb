# Scatter: hand the prep context to GE's real thread pool instead of driving
# DoTask by hand.
#
# What the vtable offsets in CPrep3d::DoTask actually are (address point =
# _ZTV11COsem3dPrep+16):
#     *0x20 DoTask        *0x30 Finalize      *0x80 DoPrep / GetScatterCounts
#     *0x88 AddViewsToCorr  *0x90 AreViewsAvailable
#     *0xa0 DoPostCTACPrep  *0xa8 DoPostScatterPrep  *0xb0 DoPostRandomsPrep
#
# So CPrep3d::DoTask is ONE data-parallel function that all N pool threads run
# together: it preps views, calls the scatter model's own DoTask() inline
# (*0x20 at Prep3d.cpp:462 and :500), and synchronises the N threads at four
# CRendezvous::Join() barriers (473, 484, 509, 527).  The scatter is not a
# second context running "beside" the prep -- it is a phase inside the same
# task, fenced by barriers.
#
# That is why neither earlier attempt got scatter:
#   nThreads=6 + 1 driving thread -> Join() waits for 6 arrivals, never opens
#                                    (stall at view 28, 0% CPU)
#   nThreads=1                    -> barriers open, but ZERO MSCAT phases ran
#                                    and m_pScatter stayed empty
#
# The fix is to stop driving DoTask by hand and let the pool run it, which is
# what Do3dEmissionImage does (StartContext at IgManager.cpp:248).  That needs
# a std::string for the context name, built here with the gcc-4.3 COW-string
# ctor at the PLT.
source /vendor/boot.gdb
source /vendor/lib.gdb
set var sharcCmpDebugFlag = 0x4
set var PrintTraceFlag = 1
source /vendor/job.gdb

python
banner("open + allocate")
show("((int (*)(void *)) sharcCmpOpenDataFiles)(&IgJobReq)")
gdb.execute("set $ig = (CIgManager *) $ig")
gdb.execute("set $rd = $ig->m_pRawDataMem._M_ptr")
gdb.execute("set $cd = $ig->m_pCorrDataMem._M_ptr")
gdb.execute("set $im = $ig->m_pImageDataMem._M_ptr")
for c in ("$ig->CleanDataBuffers(0)", "$ig->InitParamStruct()",
          "$rd->DeallocateBuffs(0, $ig)", "$cd->DeallocateBuffs(0, $ig)",
          "$rd->ReallocateBuffs(0, $ig)", "$cd->ReallocateBuffs(0, $ig)",
          "$im->ReallocateBuffs(0, $ig)"):
    show(c)

NU    = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_u")       or 0)
NV    = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_v_theta") or 0)
NVIEW = int(deep("$ig->m_pParamStruct._M_ptr", "rawData", "number_phi")     or 0)
GEOM  = dict(number_u=NU, number_v_theta=NV, number_phi=NVIEW,
             axes="view(%d) x v(%d) x u(%d)" % (NVIEW, NV, NU))

banner("loads")
with unlocked():
    show("$ig->LoadRawData()")
    show("$ig->InitLuts()")
    gdb.execute("set var *(int *) (0xf38968 + 8) = 0")
    show("((int (*)(void *, void *)) sharcCmp3dRemoteSinglesLoad)(&IgJobReq, (void *) 0xf38968)")
    gdb.execute("set var *(int *) (0xf38980 + 8) = 0")
    show("((int (*)(void *, void *)) sharcCmp3dRemoteDeadtimeLoad)(&IgJobReq, (void *) 0xf38980)")
    show("((int (*)(void *, char)) setup3dEmissJob)(&IgJobReq, (char) 0)")
show("$ig->InitRandomsFromSingles()")

# attenuation off AFTER the loads, so the PIFA is read but CCTAC_3D is skipped
gdb.execute("set var IgJobReq.attenuationFlag = 0")
gdb.execute("set var IgJobReq.fCTACFlag = 0")

banner("contexts, built for the real pool width")
NTH = num("$ig->GetNumOfProcessingThreads()") or 6
say("   nThreads = %d\n" % NTH)
for nm_, size, expr in (
        ("$ctac", 344, "((int (*)(void *, void *)) 0x441510)($ctac, $ig)"),
        ("$scat", 864, "((int (*)(void *, void *, int, char)) 0x48e070)($scat, $ig, %d, (char) 0)" % NTH),
        ("$prep", 896, "((int (*)(void *, void *, int, void *, void *)) 0x482500)"
                       "($prep, $ig, %d, $ctac, $scat)" % NTH)):
    setv("set %s = (char *) malloc(%d)" % (nm_, size))
    setv("set $x = (int) memset(%s, 0, %d)" % (nm_, size))
    show(expr)

banner("build a std::string by hand (no ctor call)")
# gcc 4.3 std::string is COW: the object is ONE pointer to the character data,
# and the data is preceded by _Rep { size_type length; size_type capacity;
# _Atomic_word refcount; } = 24 bytes on x86-64.  Building that by hand is
# pure memory writes -- no inferior call, so nothing to go wrong.  Calling
# the ctor through the PLT died with "Cannot access memory at address 0x33".
import struct as _s
inf = gdb.selected_inferior()
setv("set $rep = (char *) malloc(64)")
rep = int(val("$rep"))
name = b"Prep3d\x00"
inf.write_memory(rep, _s.pack("<qqi", len(name) - 1, len(name) - 1, 1)
                      + b"\x00" * 4 + name)
setv("set $nm = (char **) malloc(8)")
nm = int(val("$nm"))
inf.write_memory(nm, _s.pack("<Q", rep + 24))
show("*(char **) $nm")
show("*(char *) (*(char **) $nm)")

banner("RunContext: let the pool run CPrep3d::DoTask on all threads")
with unlocked():
    # CThreadPool::RunContext(CContext*, std::string const&) = Start + Wait
    show("((int (*)(void *, void *, void *)) 0x4a6540)($ig->m_pThreadPool._M_ptr, $prep, $nm)")

banner("what came out")
for b in ("m_pPrompts", "m_pRandoms", "m_pScatter", "m_pScatterPrompts",
          "m_pScatterTOF", "m_pScatterTailScaleFactors", "m_pNormDT", "m_pAttn"):
    show("$cd->%s->m_pCyclicMemBuff->m_uiCounterElement" % b)

dump_views("$cd->m_pRandoms", "randoms.f32", "f4",
           dict(GEOM, produced_by="COsem3dPrep::DoPrep, run on the thread pool"))
dump_views("$cd->m_pScatter", "scatter.f32", "f4",
           dict(GEOM, produced_by="CScatterFully3dModel via CPrep3d::DoTask"))
banner("DONE")
end
