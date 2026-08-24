# Chase the scatter, with attenuation left ON.
#
# Corrected finding: the thing that decided whether the scatter model ran was
# NOT the thread count, it was IgJobReq.attenuationFlag.
#     prep2  attn=2, nThreads=6 -> 97 MSCAT lines, full scatter chain
#     prep4  attn=0, nThreads=6 ->  0
#     prep6  attn=0, nThreads=1 ->  0
# Turning attenuation off to dodge the CCTAC_3D crash also switched the SSS
# scatter off, which makes physical sense: SSS needs the mu information.
#
# So: leave the job's attenuationFlag alone (it is 2), keep nThreads at the
# pool width, and take back control with a bounded breakpoint instead of
# letting the CTAC poll loop spin forever.  CPrep3d::DoTask polls with
# usleep(), so an ignore count on usleep is a reliable way to regain control
# after the scatter phases have run.
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

banner("loads (attenuationFlag left as the job has it)")
with unlocked():
    show("$ig->LoadRawData()")
    show("$ig->InitLuts()")
    setv("set var *(int *) (0xf38968 + 8) = 0")
    show("((int (*)(void *, void *)) sharcCmp3dRemoteSinglesLoad)(&IgJobReq, (void *) 0xf38968)")
    setv("set var *(int *) (0xf38980 + 8) = 0")
    show("((int (*)(void *, void *)) sharcCmp3dRemoteDeadtimeLoad)(&IgJobReq, (void *) 0xf38980)")
    show("((int (*)(void *, char)) setup3dEmissJob)(&IgJobReq, (char) 0)")
show("$ig->InitRandomsFromSingles()")
show("IgJobReq.attenuationFlag")
show("IgJobReq.emissionScatterFlag")

banner("forge TransSysGeometry from the emission geometry")
# CPetReconContextCTAC::ValidateCTAC compares ~20 fields EX vs CTAC:
#   radial/axial ModulesPerSystem, BlocksPerModule, CrystalsPerBlock,
#   detectorRadialSize, detectorAxialSize, effectiveRingDiameter,
#   transaxial_crystal_0_offset, vqc_X/Y/ZaxisTranslation|Roll,
#   scanner_first_slice, interCrystalPitch, interBlockPitch  <- all from
#     SHARC_RDF_SYS_GEO_DATA: sysGeometry (0xf2f6a0) vs TransSysGeometry
#     (0xf2fea0), 2048 bytes each
#   patientEntry, patientPosition, tableLocation             <- from the PIFA
#
# TransSysGeometry is all zeros because no transmission RDF was ever opened,
# which is why it fails with "radialModulesPerSystem EX: 32 CTAC: 0".  Copying
# the emission geometry onto it makes every geometry comparison an identity.
# Pure ptrace writes, no inferior call.
_emis = bytes(gdb.selected_inferior().read_memory(0xf2f6a0, 2048))
gdb.selected_inferior().write_memory(0xf2fea0, _emis)
show("TransSysGeometry.radialModulesPerSystem")
show("TransSysGeometry.effectiveRingDiameter")
show("sysGeometry.radialModulesPerSystem")

banner("contexts at pool width, attenuation ON")
NTH = num("$ig->GetNumOfProcessingThreads()") or 6
for nm_, size, expr in (
        ("$ctac", 344, "((int (*)(void *, void *)) 0x441510)($ctac, $ig)"),
        ("$scat", 864, "((int (*)(void *, void *, int, char)) 0x48e070)($scat, $ig, %d, (char) 0)" % NTH),
        ("$prep", 896, "((int (*)(void *, void *, int, void *, void *)) 0x482500)"
                       "($prep, $ig, %d, $ctac, $scat)" % NTH)):
    setv("set %s = (char *) malloc(%d)" % (nm_, size))
    setv("set $x = (int) memset(%s, 0, %d)" % (nm_, size))
    show(expr)

show("((int (*)(void *)) 0x482290)($prep)")     # COsem3dPrep::Initialize
show("((int (*)(void *)) 0x434de0)($prep)")     # CPrep3d::CreateTaskList

banner("pointers the scatter path depends on")
show("*(void **) ($prep + 0x190)")   # CPrep3d field the DoTask guard tests
show("*(void **) ($prep + 0x198)")
show("$scat")
show("$ctac")
show("*(void **) ($scat + 0x98)")    # CScatterFully3dModel::CreateTaskList early-exits if NULL

banner("bounded DoTask: regain control via an ignore count on usleep")
_out = ex("break usleep") or ""
import re as _re
_m = _re.search(r"Breakpoint (\d+)", _out)
if _m:
    ex("ignore %s 400" % _m.group(1))
with unlocked():
    show("((int (*)(void *)) 0x435e70)($prep)")
ex("info breakpoints")

banner("scatter buffers after the phases ran")
for b in ("m_pPrompts", "m_pRandoms", "m_pScatter", "m_pScatterPrompts",
          "m_pScatterTOF", "m_pScatterTailScaleFactors", "m_pNormDT", "m_pAttn"):
    show("$cd->%s->m_pCyclicMemBuff->m_uiCounterElement" % b)
    show("$cd->%s->m_pCyclicMemBuff->m_uiAllocElement" % b)

for nm_, buf in (("scatter.f32", "$cd->m_pScatter"),
                 ("scatter_prompts.f32", "$cd->m_pScatterPrompts"),
                 ("scatter_tof.f32", "$cd->m_pScatterTOF"),
                 ("randoms.f32", "$cd->m_pRandoms")):
    dump_views(buf, nm_, "f4", dict(GEOM, produced_by="CPrep3d::DoTask, attenuation ON"))
banner("DONE")
end
