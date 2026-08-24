# Bring GE's pet_recon up to a fully-initialised idle state, off the console.
#
# Two things are neutralised, both deliberate:
#
#  CPCCommThread  is returned from at entry.  It is the socket server that
#    talks to cmpRx, and it needs the console's msgHandler daemon (CT85_OC0
#    port 1513) -- a binary that is NOT in petsw/.  It also opendir()s
#    /petRDFS/OVLFILES, which cannot exist in this namespace.  Either failure
#    ends in exit(-1) and kills the process.
#    Cost: it is also what opens the CPC->IG FIFO, so fdCpcIgFIFO is left
#    invalid.  See run_job.gdb / run_job_fifo.gdb.
#
#    But its FIRST three statements (cpcMain.cpp:234-236) are not socket work
#    at all -- they publish the AP-side globals that every sharcCmp*/sharcAp*
#    kernel dereferences:
#        m_pRawDataMem  = arg[0]   (0xf38948)
#        m_pImageDataMem= arg[1]   (0xf38950)
#        m_pParamStruct = arg[2]   (0xf38958)
#    Skipping them is what made LoadRawData() segfault at 0x516da5, in
#    sharcCmp3dRemoteLoadPrompts:  mov m_pRawDataMem,%rax ; mov (%rax),%r14
#    on a NULL global.  CIgManager's own m_pRawDataMem member was fine all
#    along -- the AP side simply never looks at it.  So the three stores are
#    replayed here before returning.  CIgManager::InitCPCThread packs arg as
#    {m_pRawDataMem._M_ptr, m_pImageDataMem._M_ptr, m_pParamStruct._M_ptr},
#    which is why arg[1] is the image and arg[2] the params, not the order the
#    class declares them in.
#
#    pet_recon is a non-PIE ELF EXEC, so those three addresses are fixed; they
#    are verified against the symbol table at boot (see check below).
#
#  CGpuManager::Initialize is forced to return 0.  OclMgr construction throws
#    because there is no OpenCL platform here and GE never shipped the .cl
#    kernel sources (find petsw -name '*.cl' -> 0 files; they lived on the
#    accelerator chassis at 10.1.1.x).  This costs nothing for corrections:
#    every Ocl* class in the binary is OsemTofGpu::* -- forward/back projection,
#    ratio step, image update, PSF.  The GPU does OSEM, never randoms, scatter,
#    normalisation or dead time, all of which are sharcAp* CPU kernels.
set confirm off
set pagination off

# A SIGSEGV inside an inferior call otherwise leaves gdb parked in the dead
# frame ("GDB remains in the frame where the signal was received"), and every
# later call runs on that broken stack -- one flake poisons the whole run.
set unwindonsignal on
set unwind-on-terminating-exception on

# scheduler-locking is set at the bottom of this file: before `run` there is
# no live thread and gdb rejects it with "Target 'exec' cannot support this
# command".

python
import os, gdb
# The inferior needs GE's libs; gdb itself must NOT see them (their libstdc++
# is older than gdb's and stops it starting).  run_petsw.sh / the Dockerfile
# export PETSW_VLIB pointing at the five libs from the console's /usr/lib64.
gdb.execute("set environment LD_LIBRARY_PATH /usr/PET/lib64/linux2:"
            + os.environ.get("PETSW_VLIB", "/vendorlib"))

def _s(expr, text):
    """Write a NUL-terminated string into a char array in the inferior.

    An inferior strcpy() call would resume every thread and trips over glibc's
    strcpy ifunc besides; this is a plain ptrace poke.
    """
    v = gdb.parse_and_eval(expr)
    addr, n = int(v.address), v.type.sizeof
    b = text.encode() + b"\x00"
    if len(b) > n:
        raise gdb.GdbError("%s: %d bytes into %d" % (expr, len(b), n))
    gdb.selected_inferior().write_memory(addr, b + b"\x00" * (n - len(b)))
end

break CPCCommThread
commands
  silent
  # cpcMain.cpp:234-236, replayed.  $rdi still holds arg: the breakpoint is on
  # the function's first instruction, before any push touches it.
  set var *(unsigned long *) 0xf38948 = ((unsigned long *) $rdi)[0]
  set var *(unsigned long *) 0xf38950 = ((unsigned long *) $rdi)[1]
  set var *(unsigned long *) 0xf38958 = ((unsigned long *) $rdi)[2]
  printf "AP globals published: rawDataMem=%p imageDataMem=%p paramStruct=%p\n", \
         *(void **) 0xf38948, *(void **) 0xf38950, *(void **) 0xf38958
  return (void *)0
  disable 1
  continue
end

break CGpuManager::Initialize
commands
  silent
  return (int)0
  disable 2
  continue
end

break CIgManager::RunCyclic
commands
  silent
  disable 3
  # Grab the live CIgManager.  It owns the correction entry points that take no
  # arguments at all -- LoadRawData(), InitRandomsFromSingles(),
  # Do3dNormProcessing() -- so $ig is the handle for driving them directly,
  # below the job/socket layer entirely.
  set $ig = this
  printf "\n=== pet_recon idle and initialised, $ig = CIgManager* ===\n"
end

run

# Now that there are threads to lock: an inferior call resumes EVERY thread by
# default, so pet_recon's six processing threads run while gdb sits in the
# middle of, say, ReallocateBuffs -- which produced an intermittent SIGSEGV
# inside glibc's allocator.  Default to locking; lib.gdb's unlocked() lets the
# pool run again for the handful of calls -- LoadRawData above all -- that
# genuinely dispatch work to it.
set scheduler-locking on

# Clear the x86 direction flag.  gdb hands an inferior call the EFLAGS from the
# stop point, and with DF=1 glibc's `rep stos` memset runs backwards -- see
# clear_df() in lib.gdb for the SIGSEGV that costs.  lib.gdb clears it before
# every call as well; this covers scripts that call gdb directly.
set $eflags = $eflags & ~0x400
