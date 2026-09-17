# probes

The scaffolding that produced `extract.gdb`.

None of these scripts is required to run the tool. They are kept because each
established a specific result, and the reasoning in `../README.md` cites them as
evidence. Run them as the tool is run, from the parent directory, with a
destination for output:

    ../run.sh --out $D710_OUT/_probes probes/<name>.gdb

They source `/vendor/boot.gdb`, `/vendor/lib.gdb` and `/vendor/job.gdb`, and
`--out` is what is mounted at `/out` inside the container. There is no default,
so nothing a probe writes can land in the source tree.

Several carry premises that were later overturned. Where a probe disagrees with
`../README.md`, that document is the authority.

## What each probe established

| probe | result |
|---|---|
| `dump_cfg.gdb` | dumps `apCfg` and `sysGeometry` live; the source of `../PARAMS.md` |
| `layout.gdb`, `explore.gdb`, `state.gdb` | class layouts from DWARF — `CRawDataMem`, `CIgManager`, `CCorrDataMem`, `CViewBuffer` — and hence how the buffers are reached at all |
| `cyclic.gdb` | `m_uiAllocElement = 1` after a nominally successful Reallocate, establishing that Deallocate must come first |
| `pipe.gdb` | the corrected `RunCyclic` order, after which `LoadRawData()` returns `GRE_IG_SUCCESS` |
| `segv3.gdb` | caught the SIGSEGV in `CCyclicMemBuffer::AllocateMem` with `rdi = start − 17`, identifying the gdb direction-flag leak |
| `normdt.gdb` | the norm and dead time sweep over all 288 views |
| `prep.gdb` | all three contexts (`CCTAC_3D`, `CScatterFully3dModel`, `COsem3dPrep`) construct, and their vtable entry points |
| `prep6.gdb` | `nThreads = 1` opens the `CRendezvous` barrier: 28 views became 281, yielding randoms |
| `scat.gdb` | attenuation enabled gives 97 `MSCAT_*` phases; also the `usleep` ignore-count technique for regaining control |
| `ctac2.gdb` | measured the starvation: `$rd->m_pAttn` counter 0, `fileStatus[1] = 0` |
| `ctac.gdb` | forging `TransSysGeometry` reduces `ValidateCTAC` complaints to zero |
| `mu.gdb` | mu-map injection alone: polls fall from 387 to 0, but `nThreads = 6` still blocks on a barrier |
| `mu1.gdb` | the working combination — attenuation enabled, `TransSysGeometry` forged, mu injected, `nThreads = 1` — which yields scatter |

## Dead ends, kept so that they are not retried

| probe | reason for failure |
|---|---|
| `run_job.gdb`, `run_job_fifo.gdb` | the job layer via `sharcCmpProcessJobOnAp` returns −1 without the CPC FIFO, and deadlocks with a substitute pipe, because gdb calls it from the very thread that would have to answer |
| `probe.gdb` | `sharcCmp3dRemoteLoad` with a zeroed load structure, giving SIGSEGV |
| `buf.gdb`, `corr.gdb`, `load.gdb`, `loaddbg.gdb`, `init.gdb` | reach for the prompts buffer without the AP globals or the Deallocate step, neither of which is optional |
| `randoms.gdb`, `randoms2.gdb` | `InitRandomsFromSingles()` returns SUCCESS with an all-zero buffer, because it computes only the RFS *parameters* |
| `prep2.gdb`, `prep3.gdb`, `prep4.gdb`, `prep5.gdb` | partial prep stages; `prep4` and `prep5` set `attenuationFlag = 0`, which silently disables SSS scatter |
| `pool.gdb` | hands the context to the real thread pool, but `StartContext` calls `Initialize()` itself and returns −1 on the CTAC failure |
| `segv.gdb` | a direction-flag probe that happens to pass, since the flag depends on where the process stopped — which is what makes the bug appear intermittent |
