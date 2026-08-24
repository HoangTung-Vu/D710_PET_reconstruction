# probes/ — the scaffolding that produced `extract.gdb`

None of these are needed to run the tool. They are kept because each one
*established* something, and the reasoning in `../README.md` cites them as
evidence. Run them the same way as the tool, from the parent directory, with
somewhere to write:

    ../run.sh --out $D710_OUT/_probes probes/<name>.gdb

They source `/vendor/boot.gdb`, `/vendor/lib.gdb` and `/vendor/job.gdb`, and
`--out` is what lands on `/out` inside the container — there is no default, so
nothing a probe writes can end up in the source tree.

Several carry premises that were later overturned, so **`../README.md` is the
authority** wherever a probe's own comments disagree with it.

## What each probe establishes

| probe | what it established |
|---|---|
| `dump_cfg.gdb` | dumps `apCfg` + `sysGeometry` live → the source of `../PARAMS.md` |
| `layout.gdb`, `explore.gdb`, `state.gdb` | class layouts from DWARF: `CRawDataMem`, `CIgManager`, `CCorrDataMem`, `CViewBuffer` — how the buffers are reached at all |
| `cyclic.gdb` | `m_uiAllocElement = 1` after a "successful" Reallocate → **Deallocate must come first** |
| `pipe.gdb` | the corrected `RunCyclic` order — `LoadRawData()` returns `GRE_IG_SUCCESS` |
| `segv3.gdb` | caught the SIGSEGV in `CCyclicMemBuffer::AllocateMem` with `rdi = start − 17` → **the gdb direction-flag leak** |
| `normdt.gdb` | the norm × dead time sweep over all 288 views |
| `prep.gdb` | all three contexts (`CCTAC_3D`, `CScatterFully3dModel`, `COsem3dPrep`) construct; vtable entry points |
| `prep6.gdb` | `nThreads = 1` opens the `CRendezvous` barrier → 28 views became 281, **randoms** |
| `scat.gdb` | attenuation **on** → 97 `MSCAT_*` phases; the `usleep` ignore-count trick for regaining control |
| `ctac2.gdb` | measured the starvation: `$rd->m_pAttn` counter 0, `fileStatus[1] = 0` |
| `ctac.gdb` | forging `TransSysGeometry` takes `ValidateCTAC` complaints to zero |
| `mu.gdb` | mu-map injection alone: polls 387 → 0, but `nThreads = 6` still freezes on a barrier |
| `mu1.gdb` | **the combination that works** — attenuation on + `TransSysGeometry` forged + mu injected + `nThreads = 1` → scatter |

## Dead ends, kept so they are not retried

| probe | why it failed |
|---|---|
| `run_job.gdb`, `run_job_fifo.gdb` | the job layer via `sharcCmpProcessJobOnAp`: returns −1 without the CPC FIFO, and **deadlocks** with a fake pipe — gdb calls it from the very thread that would have to answer |
| `probe.gdb` | `sharcCmp3dRemoteLoad` with a zeroed load struct → SIGSEGV |
| `buf.gdb`, `corr.gdb`, `load.gdb`, `loaddbg.gdb`, `init.gdb` | reach for the prompts buffer without the AP globals or the Deallocate step — neither is optional |
| `randoms.gdb`, `randoms2.gdb` | `InitRandomsFromSingles()` returns SUCCESS with an all-zero buffer — it only computes RFS *parameters* |
| `prep2.gdb`, `prep3.gdb`, `prep4.gdb`, `prep5.gdb` | partial prep stages; `prep4`/`prep5` set `attenuationFlag = 0`, which **silently disables SSS scatter** |
| `pool.gdb` | handing the context to the real thread pool: `StartContext` calls `Initialize()` itself and returns −1 on the CTAC failure |
| `segv.gdb` | a DF probe that happens to pass — the direction flag depends on where the process stopped, which is what makes the bug look intermittent |
