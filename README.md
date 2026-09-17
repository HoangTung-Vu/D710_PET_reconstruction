# D710

PET reconstruction for the GE Discovery 710, from the scanner's own raw RDF data.

The forward model is `y = S·(G x) + b`. All four correction terms — randoms,
scatter, normalisation and dead time — are obtained from GE's own kernel
(`pet_recon`, driven under gdb inside a container) rather than reimplemented in
Python. Attenuation is the only term built here, from CT.

## Running the pipeline

One-time preparation:

```bash
docker load -i d710_full.tar                      # vendor image, if not present
cd D710 && conda env create -f environment.yml    # host environment, petct_recon
cp .env.example .env                              # per-machine settings, not committed
```

On a machine that has apptainer but not docker — the workstation — see
[Running under apptainer instead of docker](#running-under-apptainer-instead-of-docker).
The two entry points take the same arguments and give the same results.

### The host environment must be conda

The constraint comes from `parallelproj`, which `d710 attn` uses directly and
`d710 lm` reaches through PyTomography. The package is not published on PyPI,
and `pytomography` does not declare it in `Requires-Dist`, not even as an
extra. No resolver therefore knows it exists: `pip` and `uv` produce a
PyTomography that imports but cannot reconstruct, because the import is lazy
and lives in `pytomography.projectors.PET`.

Since nothing declares the dependency, nothing imposes a version ceiling
either, and that is the dangerous part. `parallelproj` 2.x is a different
package under the same name: the compiled half moved into `parallelproj_core`,
a per-platform CPython extension no longer loaded through ctypes via
`$PARALLELPROJ_C_LIB`, and the six functions PyTomography calls at top level
(`joseph3d_fwd`, `joseph3d_back`, and the four
`joseph3d_{fwd,back}_tof_{sino,lm}`) disappeared from the `parallelproj`
namespace. Release 2.0.2 renamed the four TOF entry points and exported none of
them; `__all__` holds exactly six metadata names. Without a pin, the next
environment build resolves to 2.0.2 and all six raise `AttributeError`.
`environment.yml` pins **1.10.2**, the last 1.x release. There is no upstream
fix: 3.4.0 is the current PyTomography.

### A complete case

Run from inside `D710/`:

```bash
export D710_OUT=~/UET/d710_out         # where output goes; there is no default
conda activate petct_recon             # or: export D710_PYTHON=<the env's python>
CASE=fdg26081901
SRC=~/UET/Handson_PET_CT_Reconstruction/data/cases/20260819_FDG26081901_ok

./d710 exam --case $CASE \
    --raw  $SRC/raw/petRDFS/NQLHXWDK/PZAMCDES/USIRBPEU \
    --ct   $SRC/dicom/CT_s002_CT_WB_AC_5mm \
    --listmode --lists $SRC/raw/petLists/NQLHXWDK/PZAMCDES/USIRBPEU

./d710 attn      --case $CASE
./d710 lm recon  --case $CASE --resume
./d710 export    --case $CASE --format both --lm
```

| step | command | produces |
|---|---|---|
| 1 | `d710 exam` | sinograms, the event table and GE's four correction terms |
| 2 | `d710 attn` | attenuation from CT (`--ct` is read from the bed sidecar) |
| 3 | `d710 lm recon` | the list-mode reconstruction, all 55 TOF bins, to `recon_lm.npz` |
| 4 | `d710 export` | Bq/mL and SUV, as NIfTI and DICOM |

Step 1 runs inside `d710:full` and needs nothing but bash, a container runtime
and any `python3`. Steps 2 to 4 run on the host, in `petct_recon`. **No step
needs SIRF.**

`exam` skips beds that are already complete, so it is safe to rerun after an
interruption; `--force` rebuilds from the start. `attn` and `lm recon` behave
the same way, and `lm recon --resume` detects a changed configuration and
rebuilds the affected bed rather than reusing a stale result.

| command | function | runtime |
|---|---|---|
| `d710 decode` | RDF to Interfile and singles (plus the event table `bed<n>.lm.npy`) | `d710:full` |
| `d710 estimate` | GE's kernel to `randoms/scatter/normdt/norm_only.f32` | `d710:full` |
| `d710 tostir` | `.f32` to STIR Interfile, with a bit-exact self-check | `d710:full` |
| `d710 exam` | all three, for every bed | `d710:full` |
| `d710 attn` | CT to `work/bed<n>/attn.hs`, by parallelproj | host python |
| `d710 lm` | list-mode OSEM, to `recon_lm.npz` | host python |
| `d710 lowdose` | a reduced-dose copy of a case | host python |
| `d710 export` | Bq/mL and SUV to NIfTI/DICOM (`--lm` for `recon_lm.npz`) | host python |
| `d710 read` | read one vendor `.f32` | `d710:full` |
| `d710 shell` | interactive shell inside the image | `d710:full` |
| `d710 osem` | the sinogram path, kept for comparison — see below | `sirf-local:0.1` |

### The sinogram path, and why it is separate

`d710 osem` reconstructs from sinograms with SIRF/STIR. It is the only command
left that needs SIRF, it is kept for comparison against the list-mode path
rather than used, and it runs only on a machine with docker:

```bash
./d710_isolate_stir.sh osem --case $CASE --resume
./d710 export --case $CASE --format both
```

It reads the same `work/bed<n>/attn.hs` the host wrote and will not build one,
so `d710 attn` must have run first. Everything else `d710_isolate_stir.sh` is
given is forwarded to `./d710` unchanged.

The comparison is not close. Measured on the same bed of the paediatric case,
2 iterations x 24 subsets, the same four corrections:

| | sinogram, TOF 5 bins | list-mode, all 55 bins |
|---|---|---|
| time per bed | 1 h 36 min | 4 min 29 s |

`parallelproj` projects the whole sinogram on every subset, so subsets buy
nothing there, and the cost goes as `n_subsets x n_iterations`. In list mode
each event's ray is truncated to +-3 sigma about its own TOF position and is
therefore shorter than the full chord.

## The workstation: apptainer, one image, no SIRF

Two commands need a container — `decode` and `estimate`, both inside
`d710:full`. Everything after them is host python in the `petct_recon` conda
environment, so the workstation carries one `.sif` and no SIRF at all.

| | runs in | reached by |
|---|---|---|
| `decode`, `estimate`, `tostir`, `exam` | `d710:full` (apptainer) | `./d710_apptainer` |
| `attn`, `lm`, `lowdose`, `export` | the host | `./d710` (or `./d710_apptainer`, which forwards) |
| `osem` | `sirf-local:0.1` (docker) | not on the workstation — see below |

One-time preparation, on a machine that already holds the docker image:

```bash
apptainer build d710_full.sif docker-daemon://d710:full
```

Copy the `.sif` to the target machine and declare it once in `D710/.env`:

```bash
D710_SIF=/path/to/d710_full.sif
```

If it is not declared, it is looked for in `$D710_SIF_DIR`, `D710/sif/`,
`D710/`, `$D710_OUT/sif` and `~/sif`. Verify the machine before the first case;
every condition checked here otherwise fails later and far less legibly:

```bash
./d710_apptainer doctor
```

The check covers apptainer itself, the `.sif`, the decoder self-test, fakeroot,
and — the half that matters most now — the host environment: the seven Python
packages, the GPU, and whether the installed `torch` and `libparallelproj`
builds can actually use it. On a machine with a GPU, read the [GPU](#gpu)
section before the first case; the stock pins in `environment.yml` do not work
there and one of them fails loudly rather than quietly.

A complete case:

```bash
export D710_OUT=~/UET/d710_out
CASE=fdg26081901
SRC=~/UET/Handson_PET_CT_Reconstruction/data/cases/20260819_FDG26081901_ok

./d710_apptainer exam --case $CASE \
    --raw  $SRC/raw/petRDFS/NQLHXWDK/PZAMCDES/USIRBPEU \
    --ct   $SRC/dicom/CT_s002_CT_WB_AC_5mm \
    --listmode --lists $SRC/raw/petLists/NQLHXWDK/PZAMCDES/USIRBPEU

conda activate petct_recon
./d710 attn     --case $CASE
./d710 lm recon --case $CASE --resume
./d710 export   --case $CASE --format both --lm
```

The last three run on the host, outside any container, so apptainer plays no
part in them; `./d710_apptainer` forwards them to `./d710` unchanged, which is
why writing `./d710_apptainer attn` also works.

`d710 osem` is the one command the workstation cannot run: it is the sinogram
path, it needs SIRF, and it is kept for comparison rather than used. It stays on
a machine with docker, behind `./d710_isolate_stir.sh`.

### Wrappers, not copies

`d710` holds the pipeline logic — the TOF switch must reach both decode and
estimate, beds are ordered numerically rather than lexically, SINO files are
resolved per bed, and the interpreter is discovered at run time — so a second
copy would diverge at the first change. `d710_apptainer` changes only the two
things apptainer actually changes, then calls `./d710` itself:

1. **The image name is the `.sif` path.** `D710_IMAGE` is already the single
   place the tree reads an image name from, so pointing it at a `.sif`
   accomplishes most of the port.
2. **`apptainer_shim/` is placed first on `$PATH`**, and it contains a file
   named `docker` that translates directly into `apptainer exec`.

The shim is required rather than convenient: two other places in the tree
invoke `docker` themselves and neither is reachable by editing an entry-point
script — `vendor/run.sh` (gdb driving `pet_recon`, that is, the `estimate`
step) and `utils/container.py` (`rdf_info`, `cal_tags`, `ct_to_pifa`, and the
WCC lookup `d710 export` makes). With the shim, both pass through one
translation layer and `d710` needs no change.

### Two cases that do not translate mechanically

**`--writable-tmpfs` is always enabled**, for two independent reasons. First,
it is the actual meaning of `docker run --rm`: a write layer discarded on exit.
Second, without an overlay apptainer cannot create mount points at all. A
`.sif` is squashfs, and `/d710`, `/vendor`, `/case`, `/raw`, `/lists`, `/ct`
and `/cal` do not exist inside `d710:full`, whose Dockerfile creates only
`/out`, `/vendorlib` and `/petRDFS`. If an out-of-space error appears — the
default session tmpfs is 64 MiB — point `D710_APPTAINER_OVERLAY` at a directory
on disk.

**`estimate` requires `--fakeroot`.** `vendor/run.sh` is the one place that
deliberately does not pass `--user`, because `pet_recon` opens
`/usr/PET/systemConfig/cmcfg.xml` read-write and that file is root-owned and
only `u+w` in the image. The shim follows docker's rule: no `--user` means
uid 0, and uid 0 under apptainer means `--fakeroot`. It probes for fakeroot
before running and, where it is unavailable, prints the workaround (extract
`systemConfig` and bind it back over the image via `D710_APPTAINER_BIND`);
otherwise the permission failure surfaces two hundred lines into a gdb log. On
this path apptainer is the better of the two: the resulting `.f32` files belong
to the invoking user rather than to root.

### GPU

The shim adds `--nv` when `/dev/nvidiactl` is present, but nothing inside
`d710:full` uses it: the GPU path of `pet_recon` is forced off in
`vendor/boot.gdb` because GE never shipped the OpenCL kernel sources. The GPU
is worth having for the host half — `d710 attn --device cuda` and `d710 lm` —
and two things have to line up for it:

* **`libparallelproj` must be a `cuda*` build.** The pin in `environment.yml`
  is `cpu_*`, which has no `libparallelproj_cuda.so`. Worse, on a machine that
  *has* a GPU the cpu build does not merely waste it, it fails:
  `parallelproj/backend.py` sets `cuda_present = shutil.which("nvidia-smi") is
  not None` and then requires the CUDA library with no fallback, raising
  `ImportError: Cannot find parallelproj cuda lib`.

  ```bash
  conda install -c conda-forge "libparallelproj=1.10.2=cuda129_h897a41e_203"
  ```

  `conda search -c conda-forge 'libparallelproj=1.10.2'` lists exactly three
  build families: `cpu_*`, `cuda129_*` and `cuda130_*`.

* **`torch`'s CUDA major version must not exceed the driver's.** Within CUDA 12
  any 12.x wheel runs on any 12.x driver, but a CUDA 13 wheel needs a CUDA 13
  driver. A machine whose `nvidia-smi` reports "CUDA Version: 12.6" therefore
  cannot run `torch ...+cu130`, and the only symptom is
  `torch.cuda.is_available()` returning `False`.

  ```bash
  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
  ```

`./d710_apptainer doctor` reports both and names the fix.

Additional environment variables, beyond everything `./d710 --help` lists:

| variable | function |
|---|---|
| `D710_SIF` | the image |
| `D710_SIF_DIR` | where to look for it |
| `D710_APPTAINER_BIN` | the apptainer binary, if not on `PATH` |
| `D710_APPTAINER_FAKEROOT` | `1` to force, `0` to forbid, `auto` (default, probed once) |
| `D710_APPTAINER_OVERLAY` | `tmpfs` (default), `none`, or a directory on disk |
| `D710_APPTAINER_NV` | `1`, `0` or `auto` |
| `D710_APPTAINER_BIND` | additional binds, space-separated |
| `D710_APPTAINER_QUIET` | `1` to suppress the echo of each apptainer command |

## Where the code runs, and what it needs

| | host python (`petct_recon`) | `d710:full` | `sirf-local:0.1` |
|---|---|---|---|
| commands | `attn`, `lm`, `lowdose`, `export` | `decode`, `estimate`, `tostir`, `exam`, `read`, `shell` | `osem` |
| reached by | `./d710` | `./d710` or `./d710_apptainer` | `./d710_isolate_stir.sh` |
| on the workstation | yes | yes (apptainer) | no |

Nothing on the list-mode path imports `sirf` or `stir`. The segment layout is
read from the Interfile header by `utils/interfile.py`, every term is read with
`np.fromfile`, and attenuation — the last thing that needed STIR — is projected
by `utils/attn_proj.py` with parallelproj, in the same world frame
`PETLMSystemMatrix` reconstructs in.

### Attenuation without STIR

`utils/attn_proj.py` replaces SIRF's
`AcquisitionSensitivityModel.compute_attenuation_factors`. For each bin it
averages `exp(-int mu dl)` over the ring pairs the bin merges, with `mu` in
1/mm and the path in mm, using `parallelproj.joseph3d_fwd` on the crystal
positions `utils/geometry.py` computes. On a whole D710 bed that is 576 ring
pairs x 109,728 LORs = 63.2 M rays, about 40 s on 16 CPU cores and far less on
a GPU (`--device cuda`).

**Which detector carries which ring had to be measured.** A plane merges
several ring pairs, and the pairing decides how the LOR tilts axially. Reversing
it mirrors the segment axis -- `+s` for `-s` at the same axial index, with
segment 0 a fixed point, since its pairs `(r+1, r)` and `(r, r+1)` already form
a symmetric set.

The evidence is `d710 lm check`, which re-histograms the list-mode events --
decoded from GE's `LIST*.BLF` -- through `BinMap.flat()` and compares element by
element against the prompts sinogram the same decoder produced from GE's
independent `SINO*` file. Two vendor data paths, one acquisition. On
`fdg26081901`:

| bed | events | bins differing, as shipped | bins differing, pairing reversed |
|---|---|---|---|
| 1 | 29,962,337 | **0** | 24,172,076 |
| 3 | 38,272,246 | **0** | 28,315,342 |
| 6 | 37,691,517 | **0** | 29,076,944 |

and under the reversal the histogram of segment `+s` came out equal to the
decoded segment `-s`, count for count. The check can fail, so its passing is
evidence. `tests/test_lm_data.py` fixes both halves; `utils/binmap.py` holds the
statement of the convention and is the only place it is written down.
`utils/attn_proj.py` reads `BinMap.ring_pairs_by_plane()` rather than restating
the rule, which is what makes `lm check` prove the attenuation geometry too.

### The decoder declares the ring differences with STIR's sign

Until 2026-09-18 the decoder labelled the first oblique block `+k` and `BinMap`
paired ring `a` with `det1`. That reproduced GE's data exactly -- but STIR's own
rule is `ring2 - ring1 = rd` (`ProjDataInfoCylindrical.inl`, "KT 01/08/2002
swapped rings"), so **STIR read the same header as the mirror of what the file
holds**, and `d710 osem` back-projected every oblique bin along a ray tilted the
wrong way.

Two one-line changes fix it, and they cancel:

* `custom_tool/gerdf/interfile.py::ordered_segments` labels the first oblique
  block `-k`, so the declaration matches STIR's sign;
* `utils/binmap.py` fills `pl[b, a]` instead of `pl[a, b]`, adopting the same
  sign.

Because they cancel, `pl`, `flat()`, `lor_table()` and every `attn.s` come out
**numerically identical**, and the `.s` payload is byte-identical -- checked by
sha256 on three beds, and `attn.s` rebuilt from scratch hashes the same. Only
STIR's reading changes. Measured on `fdg26081901` against GE's own image, on the
SUV scale inside the body:

| | r vs GE | SUV ratio |
|---|---|---|
| sinogram, before | 0.9602 | 0.9275 |
| **sinogram, after** | **0.9728** | 0.9270 |
| list-mode, before and after | 0.9760 | 0.9308 |

The list-mode path never took its geometry from the header -- PyTomography
places each event from its two crystal ids through `scanner_LUT` -- so it is
bit-for-bit unchanged. That is also why the earlier `attn` correction barely
moved it: measured on one bed with everything else held fixed, the ring pairing
of `attn` alone changes the image by **+0.21 %**. Misplacing a ray is a
structural error; mis-weighting it by a percent is not.

> **A header written before 2026-09-18 says the opposite and there is no marker
> to detect it.** Anything decoded earlier must be decoded again -- the payload
> is byte-identical, only the header text differs, so `d710 decode` followed by
> `d710 tostir` is enough, and `attn.hs`, `lm.npz` and `recon_lm.npz` all stay
> valid. `d710 lm check` is the tripwire: it fails outright on a stale header.
> `K_EXPORT` (sinogram) needs re-measuring, because the sinogram images change;
> `K_EXPORT_LM` does not.

### `parallelproj` means two different things here

They have different sonames and therefore coexist in one environment without
conflict:

| file | loaded by | origin |
|---|---|---|
| `lib/libparallelproj_c.so.1.10.2` | the Python package `parallelproj`, directly by `d710 attn` and through PyTomography by `d710 lm` | conda-forge, loaded by ctypes |
| `dlevel/INSTALL/lib/libparallelproj.so.2.0.7` | STIR, for `osem --projector parallelproj` | built by SIRF-SuperBuild, used through its C++ API |

The 2.x incompatibility described above concerns only the first row. The 2.0.7
build that STIR links is a C++ library, does not pass through the Python
namespace, and is unaffected.

**`attn.hs` files written before this change must be rebuilt**, because the
header layout changed with them — `utils/attn.py` clones the header from the
prompts, the same layout as every other file in `work/bed<n>/`, rather than
writing SIRF's own segment order:

```bash
./d710 attn --case <case> --force
```

### One image grid, shared by both reconstruction paths

The two SIRF builds interpret `--xy` differently. The build inside
`sirf-local:0.1` fixes the FOV at 718.01 mm and lets the voxel size follow `xy`;
the host build fixes the voxel at 2.1306 mm and lets the FOV follow. The same
`--xy 256` therefore yields 2.8047 mm in the container and 2.1306 mm on the
host — two different scales, and `K` is inversely proportional to voxel volume.

Every geometric constant and machine configuration value now lives in
`utils/scanner.py`, in one place. The default `XY = 337` is measured rather than
chosen: it is the only matrix size that yields 2.130600 mm in *both* builds, at
a 718.01 mm FOV. `scanner.sirf_grid` re-checks this at run time and adjusts `xy`
if the SIRF build in use would produce a different voxel size.

**Any `recon.npz`, `lm.npz` or `recon_lm.npz` built on the earlier grid must be
regenerated.** See `lm/README.md` and `lowdose/README.md`.

`d710 exam` requires only bash, docker (or apptainer) and any `python3` on the
host: no conda, no numpy, no pydicom, no i386 multiarch and no `custom_tool/`
checkout. `attn`, `lm`, `lowdose` and `export` need the project environment,
because PyTomography is not in the image.

`d710` does not assume that `python3` is the correct interpreter. On a machine
where `/usr/bin` precedes conda on `PATH` — a common arrangement — `python3` is
the system interpreter even after `petct_recon` has been activated, while
`python` is conda's. `d710` therefore tries, in order: `$D710_PYTHON` (if set it
is used exactly, and an unsuitable value is an error rather than a silent
substitution), `python3`, `python`, `$VIRTUAL_ENV/bin/python`,
`$CONDA_PREFIX/bin/python`, `/usr/bin/python3` and `/usr/local/bin/python3`. A
command that needs additional packages and finds none prints the complete list
of candidates tried. This works with a conda environment, with a bare venv, and
with no conda at all; `tests/test_python_resolution.py` fixes the behaviour.

## Installing the Python dependencies

`environment.yml` is a verbatim `conda env export` of the host environment, not
a hand-written list. It pins both versions and build strings, so it reproduces
the environment the measurements were made in — and, for the same reason, it can
only be rebuilt on linux-64.

```bash
conda env create -f environment.yml
conda activate petct_recon
pytest -q
python -m utils.export --case ped --format nifti
```

`d710` discovers this environment on its own, since `$CONDA_PREFIX/bin/python`
is among the candidates it tries, or it can be named explicitly:

```bash
export D710_PYTHON=$HOME/miniconda3/envs/petct_recon/bin/python
```

**Why not `uv` or `pip`.** As described above, `parallelproj` is absent from
PyPI and undeclared by every package that uses it, so this is a constraint
rather than a preference. The project previously used `uv` with a
`pyproject.toml` and a `uv.lock`; `uv sync` built everything except the one
package that makes `d710 lm` work, and both files were removed.

**`libparallelproj` is pinned to a `cpu_*` build.** The measurement machine has
no GPU, and `parallelproj` selects CUDA only when it finds `nvidia-smi` on
`PATH`, so the CUDA build (roughly 1.2 GB against 39 KB) buys nothing. On a
machine with a GPU, switch to `cuda129_*` or `cuda130_*`. `torch`, by contrast,
is the default PyPI build and therefore does carry CUDA wheels — a deliberate
asymmetry, since a machine reproducing this work may well have a GPU.

> **On a machine that has a GPU, the `cpu_*` pin is not merely wasteful; it
> fails.** `parallelproj/backend.py` sets `cuda_present = shutil.which(
> "nvidia-smi") is not None`, and when `cuda_present` is true it requires
> `libparallelproj_cuda.so` with no fallback to CPU, raising
> `ImportError: Cannot find parallelproj cuda lib`. The `cpu_*` build does not
> contain that file. There are two remedies:
> ```bash
> conda install -c conda-forge "libparallelproj=1.10.2=cuda129_h897a41e_203"
> ```
> or, if the GPU is too old for CUDA 12 and later (Kepler `sm_35`/`sm_37`),
> force CPU mode by hiding `nvidia-smi` from the PATH of the python process
> alone. Invoke the module directly, since `d710` is bash and needs the real
> `PATH`:
> ```bash
> env PATH=/nonexistent D710_OUT="$D710_OUT" PYTHONPATH="$PWD" \
>     python -u -m lm recon --case <case> --resume
> ```
> `conda search -c conda-forge 'libparallelproj=1.10.2'` lists exactly three
> build families: `cpu_*`, `cuda129_*` and `cuda130_*`.

### CPUs without AVX2 (Sandy Bridge and Ivy Bridge, such as the HP Z820)

The symptom is that `d710 lm recon` prints `=== bed 1` and then terminates with
`Illegal instruction (core dumped)` and no traceback.
`sudo dmesg | grep 'invalid opcode'` identifies `kornia_rs.cpython-*.so`, the
Rust extension of `kornia-rs`, whose prebuilt wheel uses AVX2 and FMA.

No code in this tree imports `kornia`. It arrives through
`pytomography.utils.spatial`, which does
`from kornia.geometry.transform import rotate`; `kornia/__init__.py` imports
`kornia.io` at top level, and `kornia/io/io.py` imports `kornia_rs`. Installing
`kornia` with `--no-deps` therefore does not help: `kornia_rs` must be
importable.

`kornia_rs` provides image file input and output only (JPEG, PNG, TIFF), which
the PET pipeline never uses. A stub is supplied, enabled per machine. It is
deliberately not on the default `PYTHONPATH`, because on a normal machine it
would shadow the real extension:

```bash
echo "PYTHONPATH=$PWD/tools/stubs" >> .env
```

[tools/stubs/kornia_rs.py](tools/stubs/kornia_rs.py) uses a module-level
`__getattr__`, so anything that genuinely calls into it raises immediately and
prints how to build the real extension
(`RUSTFLAGS='-C target-cpu=sandybridge' pip install --no-binary kornia-rs
kornia-rs==0.1.14`). It never returns a wrong value silently.

To confirm that the import path is clear:

```bash
PYTHONPATH=tools/stubs python -c "
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import OSEM, BSREM
from pytomography.transforms.shared import GaussianFilter
print('OK')"
```

**`sirf` and `stir` are absent from `environment.yml`, deliberately.** They are
built from source into `$CONDA_PREFIX/dlevel/` of a *different* environment,
`petct_reconstruction`. That is a C++ build bound to the libraries of that
environment and requiring its `LD_LIBRARY_PATH` at load time; no dependency file
reproduces it on another machine. Day-to-day use does not need them at all:
only `osem` does, and it runs from the `sirf-local:0.1` image via
`./d710_isolate_stir.sh`. Every other command — `decode`, `estimate`, `tostir`,
`exam`, `attn`, `lm`, `lowdose`, `export` — is unaffected by their absence.

> **`petct_recon` is not `petct_reconstruction`.** The names are similar but the
> environments are distinct. `petct_recon` is the host runtime defined by
> `environment.yml` (PyTomography and parallelproj); `petct_reconstruction` is
> the environment SIRF was built from source in, and
> `conda env export --from-history` shows it was created for exactly that
> purpose (cmake, swig 4.2.1, gcc, boost, eigen, fftw, hdf5). Never run
> `conda env update --prune` against `petct_reconstruction`: pruning removes the
> toolchain and breaks the SIRF build in `dlevel/`, which takes 30 to 60 minutes
> to rebuild.

## Output: `$D710_OUT`, never inside the source tree

The output root is resolved as `--out`, then `$D710_OUT`, then an error. The
absence of a default is deliberate.

```
$D710_OUT/<case>/
    decoded/        bed<n>.{hs,s,json,singles.npy,convert.log,prd}
    vendor/bed<n>/  {randoms,scatter,normdt,norm_only}.f32(+.json),
                    prompts.u16, singles.i32, dt_int.f32, dt_mux.f32,
                    job.gdb, extract.log, estimate.json, data/, ovl/
    work/bed<n>/    {randoms,scatter,background,normdt,norm_only,attn}.{hs,s},
                    to_stir.json
    recon.npz       the stitched volume, counts per voxel, from osem to export
    recon_lm.npz    the same, from the list-mode path (`d710 lm recon`)
    export/         <case>_bqml.nii.gz, <case>_suvbw.nii.gz, dicom/
    scratch/        SIRF's tmp_*.hs/.s, removable at any time
    logs/
```

A case is deleted with `rm -rf $D710_OUT/<case>`. Two cases may be processed
concurrently: each bed mounts its own directory onto `/out`, and there is no
longer a shared staging directory.

An earlier layout (`raw_prompt/`, `work/<case>_bed<n>/`, `vendor/out/`) is
migrated with `tools/migrate_out.sh --from <old tree> --to $D710_OUT`, which
performs a dry run by default and only ever moves files.

## Source layout

```
d710              the CLI, and the only entry point
d710_apptainer    the apptainer form of `d710`; a wrapper that calls `d710`
d710_isolate_stir.sh   `d710 osem` inside sirf-local:0.1; docker only
apptainer_shim/   a `docker` that translates to `apptainer exec`; first on $PATH
Dockerfile        a record of the image contents (the image is delivered, not built)
environment.yml   the host environment `petct_recon`, a verbatim conda export
decode/           the per-bed loop that runs inside the container
vendor/           the driver for GE's kernel, and the principal reference document
lm/               the list-mode algorithm (PyTomography); see lm/README.md
osem/             the sinogram algorithm (SIRF), kept for comparison
lowdose/          low-dose simulation by event decimation; see lowdose/README.md
utils/            everything shared that is not part of an algorithm
utils/scanner.py    every geometric constant, machine setting and image grid
utils/interfile.py  the Interfile header reader, which does not need STIR
utils/attn_proj.py  attenuation by parallelproj line integral
tests/            the checks on those conventions; see tests/README.md
tools/            migrate_out.sh, lm_frame.py, tof_direction.py and others
```

A future algorithm — FBP, MLEM, a deep prior — belongs in its own package
alongside `lm/`, reusing `utils/`. That is why `utils/` must contain nothing
algorithm-specific: a function that is meaningful only for OSEM belongs in
`osem/`. Nothing under `utils/` imports `sirf` or `stir`; the STIR
cross-checks that used to sit in `utils/geometry.py` are in
`tests/stir_oracle.py`, where only the tests reach them.

## The three inputs to sinogram OSEM, which are not interchangeable

| | file | how it is attached |
|---|---|---|
| raw prompts `y` | `<case>/decoded/bed<n>.hs` | `recon.set_input` |
| `S` | `<case>/work/bed<n>/normdt.hs` times af | `set_acquisition_sensitivity`, **before** `set_up` |
| `b` | `<case>/work/bed<n>/background.hs` | `set_background_term` |

`S` must be attached before `set_up` so that STIR folds it into the sensitivity
image; that is what makes the correction quantitative rather than a mere
reweighting. `b` bypasses `S` because randoms and scatter already lie in the
measured count domain. `tests/test_forward_model.py` rebuilds `y = S·(Gx) + b`
on a miniature scanner and compares it against known `S` and `b`. The deadline
for `S` is the `set_up` of the *reconstructor*, not of the acquisition model:
measured on SIRF 3.10.1, attaching it before or after `am.set_up` yields an
identical sensitivity image.

## Testing

```bash
conda activate petct_recon
export D710_OUT=~/UET/d710_out
python -m pytest -q            # or tests/run_tests.sh
```

The synthetic tests run on a miniature scanner and need no data. The data-backed
tests read `$D710_OUT/<case>/` and skip when it has not been built, including
when `$D710_OUT` is unset. See `tests/README.md`.

## Four conditions that are handled explicitly

All four arose in practice rather than in anticipation.

1. **ExamInfo must match.** `to_stir.py` clones the header from `bed<n>.hs`
   itself, changing only the data file name, the number format and the bytes per
   pixel, so every term shares one ExamInfo by construction. A freshly generated
   header disagrees on the energy window, and STIR raises
   `BinNormalisation set-up with different ExamInfo` only much later, inside
   `make_Poisson_loglikelihood`.
2. **`tmp_*.hs/.s` files are 231 MB each.** SIRF writes them into the current
   working directory, one pair per `get_uniform_copy`, and retains them until the
   object is collected. `utils.sirf_env.setup()` changes directory into
   `<case>/scratch`.
3. **Invariants must be aggregated per plane.** The raw sinogram runs at about
   0.06 counts per bin, so `p < r` holds at roughly 82 % of bins from Poisson
   noise alone. A per-bin comparison is meaningless.
4. **`normdt` already carries the span-2 multiplicity.** Multiplying by
   `ring_pair_multiplicity()` a second time squares it, giving a factor of four
   at the odd planes.

One further point used to bite here and no longer can. A file written by SIRF
uses a different layout from the decoded files — segments ascending, and the
view axis before the axial axis — yet `as_array()` still returns the same plane
order, so a SIRF-written `attn.hs` multiplied `normdt` correctly as a numpy
array while being unreadable by `np.fromfile`. `attn.hs` is now written by
`utils/attn.py` with the header cloned from the prompts, so every file in
`work/bed<n>/` has one layout and `utils/interfile.py` refuses the other
(`Header.require_plane_major`).

## The four invariants across all six beds of the paediatric case

| bed | table mm | kcps | prompts | randoms | scatter | S/(T+S) | livetime | ΣR/delays |
|---|---|---|---|---|---|---|---|---|
| 1 | −767.7 | 208 | 18,759,294 | 7,974,248 | 3,466,187 | 0.321 | 0.9569 | 0.994 |
| 2 | −643.5 | 264 | 23,736,423 | 11,124,030 | 4,194,211 | 0.333 | 0.9493 | 0.994 |
| 3 | −519.2 | 338 | 30,371,564 | 15,019,874 | 4,894,771 | 0.319 | 0.9418 | 0.994 |
| 4 | −394.9 | 411 | 37,010,299 | 20,052,033 | 5,639,968 | 0.333 | 0.9339 | 0.994 |
| 5 | −270.7 | 486 | 43,774,817 | 24,193,882 | 7,299,855 | 0.373 | 0.9285 | 0.994 |
| 6 | −146.4 | 969 | 87,220,937 | 22,169,301 | 14,736,322 | 0.227 | 0.9323 | 0.994 |

`Σp ≥ Σr` and `Σs ≤ Σ(p−r)` are violated at 0.00 % of planes across all six
beds, and the bin mapping is bit-exact on all six.

On NEMA bed 2, `Σs ≤ Σ(p−r)` is exceeded at 11 of 553 planes, all eleven within
the four outermost planes of a segment — the planes that merge the fewest ring
pairs, and where the SSS tail fit therefore has the least data. The total excess
is 0.036 % of the bed's scatter. `tests/test_pipeline_data.py` fixes both
statements.

Livetime follows randoms rather than prompts. Bed 6 has the highest prompt rate
(969 kcps) yet not the lowest livetime, because its randoms (22.2 M) are lower
than bed 5's (24.2 M): randoms scale as singles squared, and dead time follows
singles. This is a natural cross-check on the direction of `normdt`.

## Provenance of the sinogram OSEM pipeline

`osem/` follows SIRF's own examples rather than an invented API:

| example | contribution |
|---|---|
| `SIRF/examples/Python/PET/osem_reconstruction.py` | `make_Poisson_loglikelihood` and `OSMAPOSLReconstructor` |
| `.../get_multiplicative_sinogram.py` | `AcquisitionSensitivityModel`, and how norm combines with attenuation |
| `.../listmode_reconstruction.py` | `set_acquisition_sensitivity` together with `set_background_term` |

The full path is
`$CONDA_PREFIX/dlevel/build/sources/SIRF/examples/Python/PET/`. No SIRF example
combines all of prompts, randoms, scatter, norm and CTAC on a real sinogram;
`osem/` is where they are combined.

## Status

| component | status |
|---|---|
| sinogram, geometry and `.f32` to Interfile | complete, bit-exact on every run |
| randoms | complete — GE's kernel, 12.3 % of prompts (NEMA bed 2) |
| scatter (SSS) | complete — GE's kernel, S/(T+S) = 32.9 % |
| normalisation | complete — the scanner's own 3D norm, resolved from the exam header |
| dead time | complete — `normdt/norm_only`; count-rate dependent |
| CT attenuation | complete — `utils/attn_proj.py`, parallelproj rather than STIR; mu-map orientation measured on both axes, ring order measured against SIRF |
| full multi-bed runs | complete — `d710 exam` |
| decay correction and axial stitching | complete — referred to injection time, weighted by the sensitivity image |
| DICOM and NIfTI export | complete — `utils/export.py`, `Units = BQML` |
| list-mode (PyTomography) | complete — `lm/`, bit-exact bin mapping on all six beds; all 55 TOF bins in 2 m 01 s per bed, faster than non-TOF |
| transaxial FOV | complete — a disc of radius 356.7 mm applied to the initial estimate; previously 34 % of counts fell into the corners of the square grid |
| low-dose simulation | complete — `lowdose/`, with binomial and per-plane invariant checks |
| image grid | complete — `utils/scanner.py`, 337 × 2.1306 mm, identical in both reconstruction paths |
| the constant `K` | measured — `K_EXPORT` (sinogram) and `K_EXPORT_LM` (list-mode), two constants rather than one; requires re-measurement after the `MU_*_511` correction of 2026-09-06 |

`K` has been measured but currently requires re-measurement. It was obtained by
comparison against GE's own BQML reconstruction on five FDG cases
(`tools/compare_vendor.py`, then `tools/calib_k.py`) rather than on NEMA. There
are two constants, because the two reconstruction paths do not place the same
number of counts in a voxel: using one value for both is wrong by roughly a
factor of 2.1 (`utils/scanner.py`).

`K` is valid only for the exact correction chain it was measured with and for a
voxel pitch of 2.1306 mm, because the projector accumulates along voxel pitch
rather than volume; a value measured at 2.1306 mm and applied at 1.3672 mm reads
1.56 times high. On 2026-09-06, `MU_WATER_511` and `MU_BONE_511` were corrected
to the values the machine declares (`cmcfg.XR.xml`), which changed attenuation,
so both constants are currently invalid. Re-measure by running `run_all_ok.sh`,
then `export_all_ok.sh`, then `tools/calib_k.py`.

Replacing SIRF's attenuation with `utils/attn_proj.py` does **not** add to that:
the prompt-count weighted mean attenuation factor moved by -0.0006 %, which is
five orders of magnitude below anything `K` is quoted to.

Further detail: `vendor/README.md` is the principal reference;
`vendor/PARAMS.md` holds the live parameters read from the running process;
`vendor/cal/README.md` covers calibration; `decode/README.md` covers decoding;
`tests/README.md` covers testing; `RECONSTRUCTION_MATH.md` gives the
mathematical derivation of both algorithms.
