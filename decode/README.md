# decode

Decoding GE raw RDF data with GE's own code.

This stage reads Discovery 710 raw data into Interfile sinograms, singles and
PETSIRD list-mode files. Neither codec is reimplemented: both of GE's codecs
remain unreversed, and reversing them is unnecessary, because the vendor
binaries run off the console.

```bash
docker load -i d710_full.tar
export D710_OUT=~/UET/d710_out

d710 decode --raw <petRDFS/.../DIR> --case nema
d710 decode --raw <petRDFS/.../DIR> --lists <petLists/.../DIR> --listmode --case nema
```

The host needs docker only: no conda, no i386 multiarch and no `petsw/`
checkout, since everything required is inside the image.

`decode_in.sh` is the per-bed loop that runs *inside* the container; `d710`
performs the mounting. It is a separate file rather than baked into the image,
so that changing the loop does not require rebuilding 7 GB.

Output is written to `$D710_OUT/<case>/decoded/`.

## Output

| file | content |
|---|---|
| `bed<n>.hs` / `.s` | prompt sinogram, 288 views × 553 planes × 381 tangential bins |
| `bed<n>.json` | the parsed RDF header (bed, table position, prompts, TOF, dose) |
| `bed<n>.singles.npy` | per-crystal singles, 576 × 24 |
| `bed<n>.prd` | PETSIRD list mode, written only with `--listmode` |
| `bed<n>.convert.log` | the decode log, which contains the MATCH line |

The next steps are `d710 estimate` (randoms, scatter, norm, dead time) and then
`d710 tostir`. `d710 exam` runs all three.

## Two decode paths, two different mechanisms

The same image contains two quite different ways of running GE's code.

**Sinograms, through `librdf.so.0`.** The per-row entropy codec
(`RDF_RIVN_4BIT_V1`) withstood three rounds of statistical attack. However,
`/usr/PET/lib/linux2/librdf.so.0` is a 32-bit x86 ELF with complete DWARF and
runs unmodified. A 64-bit Python process cannot `dlopen` a 32-bit shared object,
since the two ABIs cannot share a process, so `native/rdfx.c` is a small `-m32`
binary that performs the call and writes the raw array to a file;
`gerdf/vendor.py` runs it and then memory-maps the result.

**List mode, through `unglepl`.** The GLEPL codec that compresses `LIST*.BLF` is
likewise unreversed and likewise need not be: the console's own `unglepl` runs
here at roughly 200 MB/s. It is an i386 binary and requires six libraries that
were never copied off the console; symbol closure shows that exactly one symbol
among the six is actually called, `getcfg`. `native/stub/` therefore provides
five empty stubs carrying only the correct SONAME, plus one real
`libreadcfg.so`.

## Why the container is simpler than the host

The console binaries open their configuration files by absolute path, under
`/usr/PET/` and `/usr/g/`. On a host those paths do not exist, so
`native/petsw_run.sh` has to construct a writable overlay over `/usr` and bind
the `petsw/` tree into it, inside an unprivileged mount namespace.

Inside the image, the `Dockerfile` has already copied that tree to exactly those
two paths, which makes the namespace construction unnecessary — and not merely
unnecessary: a container has no `CAP_SYS_ADMIN`, `unshare` returns
`Operation not permitted`, and every console binary becomes unusable. Setting
`PETSW_ROOT=/` in the image tells `petsw_run.sh` to `exec` directly, setting
only `LD_LIBRARY_PATH`.

`stub/` must precede any directory holding a 64-bit library of the same name:
`/vendorlib` carries `libmsghand`, `libcupipc` and `libeventmgr` for the x86-64
`pet_recon`, and a 32-bit loader that encounters them only wastes time
rejecting the wrong ELF class.

## Data is not copied into the container

`d710 decode` uses bind mounts:

```
<raw>    -> /raw    read-only
<lists>  -> /lists  read-only
<out>    -> /out    writable
decode/  -> /decode read-only   (the per-bed loop, editable without a rebuild)
```

There are three reasons rather than one:

1. An exam is tens of gigabytes. Copying it doubles the storage, and the copy
   lands in the container's write layer.
2. The original acquisition data cannot be regenerated. `rdfx` opens files with
   `accessMode 0` because modes 1 and 2 both imply `O_RDWR`; a read-only mount
   is a second barrier enforcing the same thing.
3. The container runs with `--user $(id -u):$(id -g)`. Without it, everything
   under `--out` would be owned by root, since the container has only a root
   user and bind mounts pass the host uid through unchanged.

There is one exception: GLEPL writes its decompressed output beside the *output*
rather than beside the input (`/out/.gerdf_lm`, the size of the `.BLF` file),
and it is removed after each bed.

## How correctness is established

`convert` refuses to print MATCH unless the decoded total equals `totalPrompts`
in the header, and `decode_in.sh` aborts when MATCH is absent. The decode step is
therefore also the count check, and a truncated file cannot pass as valid data.

Measured on an eight-bed exam:

```
bed1  26,114,669  bed3  32,977,313  bed5  44,113,570  bed7  49,041,434
bed2  30,764,762  bed4  41,057,162  bed6  41,568,323  bed8  72,059,155
                                            8/8 MATCH, 38 s for the whole exam
```

List mode, bed 1: a 93.3 MB `.BLF` expands to 162.8 MB under `unglepl` and
yields 26,114,944 PETSIRD events in 59 seconds. The 275-event difference against
the sinogram is the pre-roll, removed with `--drop-preroll`.

## Known pitfalls

* **`-Wl,--export-dynamic` is required, not optional.** `librdf.so.0` is
  under-linked and binds `ErrLog` back to whatever loaded it, which here is the
  definition in `rdfx.c` rather than a `dlopen` of `libErr.so.0`. Without the
  flag, `ErrLog` does not enter `.dynsym` and `dlopen(RTLD_NOW)` fails with
  `undefined symbol: ErrLog`. The multilib branch of `build.sh` once lacked it,
  which went unnoticed because the host has no multilib while the container
  does. The final line of the `Dockerfile` is now a smoke test for exactly this
  failure.
* **`fopen64`, not `fopen`.** `rdfx` is 32-bit, so `FILE*` carries a 32-bit
  offset and fails silently at 2 GiB. A complete TOF dump is 3.34 GB: `fwrite`
  begins to fail while the in-memory counters still agree with the header.
* **Two-dimensional `SINO*` files cannot be decoded here.** Norm and calibration
  scans are uncompressed 2D arrays (`data_type` other than 7); `convert` reports
  a clear error rather than guessing.
