#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Run a gdb script against GE's pet_recon in the d710:full container.

  ./run.sh --out <dir>                    an interactive shell
  ./run.sh --out <dir> extract.gdb        gdb -batch -x /vendor/extract.gdb

d710:full already contains the vendor tree (/usr/PET, /usr/g, /vendorlib) and
the decoder; see ../Dockerfile. Nothing from petsw/ is mounted, and
/usr/PET/systemConfig is writable in the image layer, which the configuration
manager requires.

Only two small mounts are made:
  /vendor  this directory, read-only, so gdb scripts are picked up live
  /out     whatever --out names, writable, where dumps and logs are written

--out is required and has no default. It was formerly a fixed ./out beside this
script, which meant every run staged through one global directory: two beds
estimated concurrently overwrote each other's dumps, and 7.5 GB of results
accumulated inside the source tree. The caller now mounts the bed's own
directory directly onto /out and gdb writes the final files in place, so there
is no staging, no move, and two runs in parallel cannot collide.

`OUT = "/out"` in lib.gdb is correct as it stands: that is the path inside the
container.

--mount selects the base-image fallback, binding petsw/ from the host onto the
`d710` stage instead of using the baked tree. It needs a writable copy of
systemConfig (about 1.4 GB), kept in --cache so that it survives between beds.
d710:full exists precisely to avoid this, and is what you should normally have.
USAGE
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"

image="${D710_IMAGE:-d710:full}"
OUT="${D710_RUN_OUT:-}"
DATA="${D710_DATA:-}"
CACHE=""
mount_fallback=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out)  OUT="$2"; shift 2 ;;
    --data)    DATA="$2"; shift 2 ;;
    --cache)   CACHE="$2"; shift 2 ;;
    --mount)   mount_fallback=1; image=d710; shift ;;
    -h|--help) usage; exit 0 ;;
    *) break ;;
  esac
done

[[ -n "$OUT" ]] || {
  echo "error: --out is required (or set D710_RUN_OUT)." >&2
  echo "  There is deliberately no default: output must never land in the" >&2
  echo "  source tree.  Normally you would not call this by hand at all --" >&2
  echo "  use  d710 estimate  or  d710 shell." >&2
  exit 2; }

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
mkdir -p "$OUT/ovl"
CACHE="${CACHE:-${D710_OUT:+$D710_OUT/.cache}}"
CACHE="${CACHE:-$OUT/.cache}"

mounts=()
if (( mount_fallback )); then
    petsw="$(cd "$root/../custom_tool/petsw" && pwd)"
    vlib="$CACHE/vendorlib"
    mkdir -p "$vlib"
    for l in libreadcfg.so libreadcfg.so.1 libeventmgr.so libeventmgr.so.1 \
             libmsghand.so libmsghand.so.1 libcupipc.so libcupipc.so.1 \
             libstartup.so libstartup.so.1; do
        [ -e "$petsw/usr/lib64/$l" ] && cp -n "$petsw/usr/lib64/$l" "$vlib/" 2>/dev/null || true
    done
    if [ ! -d "$CACHE/systemConfig" ]; then
        echo ">> one-time: copying a writable systemConfig (~1.4 GB) into $CACHE ..." >&2
        mkdir -p "$CACHE"
        cp -a "$petsw/usr/PET/systemConfig" "$CACHE/systemConfig"
    fi
    mounts=(-v "$petsw/usr/PET:/usr/PET:ro"
            -v "$petsw/usr/g:/usr/g:ro"
            -v "$CACHE/systemConfig:/usr/PET/systemConfig"
            -v "$vlib:/vendorlib:ro")
fi

docker image inspect "$image" >/dev/null 2>&1 || {
    echo "image '$image' missing.  It is handed over as an image, not built:" >&2
    echo "  docker load -i d710_full.tar" >&2
    echo "(../Dockerfile records what is inside it.)" >&2
    exit 1; }

extra=()
[[ -n "$DATA" ]] && extra+=(-v "$(cd "$DATA" && pwd):/data:ro")
[[ -n "${D710_JOB:-}" ]] && extra+=(-e "D710_JOB=$D710_JOB")
[[ -n "${D710_TOF:-}" ]] && extra+=(-e "D710_TOF=$D710_TOF")

args=(--rm -i
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
      --add-host CT85_OC0:127.0.0.1 --add-host CT85_OC1:127.0.0.1
      --add-host loghost:127.0.0.1  --add-host bay85ct:127.0.0.1
      --add-host bay87ct:127.0.0.1  --add-host trec:127.0.0.1
      "${mounts[@]}"
      "${extra[@]}"
      -v "$root:/d710:ro"
      -v "$here:/vendor:ro"
      -v "$OUT:/out")

if [ $# -eq 0 ]; then
    exec docker run "${args[@]}" -t "$image" bash
fi
exec docker run "${args[@]}" "$image" \
     gdb -q -batch -x "/vendor/$1" /usr/PET/release/petig/pet_recon
