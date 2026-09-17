#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${D710_ENV_FILE:-$HERE/.env}"
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r k v; do
        k="${k#"${k%%[![:space:]]*}"}"
        [[ "$k" == export[[:space:]]* ]] && k="${k#export}"
        k="${k//[[:space:]]/}"
        [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [[ -v "$k" ]] && continue
        v="${v%$'\r'}"; v="${v%%[[:space:]]#*}"
        v="${v#"${v%%[![:space:]]*}"}"
        v="${v%"${v##*[![:space:]]}"}"
        [[ "$v" == \"*\" || "$v" == \'*\' ]] && v="${v:1:${#v}-2}"
        [[ "$v" == '~/'* ]] && v="$HOME${v:1}"
        export "$k=$v"
    done < "$ENV_FILE"
fi

PY="${D710_PYTHON:-python3}"
SIRF_IMAGE="${D710_SIRF_IMAGE:-sirf-local:0.1}"
SIRF_ENV_SH="${D710_SIRF_ENV_SH:-/opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh}"

die() { echo "error: $*" >&2; exit 2; }
usage() {
    cat <<'USAGE'
d710_isolate_stir.sh -- `d710 osem`, with SIRF/STIR supplied by a docker image.

`osem` -- the sinogram reconstruction -- is the ONLY command left that needs
SIRF, and it is kept for comparison rather than used day to day; the list-mode
path is the one the project reconstructs with. A host installation of SIRF
reaches deep into the system, so SIRF comes from a prebuilt image (default
sirf-local:0.1) and the host needs only bash and docker.

  ./d710_isolate_stir.sh osem --case ped [--beds ...] [--iters n]

There is no apptainer counterpart and none is wanted: the workstation runs
decode and estimate in d710:full and everything else on the host, with no SIRF
anywhere. This script is therefore a docker-only, laptop-only convenience.

Every other command is forwarded to ./d710 untouched -- including `attn` and
`export`, which used to need this wrapper and now run on the host: `attn`
projects the CT mu-map with parallelproj (utils/attn_proj.py) and `export`
never needed more than numpy, nibabel and pydicom.

`--tof` and `--no-tof` configure decode and estimate rather than osem, which
reads TOF from the prompts header itself. `osem` accepts and discards them
without complaint, exactly as ./d710 does, so that `d710 osem --tof` and
`d710_isolate_stir.sh osem --tof` behave identically instead of one staying
silent while the other fails because argparse in `-m osem` does not know the
flag.

MOUNTS, AT THE EXACT HOST PATHS (-v /x:/x), with no path translation:
  $D710_OUT           rw   the output tree
  D710/ source dir    ro   `-m osem` runs from here
  CT directory        ro   taken from --ct and from work/bed<n>/to_stir.json
This keeps the absolute CT paths written into the sidecar and into recon.npz
valid verbatim, both inside the container and when reopened on the host.

ENVIRONMENT VARIABLES
All may also be set in D710/.env; copy .env.example and edit it.
  D710_OUT           root of the output tree (required, or pass --out)
  D710_CASE          default case name for --case
  D710_SIRF_IMAGE    the SIRF image (default sirf-local:0.1)
  D710_SIRF_ENV_SH   SIRF's env script inside the image
                     (default /opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh)
  D710_PYTHON        python3 on the host, used only to read sidecars
                     (standard library only; conda is not required)
USAGE
}

CMD="${1:-}"
case "$CMD" in
  ""|-h|--help|help) usage; exit $([[ -z "$CMD" ]] && echo 2 || echo 0) ;;
  osem) ;;
  *) exec "$HERE/d710" "$@" ;;
esac
shift

CASE="${D710_CASE:-}"; OUT=""; CT=""; BED=""; REST=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)    CASE="$2"; shift 2 ;;
    -o|--out)  OUT="$2"; shift 2 ;;
    --ct)      CT="$2"; shift 2 ;;
    --bed)     BED="$2"; shift 2 ;;
    --tof|--no-tof|--collapse-tof) shift ;;
    --tof-mash) shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)         REST+=("$1"); shift ;;
  esac
done
[[ -n "$CASE" ]] || die "--case is required (or set \$D710_CASE)"

O="${OUT:-${D710_OUT:-}}"
[[ -n "$O" ]] || die "no idea where to write the output.
  set   export D710_OUT=~/UET/d710_out
  or pass  --out <directory>"
O="${O/#\~/$HOME}"
mkdir -p "$O"
O="$(cd "$O" && pwd)"

docker image inspect "$SIRF_IMAGE" >/dev/null 2>&1 || die "no SIRF image '$SIRF_IMAGE'.
  check \`docker images\` and put the real name in D710/.env:
      D710_SIRF_IMAGE=<name:tag>
  or, for one run:  D710_SIRF_IMAGE=<name:tag> $(basename "${BASH_SOURCE[0]}") $CMD ...
  or, if SIRF is installed on this host:
      conda activate petct_reconstruction && $HERE/d710 osem --case $CASE"

ct_dirs_of() {
    local casedir="$1"
    [[ -d "$casedir" ]] || return 0
    "$PY" - "$casedir" <<'PY' 2>/dev/null || true
import glob, json, os, sys
seen = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "work", "bed*", "to_stir.json"))):
    try:
        ct = json.load(open(p)).get("estimate", {}).get("ct")
    except Exception:
        continue
    if ct and os.path.isdir(ct) and ct not in seen:
        seen.append(ct)
print("\n".join(seen))
PY
}

EXTRA_RO=()
add_ro() {
    local d p q
    for d in "$@"; do
        [[ -n "$d" && -d "$d" ]] || continue
        p="$(cd "$d" && pwd)"
        [[ "$p" == "$HERE" || "$p" == "$HERE"/* ]] && continue
        [[ "$p" == "$O"    || "$p" == "$O"/*    ]] && continue
        for q in ${EXTRA_RO[@]+"${EXTRA_RO[@]}"}; do [[ "$q" == "$p" ]] && continue 2; done
        EXTRA_RO+=("$p")
    done
}

add_ro "$CT"
while IFS= read -r d; do add_ro "$d"; done < <(ct_dirs_of "$O/$CASE")

MOUNTS=(-v "$HERE:$HERE:ro" -v "$O:$O")
for d in ${EXTRA_RO[@]+"${EXTRA_RO[@]}"}; do MOUNTS+=(-v "$d:$d:ro"); done

MOD=(osem --case "$CASE" ${BED:+--beds "$BED"} ${CT:+--ct "$CT"}
     ${REST[@]+"${REST[@]}"})

TTY=(); [[ -t 1 ]] && TTY=(-t)

ARGV=(docker run --rm -i "${TTY[@]}" --no-healthcheck
      --user "$(id -u):$(id -g)" -e HOME=/tmp
      -e D710_OUT="$O" -e PYTHONPATH="$HERE" -w "$O"
      "${MOUNTS[@]}" --entrypoint bash "$SIRF_IMAGE"
      -c '. "$0"; exec python3 -u -m "$@"' "$SIRF_ENV_SH" "${MOD[@]}")

printf '+ %s\n' "${ARGV[*]}" >&2
exec "${ARGV[@]}"
