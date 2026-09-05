#!/usr/bin/env bash
# d710_isolate_stir.sh -- exactly like `d710`, but SIRF/STIR runs in DOCKER.
#
# Why it exists: `attn`, `osem` and `export` are the only commands that need
# SIRF, and SIRF on the host has to be installed deep into the system.  Here
# SIRF comes from a prebuilt image (default `sirf-local:0.1`), and the host only
# needs bash + docker.  `lm` and `lowdose` are the other runtime -- PyTomography,
# no SIRF at all -- and are forwarded to `./d710` like everything else.
#
#   ./d710_isolate_stir.sh attn   --case ped
#   ./d710_isolate_stir.sh osem   --case ped [--beds ...] [--iters n]
#   ./d710_isolate_stir.sh export --case ped [--format nifti|dicom] [--lm]
#
# Every OTHER command (decode / estimate / tostir / exam / lm / lowdose / read /
# shell) is
# forwarded to `./d710` untouched -- they already run inside `d710:full`.  So
# `--tof` / `--no-tof` pass through here unchanged; see `./d710 --help`.
#
# `--tof` / `--no-tof` configure decode + estimate, not osem: osem reads TOF
# from the prompts header itself.  The two commands here SWALLOW them without
# complaint, just like `./d710` -- so that `d710 osem --tof` and
# `d710_isolate_stir.sh osem --tof` behave the same, instead of one staying
# silent while the other dies because `-m osem`'s argparse does not know the flag.
#
# MOUNTED AT THE EXACT HOST PATHS (`-v /x:/x`), with no path translation:
#   $D710_OUT           rw   output tree
#   D710/ source dir    ro   `-m osem` / `-m utils.export` run from here
#   CT directory        ro   taken from `--ct` and from `work/bed<n>/to_stir.json`
# That way the absolute CT paths written in the sidecar and in recon.npz stay
# valid verbatim, both inside the container and when reopened on the host.
#
# Environment variables, all of which can also be set in `D710/.env`
# (copy `.env.example` -- they are identical -- and edit):
#   D710_OUT           root of the output tree (required, or pass --out)
#   D710_CASE          default case name for --case
#   D710_SIRF_IMAGE    SIRF image          (default sirf-local:0.1)
#   D710_SIRF_ENV_SH   SIRF's env script inside the image
#                      (default /opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh)
#   D710_IMAGE         the vendor image (default d710:full) -- `export` reads the
#                      WCC factor from the cal files inside it, so it is needed
#                      here too, and is forwarded into the SIRF container
#   D710_PYTHON        python3 ON THE HOST, only to read sidecars (stdlib, no conda needed)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${D710_ENV_FILE:-$HERE/.env}"
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r k v; do
        k="${k#"${k%%[![:space:]]*}"}"                       # ltrim
        [[ "$k" == export[[:space:]]* ]] && k="${k#export}"
        k="${k//[[:space:]]/}"
        [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue   # comments, blanks
        [[ -v "$k" ]] && continue                            # the shell wins
        v="${v%$'\r'}"; v="${v%%[[:space:]]#*}"              # CRLF, # comment
        v="${v#"${v%%[![:space:]]*}"}"                       # ltrim (KEY = val)
        v="${v%"${v##*[![:space:]]}"}"                       # rtrim
        [[ "$v" == \"*\" || "$v" == \'*\' ]] && v="${v:1:${#v}-2}"
        [[ "$v" == '~/'* ]] && v="$HOME${v:1}"
        export "$k=$v"
    done < "$ENV_FILE"
fi

PY="${D710_PYTHON:-python3}"
SIRF_IMAGE="${D710_SIRF_IMAGE:-sirf-local:0.1}"
SIRF_ENV_SH="${D710_SIRF_ENV_SH:-/opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh}"
IMAGE="${D710_IMAGE:-d710:full}"
export D710_IMAGE="$IMAGE"

die() { echo "error: $*" >&2; exit 2; }
usage() { awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"; }

CMD="${1:-}"
case "$CMD" in
  ""|-h|--help|help) usage; exit $([[ -z "$CMD" ]] && echo 2 || echo 0) ;;
  attn|osem|export) ;;
  # Not a SIRF step -> the original `d710` handles it, untouched.
  *) exec "$HERE/d710" "$@" ;;
esac
shift

# ------------------------------------------------------------------ parse
# Only the flags osem/export use; everything else is forwarded to the module
# (--iters, --subsets, --xy, ...).
CASE="${D710_CASE:-}"; OUT=""; CT=""; BED=""; FORMAT=""; REST=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)    CASE="$2"; shift 2 ;;
    -o|--out)  OUT="$2"; shift 2 ;;
    --ct)      CT="$2"; shift 2 ;;
    --bed)     BED="$2"; shift 2 ;;
    --format)  FORMAT="$2"; shift 2 ;;
    --tof|--no-tof|--collapse-tof) shift ;;      # decode+estimate flags; swallowed like `./d710`
    --tof-mash) shift 2 ;;                       # same, but it takes an argument
    -h|--help) usage; exit 0 ;;
    *)         REST+=("$1"); shift ;;
  esac
done
[[ -n "$CASE" ]] || die "--case is required (or set \$D710_CASE)"

# --out > $D710_OUT > error -- the same rule as `d710`, deliberately no default.
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
  or run the host version instead:
      conda activate petct_reconstruction && $HERE/d710 $CMD --case $CASE"

# --------------------------------------------------------------- mount CT
# The CT directory each bed used: `work/bed<n>/to_stir.json` -> `estimate.ct`,
# exactly where `utils/terms.py:ct_dir()` reads it.  attenuation (osem) and the
# DICOM header (export) reopen them by ABSOLUTE path, so they must be present
# inside the container.
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

# docker refuses two mounts with the same target, and a mount nested inside
# $O/$HERE is redundant.
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

# ------------------------------------------------ docker INSIDE the SIRF container
# `d710 export` reads the WCC factor from the DICOM cal files that live inside
# the `d710:full` image (utils/container.py), i.e. it calls `docker run` itself.
# So the SIRF container must be able to talk to the HOST's docker: mount the
# socket + CLI, and join the socket's group.  This is NOT docker-in-docker: the
# child container is created by the host daemon, so every `-v` of its own is a
# host path -- and because everything here is mounted at identical paths, the
# paths the child sees are still correct.
DOCKER_BIN="$(command -v docker || true)"
GROUPS_ADD=()
if [[ "$CMD" == export && -S /var/run/docker.sock && -n "$DOCKER_BIN" ]]; then
    MOUNTS+=(-v /var/run/docker.sock:/var/run/docker.sock
             -v "$DOCKER_BIN:/usr/bin/docker:ro")
    GROUPS_ADD=(--group-add "$(stat -c %g /var/run/docker.sock)")
fi

# ------------------------------------------------------------------- run
case "$CMD" in
  attn)   MOD=(utils.attn_main --case "$CASE" ${BED:+--beds "$BED"} ${CT:+--ct "$CT"}) ;;
  osem)   MOD=(osem --case "$CASE" ${BED:+--beds "$BED"} ${CT:+--ct "$CT"}) ;;
  export) MOD=(utils.export --case "$CASE" ${FORMAT:+--format "$FORMAT"}) ;;
esac
MOD+=(${REST[@]+"${REST[@]}"})

TTY=(); [[ -t 1 ]] && TTY=(-t)

# --user: created files belong to the caller, not to root -- a bind mount passes
# the container's uid straight through to the host.
# --entrypoint bash: the image's own entrypoint is Jupyter's start.sh.
# env_sirf.sh has to be sourced by hand: inside the image it is only called from
# jovyan's ~/.bashrc, and this is a non-interactive shell with $HOME set to /tmp.
# --no-healthcheck: the image's HEALTHCHECK pings a Jupyter server on port 8888.
# No Jupyter runs here, so the container would show as "unhealthy" in `docker ps`
# for the whole reconstruction -- a false alarm, turned off to cut the noise.
ARGV=(docker run --rm -i "${TTY[@]}" --no-healthcheck
      --user "$(id -u):$(id -g)" ${GROUPS_ADD[@]+"${GROUPS_ADD[@]}"} -e HOME=/tmp
      -e D710_OUT="$O" -e D710_IMAGE="$IMAGE" -e PYTHONPATH="$HERE" -w "$O"
      ${D710_K:+-e D710_K="$D710_K"} ${D710_K_LM:+-e D710_K_LM="$D710_K_LM"}
      "${MOUNTS[@]}" --entrypoint bash "$SIRF_IMAGE"
      -c '. "$0"; exec python3 -u -m "$@"' "$SIRF_ENV_SH" "${MOD[@]}")

# Print the whole command, same style as utils/container.py: when something goes
# wrong, the mount list is always the first thing worth looking at.
printf '+ %s\n' "${ARGV[*]}" >&2
exec "${ARGV[@]}"
