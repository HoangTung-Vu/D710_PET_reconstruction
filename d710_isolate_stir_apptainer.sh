#!/usr/bin/env bash
# d710_isolate_stir_apptainer.sh -- exactly `./d710_isolate_stir.sh`, with
# apptainer instead of docker.
#
#   ./d710_isolate_stir_apptainer.sh attn   --case ped
#   ./d710_isolate_stir_apptainer.sh osem   --case ped [--beds ...] [--iters n]
#   ./d710_isolate_stir_apptainer.sh export --case ped [--format nifti|dicom] [--lm]
#
# Same shape as d710_apptainer, and for the same reason: this is a WRAPPER, not
# a copy.  The mount rule that makes the whole thing work -- every path bound at
# its own host path (`-v /x:/x`), so the absolute CT paths written into
# to_stir.json and recon.npz stay valid on both sides -- is subtle, already
# right, and translates to `--bind /x:/x` untouched.  So all this script does is
# point D710_SIRF_IMAGE at the .sif and put apptainer_shim/ first on $PATH; the
# real d710_isolate_stir.sh then runs unchanged, and every `docker run` it emits
# comes out as `apptainer exec`.
#
# ONE BEHAVIOUR IS DIFFERENT, and only for `export`.  The docker version mounts
# /var/run/docker.sock into the SIRF container so utils/quant.py can start a
# SECOND container and read the exam's WCC factor out of the vendor image's cal
# files.  Apptainer inside apptainer does not work, so that lookup fails here.
# It almost never matters: `d710 estimate` records wcc_activity_factor in
# estimate.json while it still has the vendor image open, and quant.py reads the
# sidecar FIRST -- the container is only consulted for beds estimated before
# that field existed.  It is also not the scale that is actually applied; K is
# K_EXPORT in utils/scanner.py, or $D710_K.  If you do see
#   "could not read <uid>.3dwcc inside the container"
# re-run `d710_apptainer estimate --force --bed n` to refresh the sidecar.
#
# THE GPU: --nv is added automatically when /dev/nvidiactl exists, but STIR in
# sirf-local links a CPU-built libparallelproj, so `osem` will not get faster
# until that image is rebuilt against a CUDA parallelproj.  The list-mode path
# (`d710 lm`) is the one that uses the GPU today, and it runs on the host.
#
# Environment: everything ./d710_isolate_stir.sh --help lists, plus everything
# ./d710_apptainer --help lists (D710_SIRF_SIF above all).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/apptainer_shim/setup.sh"

# `sirf-local:0.1` ships neither pydicom nor nibabel, and attn/osem/export all
# need them (the CT reader and the DICOM writer).  apptainer_shim/sirf_env.sh
# sources the image's own env_sirf.sh and then adds $D710_OUT/.pylibs to
# PYTHONPATH if it exists, so the gap is filled without rebuilding the image.
# $HERE is bind-mounted at its own path, so this file is readable inside.
# Only the STOCK value is replaced -- unset, or the image path that .env.example
# ships.  A D710_SIRF_ENV_SH pointing anywhere else is somebody's own script and
# is left alone.  (setup.sh has already read .env by this point, so the stock
# value usually arrives set rather than empty, which is why both cases are
# tested.)  The wrapper falls back to the same image path internally, so nothing
# has to be forwarded into the container.
_stock=/opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh
if [[ -z "${D710_SIRF_ENV_SH:-}" || "${D710_SIRF_ENV_SH}" == "$_stock" ]]; then
    export D710_SIRF_ENV_SH="$HERE/apptainer_shim/sirf_env.sh"
fi
unset _stock

CMD="${1:-}"
case "$CMD" in
    doctor) _apt_doctor; exit $? ;;
    ""|-h|--help|help)
        awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
        echo
        echo "========= ./d710_isolate_stir.sh --help ========="
        exec "$HERE/d710_isolate_stir.sh" --help
        ;;
    attn|osem|export)
        _apt_need "sirf-local:0.1" "${D710_SIRF_SIF:-}" D710_SIRF_SIF ;;
    # Not a SIRF step: d710_isolate_stir.sh forwards it to ./d710, which needs
    # the vendor image instead.
    decode|estimate|tostir|exam|read|shell)
        _apt_need "d710:full" "${D710_SIF:-}" D710_SIF ;;
esac

exec "$HERE/d710_isolate_stir.sh" "$@"
