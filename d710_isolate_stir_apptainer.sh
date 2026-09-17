#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/apptainer_shim/setup.sh"

_stock=/opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh
if [[ -z "${D710_SIRF_ENV_SH:-}" || "${D710_SIRF_ENV_SH}" == "$_stock" ]]; then
    export D710_SIRF_ENV_SH="$HERE/apptainer_shim/sirf_env.sh"
fi
unset _stock

CMD="${1:-}"
case "$CMD" in
    doctor) _apt_doctor; exit $? ;;
    ""|-h|--help|help)
        cat <<'USAGE'
d710_isolate_stir_apptainer.sh -- exactly `./d710_isolate_stir.sh`, with
apptainer instead of docker.

  ./d710_isolate_stir_apptainer.sh attn   --case ped
  ./d710_isolate_stir_apptainer.sh osem   --case ped [--beds ...] [--iters n]
  ./d710_isolate_stir_apptainer.sh export --case ped [--format nifti|dicom] [--lm]

The same shape as d710_apptainer, and for the same reason: this is a wrapper,
not a copy. The mount rule that makes the whole arrangement work -- every path
bound at its own host path (`-v /x:/x`), so that the absolute CT paths written
into to_stir.json and recon.npz remain valid on both sides -- is subtle,
already correct, and translates to `--bind /x:/x` untouched. This script
therefore only points D710_SIRF_IMAGE at the .sif and puts apptainer_shim/
first on $PATH; the real d710_isolate_stir.sh then runs unchanged, and every
`docker run` it emits becomes `apptainer exec`.

ONE BEHAVIOUR DIFFERS, and only for `export`. The docker version mounts
/var/run/docker.sock into the SIRF container so that utils/quant.py can start a
second container and read the exam's WCC factor from the vendor image's
calibration files. Apptainer inside apptainer does not work, so that lookup
fails here. It rarely matters: `d710 estimate` records wcc_activity_factor in
estimate.json while it still has the vendor image open, and quant.py reads the
sidecar first, so the container is consulted only for beds estimated before
that field existed. It is also not the scale actually applied; K is K_EXPORT in
utils/scanner.py, or $D710_K. If the message

  "could not read <uid>.3dwcc inside the container"

does appear, re-run `d710_apptainer estimate --force --bed n` to refresh the
sidecar.

THE GPU: --nv is added automatically when /dev/nvidiactl exists, but STIR in
sirf-local links a CPU build of libparallelproj, so `osem` will not become
faster until that image is rebuilt against a CUDA parallelproj. The list-mode
path (`d710 lm`) is the one that uses a GPU today, and it runs on the host.

ENVIRONMENT: everything ./d710_isolate_stir.sh --help lists, plus everything
./d710_apptainer --help lists, D710_SIRF_SIF above all.
USAGE
        echo
        echo "========= ./d710_isolate_stir.sh --help ========="
        exec "$HERE/d710_isolate_stir.sh" --help
        ;;
    attn|osem|export)
        _apt_need "sirf-local:0.1" "${D710_SIRF_SIF:-}" D710_SIRF_SIF ;;
    decode|estimate|tostir|exam|read|shell)
        _apt_need "d710:full" "${D710_SIF:-}" D710_SIF ;;
esac

exec "$HERE/d710_isolate_stir.sh" "$@"
