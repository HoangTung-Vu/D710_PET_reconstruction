# apptainer_shim/sirf_env.sh -- sourced INSIDE sirf-local:0.1, in place of the
# image's own env_sirf.sh.  `d710_isolate_stir.sh` runs
#
#     bash -c '. "$0"; exec python3 -u -m "$@"'  "$D710_SIRF_ENV_SH"  <module> ...
#
# so whatever $D710_SIRF_ENV_SH names is sourced right before python starts.
# `d710_isolate_stir_apptainer.sh` points it here; the real one is still sourced
# first, this only adds to it.
#
# WHY.  `sirf-local:0.1` does not ship pydicom (nor nibabel).  That is not an
# apptainer problem -- the same command under docker fails the same way -- it
# only shows up now because the apptainer route is the first one to run `attn`
# inside the image rather than against a host SIRF.  Three commands need them:
#
#     attn / osem   utils/attenuation.py:53  import pydicom      (reads the CT)
#     export        utils/export.py:89       pydicom + nibabel   (writes them)
#
# The interpreter is NOT the problem and must not be changed: `sirf.STIR` and
# numpy import fine in it, which is the whole reason this container exists.
#
# THE FIX IS PURE PYTHON, INSTALLED BESIDE THE OUTPUT.  pydicom and nibabel are
# pure-Python wheels, so a copy made by any interpreter imports in any other.
# $D710_OUT is already bind-mounted read-write at its own path, so a directory
# in there needs no new mount and lands outside the source tree, which is where
# everything generated belongs:
#
#     python3 -m pip install --no-deps --target "$D710_OUT/.pylibs" pydicom nibabel
#
# --no-deps on purpose: numpy is already in the image, and a second copy on
# PYTHONPATH would shadow the one SIRF was built against.
#
# Nothing happens if that directory does not exist, so this file is safe to
# source whether or not the workaround has been applied.
. "${D710_SIRF_ENV_SH_REAL:-/opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh}"

if [ -n "${D710_OUT:-}" ] && [ -d "${D710_OUT}/.pylibs" ]; then
    # Appended, never prepended: the image's own packages win every collision.
    PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${D710_OUT}/.pylibs"
    export PYTHONPATH
fi
