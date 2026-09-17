. "${D710_SIRF_ENV_SH_REAL:-/opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh}"

if [ -n "${D710_OUT:-}" ] && [ -d "${D710_OUT}/.pylibs" ]; then
    PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${D710_OUT}/.pylibs"
    export PYTHONPATH
fi
