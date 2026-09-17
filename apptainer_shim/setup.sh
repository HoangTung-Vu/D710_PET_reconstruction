
_apt_read_env() {
    local f="$1" k v
    [[ -f "$f" ]] || return 0
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
    done < "$f"
    return 0
}
_apt_read_env "${D710_ENV_FILE:-$HERE/.env}"

_apt_sif_dirs() {
    printf '%s\n' "${D710_SIF_DIR:-}" "$HERE/sif" "$HERE" \
                  "${D710_OUT:-}/sif" "$HOME/sif" "$HOME"
}

_apt_find_sif() {
    local d g p
    while IFS= read -r d; do
        [[ -n "$d" && -d "$d" ]] || continue
        for g in "$@"; do
            shopt -s nullglob
            for p in "$d"/$g; do
                [[ -f "$p" ]] && { printf '%s\n' "$p"; shopt -u nullglob; return 0; }
            done
            shopt -u nullglob
        done
    done < <(_apt_sif_dirs)
    return 1
}

_apt_abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }

if [[ -z "${D710_SIF:-}" ]]; then
    D710_SIF="$(_apt_find_sif 'd710_full.sif' 'd710-full.sif' 'd710.sif' \
                              'd710*full*.sif' 'd710*.sif' || true)"
fi
if [[ -n "${D710_SIF:-}" && -f "$D710_SIF" ]]; then
    D710_SIF="$(_apt_abs "$D710_SIF")"; export D710_SIF
    export D710_IMAGE="$D710_SIF"
fi

export PATH="$HERE/apptainer_shim:$PATH"
[[ -x "$HERE/apptainer_shim/docker" ]] || chmod +x "$HERE/apptainer_shim/docker" 2>/dev/null || true

_apt_bin() {
    local c
    if [[ -n "${D710_APPTAINER_BIN:-}" ]]; then printf '%s\n' "$D710_APPTAINER_BIN"; return 0; fi
    for c in apptainer singularity; do
        command -v "$c" >/dev/null 2>&1 && { printf '%s\n' "$c"; return 0; }
    done
    return 1
}

_apt_need() {
    local what="$1" sif="$2" var="$3"
    _apt_bin >/dev/null || {
        echo "error: no apptainer (or singularity) on \$PATH." >&2
        echo "  point at it with:  export D710_APPTAINER_BIN=/path/to/apptainer" >&2
        exit 2; }
    [[ -n "$sif" && -f "$sif" ]] && return 0
    cat >&2 <<EOF
error: no $what .sif found.
  Looked for it in:
$(_apt_sif_dirs | sed 's/^/      /' | grep -v '^      $')
  Build it on the machine that has the docker image:
      apptainer build d710_full.sif docker-daemon://d710:full
  then say where it is, once, in D710/.env:
      $var=/path/to/<file>.sif
EOF
    exit 2
}

_apt_doctor() {
    local bin rc=0
    echo "== apptainer"
    if bin="$(_apt_bin)"; then
        printf '   %s -- %s\n' "$bin" "$("$bin" --version 2>&1 | head -1)"
    else
        echo "   MISSING (export D710_APPTAINER_BIN=/path/to/apptainer)"; rc=1
    fi
    echo "== image"
    printf '   d710:full  %s\n' "${D710_SIF:-NOT FOUND}"
    [[ -f "${D710_SIF:-}" ]] || rc=1
    if [[ -n "${bin:-}" && -f "${D710_SIF:-}" ]]; then
        echo "== the vendor image"
        if "$bin" exec --cleanenv --writable-tmpfs "$D710_SIF" \
                python3 /opt/custom_tool/ge_rdf_tool.py selftest >/dev/null 2>&1; then
            echo "   decoder selftest: ok (librdf.so.0 loads)"
        else
            echo "   decoder selftest: FAILED -- decode will not work"; rc=1
        fi
        echo "== fakeroot  (needed by \`estimate\`: pet_recon writes systemConfig)"
        if "$bin" exec --fakeroot "$D710_SIF" true >/dev/null 2>&1; then
            echo "   ok"
        else
            echo "   NOT AVAILABLE -- give $(id -un) a line in /etc/subuid and"
            echo "   /etc/subgid, or use the D710_APPTAINER_BIND fallback that"
            echo "   apptainer_shim/docker prints.  decode and tostir are fine"
            echo "   either way; only \`estimate\` needs it."
            rc=1
        fi
    fi
    echo "== the host runtime  (attn, lm, lowdose, export: no container)"
    local py
    py="$(_apt_host_python)"
    if [[ -z "$py" ]]; then
        echo "   no interpreter found -- conda activate petct_recon, or set"
        echo "   D710_PYTHON=/path/to/python"
        rc=1
    else
        printf '   interpreter %s\n' "$py"
        "$py" - <<'PYEOF' || rc=1
import importlib, sys

need = ("numpy", "scipy", "pydicom", "nibabel", "torch", "pytomography",
        "parallelproj")
miss = []
for m in need:
    try:
        importlib.import_module(m)
    except Exception as e:
        miss.append(f"{m} ({type(e).__name__})")
print("   " + ("all present: " + ", ".join(need) if not miss
               else "MISSING: " + ", ".join(miss)))
def driver_cuda():
    """The CUDA version the installed driver supports, as nvidia-smi reports it."""
    import re
    import subprocess
    try:
        t = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                           timeout=20).stdout
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", t)
    return m.group(1) if m else None

try:
    import torch
    import parallelproj
    drv = driver_cuda()
    print(f"   torch {torch.__version__}  built for CUDA {torch.version.cuda}  "
          f"driver supports {drv or 'no GPU seen'}  "
          f"torch.cuda.is_available() {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   gpu {torch.cuda.get_device_name(0)}")
        if parallelproj.cuda_present is False:
            print("   NOTE parallelproj is the cpu_* build on a machine with a"
                  " GPU.\n        `attn --device cuda` and `lm` will not use"
                  " it.  Switch to\n        libparallelproj=1.10.2=cuda129_*"
                  " (see environment.yml).")
    elif drv:
        tj = int((torch.version.cuda or "0").split(".")[0])
        dj = int(drv.split(".")[0])
        print("   NOTE a GPU is present but torch cannot see it.")
        if tj > dj:
            print(f"        This torch is built for CUDA {torch.version.cuda} "
                  f"and the driver only goes to {drv}.  A CUDA {tj} wheel "
                  f"needs a\n        CUDA {tj} driver; within one major "
                  f"version any 12.x wheel runs on\n        any 12.x driver.  "
                  f"Install a cu12x torch instead:\n"
                  f"          pip install --force-reinstall torch "
                  f"--index-url https://download.pytorch.org/whl/cu126")
except Exception as e:
    print(f"   could not probe the GPU: {type(e).__name__}: {e}")
sys.exit(1 if miss else 0)
PYEOF
    fi
    echo "== gpu inside the container"
    if [[ -e /dev/nvidiactl ]]; then
        echo "   /dev/nvidiactl present -- the shim adds --nv automatically."
        echo "   Nothing in d710:full uses it: pet_recon's GPU path is forced"
        echo "   off in vendor/boot.gdb, GE never having shipped the OpenCL"
        echo "   kernels.  The GPU pays off on the HOST, in attn and lm."
    else
        echo "   no /dev/nvidiactl -- --nv off"
    fi
    echo "== output"
    printf '   D710_OUT=%s\n' "${D710_OUT:-<unset>  (required: export D710_OUT=~/UET/d710_out)}"
    [[ -n "${D710_OUT:-}" ]] || rc=1
    return $rc
}

_apt_host_python() {
    local c
    if [[ -n "${D710_PYTHON:-}" ]]; then printf '%s\n' "$D710_PYTHON"; return 0; fi
    for c in python3 python "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}" \
             "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"; do
        [[ -n "$c" ]] && command -v "$c" >/dev/null 2>&1 && \
            { printf '%s\n' "$c"; return 0; }
    done
    return 0
}
