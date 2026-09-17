
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
if [[ -z "${D710_SIRF_SIF:-}" ]]; then
    D710_SIRF_SIF="$(_apt_find_sif 'sirf_local.sif' 'sirf-local.sif' \
                                   'sirf*local*.sif' 'sirf*.sif' || true)"
fi

if [[ -n "${D710_SIF:-}" && -f "$D710_SIF" ]]; then
    D710_SIF="$(_apt_abs "$D710_SIF")"; export D710_SIF
    export D710_IMAGE="$D710_SIF"
fi
if [[ -n "${D710_SIRF_SIF:-}" && -f "$D710_SIRF_SIF" ]]; then
    D710_SIRF_SIF="$(_apt_abs "$D710_SIRF_SIF")"; export D710_SIRF_SIF
    export D710_SIRF_IMAGE="$D710_SIRF_SIF"
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
      apptainer build d710_full.sif  docker-daemon://d710:full
      apptainer build sirf_local.sif docker-daemon://sirf-local:0.1
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
    echo "== images"
    printf '   d710:full      %s\n' "${D710_SIF:-NOT FOUND}"
    printf '   sirf-local:0.1 %s\n' "${D710_SIRF_SIF:-NOT FOUND}"
    [[ -f "${D710_SIF:-}" && -f "${D710_SIRF_SIF:-}" ]] || rc=1
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
            echo "   apptainer_shim/docker prints.  decode/tostir/osem are fine"
            echo "   either way; only \`estimate\` needs it."
            rc=1
        fi
    fi
    if [[ -n "${bin:-}" && -f "${D710_SIRF_SIF:-}" ]]; then
        echo "== sirf-local: gói Python attn/osem/export cần"
        local miss
        local -a probe=(exec --cleanenv --writable-tmpfs --bind "$HERE:$HERE:ro")
        [[ -n "${D710_OUT:-}" && -d "${D710_OUT}" ]] && \
            probe+=(--bind "$D710_OUT:$D710_OUT" --env "D710_OUT=$D710_OUT")
        miss="$("$bin" "${probe[@]}" "$D710_SIRF_SIF" bash -c '
            . "$0" >/dev/null 2>&1
            for m in numpy sirf.STIR pydicom nibabel; do
                python3 -c "import $m" 2>/dev/null || printf "%s " "$m"
            done' "$HERE/apptainer_shim/sirf_env.sh" 2>/dev/null)"
        local m broken="" fixable=""
        for m in numpy sirf.STIR; do
            [[ " $miss " == *" $m "* ]] && broken="$broken $m"
        done
        for m in pydicom nibabel; do
            [[ " $miss " == *" $m "* ]] && fixable="$fixable $m"
        done
        broken="${broken# }"; fixable="${fixable# }"
        if [[ -z "$broken$fixable" ]]; then
            echo "   numpy, sirf.STIR, pydicom, nibabel: ok"
        fi
        if [[ -n "$broken" ]]; then
            echo "   HỎNG: $broken không import được -- image sai, không phải"
            echo "   thiếu gói phụ. Kiểm lại chính file .sif."
            rc=1
        fi
        if [[ -n "$fixable" ]]; then
            echo "   THIẾU: $fixable"
            echo "   Không phải lỗi apptainer -- image vốn không có, docker cũng"
            echo "   hỏng như vậy. Vá bằng bản pure-Python đặt cạnh đầu ra (đã"
            echo "   bind sẵn, nằm ngoài cây mã):"
            echo "       python3 -m pip install --no-deps \\"
            echo "           --target \"\$D710_OUT/.pylibs\" $fixable"
            echo "   apptainer_shim/sirf_env.sh nối thư mục đó vào cuối PYTHONPATH."
            rc=1
        fi
    fi
    echo "== gpu"
    if [[ -e /dev/nvidiactl ]]; then
        echo "   /dev/nvidiactl present -- the shim adds --nv automatically"
        echo "   NOTE this buys nothing yet: STIR in sirf-local links a CPU"
        echo "   parallelproj, and pet_recon's GPU path is forced off in"
        echo "   vendor/boot.gdb (no OpenCL kernels were ever shipped).  The"
        echo "   GPU pays off in \`d710 lm\`, which runs on the HOST in the"
        echo "   petct_recon conda env -- install a CUDA torch there."
    else
        echo "   no /dev/nvidiactl -- --nv off"
    fi
    echo "== output"
    printf '   D710_OUT=%s\n' "${D710_OUT:-<unset>  (required: export D710_OUT=~/UET/d710_out)}"
    [[ -n "${D710_OUT:-}" ]] || rc=1
    return $rc
}
