#!/usr/bin/env bash
# d710_isolate_stir.sh -- y hệt `d710`, nhưng SIRF/STIR chạy trong DOCKER.
#
# Lý do tồn tại: `d710 osem` và `d710 export` là hai lệnh duy nhất cần SIRF, và
# SIRF trên host phải cài đặt sâu vào hệ thống. Ở đây SIRF lấy từ một image dựng
# sẵn (mặc định `sirf-local:0.1`), host chỉ cần bash + docker.
#
#   ./d710_isolate_stir.sh osem   --case ped [--beds ...] [--iters n]
#   ./d710_isolate_stir.sh export --case ped [--format nifti|dicom]
#
# Mọi lệnh KHÁC (decode / estimate / tostir / exam / read / shell) được chuyển
# thẳng cho `./d710` không sửa gì -- chúng vốn đã chạy trong `d710:full`.
#
# MOUNT ĐÚNG ĐƯỜNG DẪN HOST (`-v /x:/x`), không dịch đường dẫn:
#   $D710_OUT           rw   cây đầu ra
#   thư mục mã D710/    ro   `-m osem` / `-m utils.export` chạy từ đây
#   thư mục CT          ro   lấy từ `--ct` và từ `work/bed<n>/to_stir.json`
# Nhờ vậy đường CT tuyệt đối ghi trong sidecar và trong recon.npz vẫn đúng
# nguyên si cả trong container lẫn khi mở lại trên host.
#
# Biến môi trường:
#   D710_OUT           gốc cây đầu ra (bắt buộc, hoặc truyền --out)
#   D710_CASE          tên ca mặc định cho --case
#   D710_SIRF_IMAGE    image SIRF          (mặc định sirf-local:0.1)
#   D710_SIRF_ENV_SH   script env của SIRF trong image
#                      (mặc định /opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh)
#   D710_PYTHON        python3 TRÊN HOST, chỉ để đọc sidecar (stdlib, không cần conda)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${D710_PYTHON:-python3}"
SIRF_IMAGE="${D710_SIRF_IMAGE:-sirf-local:0.1}"
SIRF_ENV_SH="${D710_SIRF_ENV_SH:-/opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh}"

die() { echo "error: $*" >&2; exit 2; }
usage() { awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"; }

CMD="${1:-}"
case "$CMD" in
  ""|-h|--help|help) usage; exit $([[ -z "$CMD" ]] && echo 2 || echo 0) ;;
  osem|export) ;;
  # Không phải bước SIRF -> `d710` gốc lo, không đụng vào.
  *) exec "$HERE/d710" "$@" ;;
esac
shift

# ------------------------------------------------------------------ parse
# Chỉ những cờ mà osem/export dùng; phần còn lại chuyển nguyên cho module
# (--iters, --subsets, --xy, ...).
CASE="${D710_CASE:-}"; OUT=""; CT=""; BED=""; FORMAT=""; REST=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)    CASE="$2"; shift 2 ;;
    -o|--out)  OUT="$2"; shift 2 ;;
    --ct)      CT="$2"; shift 2 ;;
    --bed)     BED="$2"; shift 2 ;;
    --format)  FORMAT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)         REST+=("$1"); shift ;;
  esac
done
[[ -n "$CASE" ]] || die "--case là bắt buộc (hoặc đặt \$D710_CASE)"

# --out > $D710_OUT > lỗi -- cùng một luật với `d710`, cố ý không có mặc định.
O="${OUT:-${D710_OUT:-}}"
[[ -n "$O" ]] || die "không biết ghi đầu ra vào đâu.
  đặt  export D710_OUT=~/UET/d710_out
  hoặc truyền  --out <thư mục>"
O="${O/#\~/$HOME}"
mkdir -p "$O"
O="$(cd "$O" && pwd)"

docker image inspect "$SIRF_IMAGE" >/dev/null 2>&1 || die "không có image SIRF '$SIRF_IMAGE'.
  đổi tên image bằng \$D710_SIRF_IMAGE, hoặc chạy bản host:
      conda activate petct_reconstruction && $HERE/d710 $CMD --case $CASE"

# --------------------------------------------------------------- mount CT
# Thư mục CT mà từng bed đã dùng: `work/bed<n>/to_stir.json` -> `estimate.ct`,
# đúng chỗ `utils/terms.py:ct_dir()` đọc.  attenuation (osem) và header DICOM
# (export) mở lại chúng theo đường TUYỆT ĐỐI, nên chúng phải có trong container.
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

# docker từ chối hai mount trùng đích, và mount lồng trong $O/$HERE là thừa.
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

# --------------------------------------------- docker BÊN TRONG container SIRF
# `d710 export` đọc hệ số WCC từ các file cal DICOM nằm trong image `d710:full`
# (utils/container.py), tức là nó tự gọi `docker run`.  Nên container SIRF phải
# nói chuyện được với docker của HOST: mount socket + CLI, và vào đúng group của
# socket.  Đây KHÔNG phải docker-in-docker: container con do daemon của host tạo,
# nên mọi `-v` của nó là đường dẫn host -- và vì ở đây mount theo đường dẫn đồng
# nhất, đường dẫn con nhìn thấy vẫn đúng.
DOCKER_BIN="$(command -v docker || true)"
GROUPS_ADD=()
if [[ "$CMD" == export && -S /var/run/docker.sock && -n "$DOCKER_BIN" ]]; then
    MOUNTS+=(-v /var/run/docker.sock:/var/run/docker.sock
             -v "$DOCKER_BIN:/usr/bin/docker:ro")
    GROUPS_ADD=(--group-add "$(stat -c %g /var/run/docker.sock)")
fi

# ------------------------------------------------------------------ chạy
case "$CMD" in
  osem)   MOD=(osem --case "$CASE" ${BED:+--beds "$BED"} ${CT:+--ct "$CT"}) ;;
  export) MOD=(utils.export --case "$CASE" ${FORMAT:+--format "$FORMAT"}) ;;
esac
MOD+=(${REST[@]+"${REST[@]}"})

TTY=(); [[ -t 1 ]] && TTY=(-t)

# --user: file sinh ra thuộc về người chạy, không phải root -- bind mount đưa
# thẳng uid của container ra host.
# --entrypoint bash: entrypoint gốc của image là start.sh của Jupyter.
# env_sirf.sh phải source bằng tay: trong image nó chỉ được gọi từ ~/.bashrc của
# jovyan, mà đây là shell không tương tác và $HOME đã đổi thành /tmp.
# --no-healthcheck: HEALTHCHECK của image ping server Jupyter ở cổng 8888.  Ở đây
# không có Jupyter nào chạy, nên container sẽ hiện "unhealthy" trong `docker ps`
# suốt thời gian tái tạo -- báo động giả, tắt đi cho đỡ nhiễu.
ARGV=(docker run --rm -i "${TTY[@]}" --no-healthcheck
      --user "$(id -u):$(id -g)" ${GROUPS_ADD[@]+"${GROUPS_ADD[@]}"} -e HOME=/tmp
      -e D710_OUT="$O" -e PYTHONPATH="$HERE" -w "$O"
      "${MOUNTS[@]}" --entrypoint bash "$SIRF_IMAGE"
      -c '. "$0"; exec python3 -u -m "$@"' "$SIRF_ENV_SH" "${MOD[@]}")

# In ra nguyên lệnh, cùng lối với utils/container.py: khi có gì sai thì cái cần
# nhìn đầu tiên luôn là danh sách mount.
printf '+ %s\n' "${ARGV[*]}" >&2
exec "${ARGV[@]}"
