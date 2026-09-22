#!/usr/bin/env bash
# Everything `python -m lmnet.train` needs, per case:
#   exam (decode + estimate + tostir, apptainer) -> lm check bed 1 -> attn
#   -> lm recon (the labels, work/bed<n>/lm.npz) -> lmnet.train prep
#
#   DATA=~/petct_recon lmnet/prep_cases.sh                  # the five default cases
#   DATA=~/petct_recon lmnet/prep_cases.sh 20260810_FDG26081008_ok
#
# Run from anywhere; needs D710_OUT and the petct_recon env active.  Every step
# skips work already done, so it is safe to Ctrl-C and run again.  A failing
# case is reported and the next one starts.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE" || exit 2
DATA="${DATA:-$HOME/petct_recon}"
: "${D710_OUT:?set D710_OUT first}"

DIRS=("$@")
if [[ ${#DIRS[@]} -eq 0 ]]; then
    DIRS=(20260728_FDG26072803_ok 20260806_FDG26080604_ok
          20260812_FDG26081213_ok 20260819_FDG26081902_ok
          20260810_FDG26081008_ok)
fi

PY="${D710_PYTHON:-python}"
export PYTHONPATH="$HERE:$HERE/vendor${PYTHONPATH:+:$PYTHONPATH}"
# CPUs without AVX2 crash on the real kornia_rs (SIGILL), see README.
if ! "$PY" -c "import kornia_rs" >/dev/null 2>&1; then
    export PYTHONPATH="$PYTHONPATH:$HERE/tools/stubs"
fi
"$PY" -c "from pytomography.projectors.PET import PETLMSystemMatrix" \
    || { echo "error: $PY cannot import pytomography" >&2; exit 2; }

mkdir -p "$D710_OUT"
free_gb=$(df -PBG "$D710_OUT" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
echo "D710_OUT=$D710_OUT  (${free_gb} GB free; ~3 GB per bed, ~7 beds per case)"

one() {
    local hits=()
    mapfile -t hits < <(compgen -G "$1/$2" || true)
    [[ ${#hits[@]} -eq 1 ]] || { echo "expected one '$2' under $1, got ${#hits[@]}" >&2; return 1; }
    printf '%s' "${hits[0]}"
}

step() {
    local name="$1" log="$2"; shift 2
    local s rc=0; s=$(date +%s)
    echo "  -- $name  (log: $log)"
    "$@" >"$log" 2>&1 || rc=$?
    printf '     %s in %dm%02ds\n' "$( ((rc == 0)) && echo ok || echo "FAILED rc=$rc, tail:")" \
        $(( ($(date +%s) - s) / 60 )) $(( ($(date +%s) - s) % 60 ))
    ((rc == 0)) || tail -n 15 "$log" | sed 's/^/     | /'
    return "$rc"
}

FAILED=()
for d in "${DIRS[@]}"; do
    src="$DATA/$d"
    name=$(sed -E 's/^[0-9]+_([A-Za-z]+[0-9]+)(_ok)?$/\1/' <<<"$d" | tr 'A-Z' 'a-z')
    echo
    echo "=== $name   ($src)   $(date '+%H:%M:%S')"
    if [[ ! -d "$src" ]]; then
        echo "  missing $src"; FAILED+=("$name"); continue
    fi
    (
        raw=$(one "$src" 'raw/petRDFS/*/*/*') || exit 1
        lists=$(one "$src" 'raw/petLists/*/*/*') || exit 1
        ct=$(one "$src" 'dicom/CT_s002_*') || exit 1
        log="$D710_OUT/$name/logs"; mkdir -p "$log"

        step "exam" "$log/lmnet_1_exam.log" \
            ./d710_apptainer exam --case "$name" --raw "$raw" --ct "$ct" \
                --listmode --lists "$lists" || exit 1
        step "lm check bed 1" "$log/lmnet_2_check.log" \
            ./d710 lm check --case "$name" --bed 1 || exit 1
        step "attn" "$log/lmnet_3_attn.log" \
            ./d710 attn --case "$name" --device cuda || exit 1
        step "lm recon (labels)" "$log/lmnet_4_lm_recon.log" \
            ./d710 lm recon --case "$name" --resume || exit 1
        step "lmnet.train prep" "$log/lmnet_5_prep.log" \
            "$PY" -m lmnet.train prep --cases "$name" || exit 1
        grep -E "bed [0-9]+:" "$log/lmnet_5_prep.log" | sed 's/^/    /'
    ) || FAILED+=("$name")
done

echo
if ((${#FAILED[@]})); then
    echo "FAILED: ${FAILED[*]}"; exit 1
fi
echo "all prepared.  next:  python -m lmnet.train train --name trial1"
