#!/usr/bin/env bash
# One night of GATE on a real case, then the comparison with the real data.
#
#   export D710_OUT=~/UET/Handson_PET_CT_Reconstruction/d710_out
#   export D710_PYTHON=~/miniconda3/envs/petct_recon/bin/python
#   nohup D710/simulation/overnight.sh > /dev/null 2>&1 &
#
# Beds are simulated in the order of $BEDS, each to $SIM_SECONDS of simulated
# time (9 s = the frame of <case>_lowcount_time), in 1 s chunks. A chunk is
# started only if it can end before the deadline ($HOURS from now, less the
# time the post-processing needs). Everything is resumable: run the script
# again the next night and it picks up at the first missing chunk.
#
# Every time a bed reaches $SIM_SECONDS the finished beds are converted,
# reconstructed with `d710 lm recon` (their own randoms and scatter, no vendor
# software), exported and compared -- so a result exists even if the night
# ends half way through the next bed.
#
# The default beds are the head and neck (7) and the one below it (6): bed 1
# is the thighs, with little to compare, and it is the one bed whose
# out-of-field activity (the legs) is missing from the PET image. Nothing is
# missing above the head.
#
# Measured on this laptop (Ryzen 7 H 255, 16 threads): one simulated second of
# one bed takes ~30.5 min, the 0.5 s singles run ~19 min. With the defaults:
#   bed 7: singles + 9 chunks   ~4.9 h, then its results (~15 min)
#   bed 6: singles + chunks     until the deadline (~4-5 of 9 by the 8 h mark);
#          run the script again the next night to finish it

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D710="$HERE/d710"

CASE="${CASE:-fdg26081008}"
LOWCOUNT="${LOWCOUNT:-${CASE}_lowcount_time}"
BEDS="${BEDS:-7 6}"
SIM_SECONDS="${SIM_SECONDS:-9}"
HOURS="${HOURS:-8}"
SEED="${SEED:-1}"
THREADS="${THREADS:-16}"
POST_RESERVE_S="${POST_RESERVE_S:-1200}"

: "${D710_OUT:?set D710_OUT to the output root}"
export D710_OUT
export D710_PYTHON="${D710_PYTHON:-python3}"
PY="$D710_PYTHON"

SIM="${CASE}_sim_gate_s${SEED}"
SIMDIR="$D710_OUT/$SIM"
ROOT="$D710_OUT/${CASE}_sim"
mkdir -p "$ROOT"
LOG="$ROOT/overnight_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

START=$(date +%s)
DEADLINE=$(( START + $(printf '%.0f' "$(echo "$HOURS * 3600" | bc -l)") ))

say() { echo "[$(date '+%F %T')] $*"; }
left() { echo $(( DEADLINE - $(date +%s) )); }

chunks_done() {
    local d="$SIMDIR/raw_simulation/gate/bed$1" n=0 k
    for ((k = 0; k < SIM_SECONDS; k++)); do
        [[ -f "$d/$(printf 'chunk_%03d' "$k")/run.json" ]] && n=$((n + 1))
    done
    echo "$n"
}

last_wall() {
    local w
    w=$(ls -t "$SIMDIR"/raw_simulation/gate/bed*/chunk_*/run.json 2>/dev/null | head -1)
    if [[ -n "$w" ]]; then
        "$PY" -c "import json,sys; print(int(json.load(open(sys.argv[1]))['wall_s']))" "$w"
    else
        echo 1900
    fi
}

POSTED=""
post_process() {
    local done_beds=() b
    for b in $BEDS; do
        [[ "$(chunks_done "$b")" -ge "$SIM_SECONDS" ]] && done_beds+=("$b")
    done
    [[ ${#done_beds[@]} -gt 0 ]] || { say "post: no bed has all $SIM_SECONDS s yet"; return 0; }
    [[ "${done_beds[*]}" != "$POSTED" ]] || { say "post: beds ${done_beds[*]} already processed"; return 0; }
    POSTED="${done_beds[*]}"
    say "post: converting beds ${done_beds[*]} (GATE runs are kept, only the case is rebuilt)"
    rm -rf "$SIMDIR/decoded" "$SIMDIR/work" "$SIMDIR/export" \
           "$SIMDIR/simulation.json" "$SIMDIR/recon_lm.npz"
    for b in "${done_beds[@]}"; do
        "$D710" simulate gate --case "$CASE" --bed "$b" --seconds "$SIM_SECONDS" \
            --seed "$SEED" --threads "$THREADS" --convert-only
        "$D710" lm check --case "$SIM" --bed "$b" | tail -1 \
            || say "warning: lm check failed on $SIM bed $b"
    done
    say "post: reconstructing with the simulation's own randoms and scatter"
    "$D710" lm recon --case "$SIM"
    "$D710" export --case "$SIM" --lm --format nifti
    say "post: raw data against the real exam"
    "$D710" simulate compare --case "$CASE" --sims "gate_s$SEED"
    say "post: images against the real low-count scan and the full dose"
    PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m simulation.compare_images \
        --case "$CASE" --sim "gate_s$SEED" --lowcount "$LOWCOUNT"
}

say "case $CASE, beds [$BEDS] to $SIM_SECONDS s each, deadline $(date -d "@$DEADLINE" '+%F %T'), log $LOG"

for b in $BEDS; do
    if ! "$D710" lm check --case "$CASE" --bed "$b" | tail -1 | grep -q YES; then
        say "error: $CASE bed $b fails 'd710 lm check' -- decoded before the 2026-09-18"
        say "  ring-pairing fix. Decode it again, then tostir and attn, and rerun this."
        exit 1
    fi
done
say "lm check: every bed bit-exact"

for b in $BEDS; do
    if [[ ! -f "$ROOT/phantom/bed$b/phantom.json" ]]; then
        say "phantom bed $b"
        "$D710" simulate phantom --case "$CASE" --bed "$b"
    fi
done

stopped=0
for b in $BEDS; do
    have=$(chunks_done "$b")
    say "bed $b: $have of $SIM_SECONDS s simulated"
    for ((s = have + 1; s <= SIM_SECONDS; s++)); do
        need=$(( $(last_wall) + POST_RESERVE_S ))
        [[ -f "$SIMDIR/raw_simulation/gate/bed$b/singles/run.json" ]] || need=$(( need + 1300 ))
        if (( $(left) < need )); then
            say "bed $b: stopping before second $s -- $(( $(left) / 60 )) min left, a chunk needs ~$(( need / 60 ))"
            stopped=1
            break 2
        fi
        say "bed $b: simulating second $s of $SIM_SECONDS"
        "$D710" simulate gate --case "$CASE" --bed "$b" --seconds "$s" \
            --seed "$SEED" --threads "$THREADS" --no-convert
    done
    say "bed $b: done"
    post_process
done

if (( stopped )); then post_process; fi
say "finished in $(( ($(date +%s) - START) / 60 )) min; results in $ROOT/compare"
