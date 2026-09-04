#!/usr/bin/env bash
# The per-bed loop, run INSIDE d710:full by decode.sh.  Reads /raw and /lists
# (both read-only), writes /out.
#
# Not meant to be called by hand -- decode.sh sets the mounts up.  It is a
# separate file, mounted at /decode rather than baked into the image, so that
# editing the loop does not mean rebuilding 7 GB.  The decoder it drives IS
# baked in, at /opt/custom_tool.
#
# No decoding happens here.  Every line below is a call into
# custom_tool/ge_rdf_tool.py, which is the only thing that can read GE's legacy
# RDF container.
set -euo pipefail

LISTMODE="${1:-0}"; FORCE="${2:-0}"; TOF="${3:-}"; LM_FORMAT="${4:-npy}"
TOOL=/opt/custom_tool/ge_rdf_tool.py
PYTHON=python3

# Proves librdf.so.0 loads and the helper runs before any file is touched --
# the failure this catches is a decoder that cannot open the container at all,
# which otherwise shows up as an empty result halfway through a long exam.
"$PYTHON" "$TOOL" selftest >/dev/null || {
  echo "error: the vendor decoder is not reachable; run" >&2
  echo "  docker run --rm d710:full python3 $TOOL selftest" >&2
  exit 1; }

shopt -s nullglob
SINOS=(/raw/SINO*)
[[ ${#SINOS[@]} -gt 0 ]] || { echo "error: no SINO* under the raw mount" >&2; exit 2; }

for sino in "${SINOS[@]}"; do
  bed=$("$PYTHON" "$TOOL" info "$sino" --json \
        | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["bed_number"])')
  stem="/out/bed${bed}"
  echo "=== bed $bed  <- $(basename "$sino")"

  "$PYTHON" "$TOOL" info "$sino" --json > "$stem.json"

  if [[ $FORCE -eq 0 && -f "$stem.hs" && -f "$stem.s" ]]; then
    echo "  sinogram : cached"
  else
    # `convert` refuses to report MATCH unless its decoded total equals the
    # header's prompts, so this line is also the count check.
    "$PYTHON" "$TOOL" convert "$sino" -o "$stem.hs" $TOF \
      | tee "$stem.convert.log" | grep -E "MATCH|MISMATCH"
    grep -q MATCH "$stem.convert.log" || {
      echo "  error: count mismatch, see $stem.convert.log" >&2; exit 1; }
  fi

  if [[ $FORCE -eq 0 && -f "$stem.singles.npy" ]]; then
    echo "  singles  : cached"
  else
    "$PYTHON" "$TOOL" singles "$sino" --save "$stem.singles.npy" > "$stem.singles.log"
    echo "  singles  : $(grep -m1 'total singles' "$stem.singles.log")"
  fi
done

if [[ $LISTMODE -eq 1 ]]; then
  for blf in /lists/LIST*.BLF; do
    n=$(basename "$blf" .BLF); n=${n#LIST}
    bed=$((10#$n + 1))                            # LIST0000 is bed 1
    stem="/out/bed${bed}"
    # npy is the event table D710/lm and D710/lowdose read; .prd (PETSIRD) is
    # kept for anyone talking to an external tool.
    #
    # `.lm.npy`, NOT `.npy`: to_npy writes a sidecar beside its output, and a
    # `bed<n>.json` would overwrite the SINO header sidecar written above.
    case "$LM_FORMAT" in npy) ext=lm.npy ;; petsird|prd) ext=prd ;;
      *) echo "error: --format must be npy or petsird" >&2; exit 2 ;; esac
    if [[ $FORCE -eq 0 && -f "$stem.$ext" ]]; then
      echo "=== bed $bed  listmode : cached"; continue
    fi
    echo "=== bed $bed  listmode <- $(basename "$blf")  -> .$ext"
    # GLEPL decompression writes a full-size copy beside the output, NOT beside
    # the input -- /raw is read-only and must stay that way.  cli.py puts it in
    # <out>/.gerdf_lm; it is the size of the .BLF, so it is cleaned up after.
    # --drop-preroll: the preroll coincidences are acquired before the frame
    # starts and GE's own sinogram leaves them out, so keeping them makes the
    # event table disagree with bed<n>.s -- measured, 158 events on ped bed 1
    # and 0 on every other bed.  Every correction term describes the frame, so
    # the events must too.
    "$PYTHON" "$TOOL" listmode-decode "$blf" -o "$stem.$ext" --drop-preroll
    rm -rf /out/.gerdf_lm
  done
fi

echo
echo "decoded into --out:"
ls -1 /out | sed 's/^/  /'
