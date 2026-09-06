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
    # No --drop-preroll: whether GE's `prompts` scalar counts the pre-roll
    # coincidences VARIES FROM BED TO BED, so no fixed choice is right for a
    # whole exam -- forcing the drop leaves every bed whose scalar does count
    # them short by exactly its pre-roll, and the event table then no longer
    # reproduces bed<n>.s.  Measured on the ped exam: bed 1's scalar excludes
    # its 158, beds 2-6's scalars include theirs (107, 282, 375, 96, 106).
    # The decoder reads the policy off each file's own header instead
    # (gerdf.listmode.resolve_preroll) and records it in the .lm.json sidecar.
    "$PYTHON" "$TOOL" listmode-decode "$blf" -o "$stem.$ext"
    rm -rf /out/.gerdf_lm

    # The count has to reconcile with the header, and an image built before
    # `resolve_preroll` existed decides this per EXAM rather than per bed, so
    # check the result instead of trusting it.  Silently-short event tables are
    # the failure this guards: every invariant still holds, only the image
    # changes.  The .prd path writes no sidecar, so it is skipped.
    if [[ "$ext" == "lm.npy" && -f "$stem.lm.json" && -f "$stem.json" ]]; then
      "$PYTHON" - "$stem" <<'PY' || exit 1
import json, sys
stem = sys.argv[1]
side = json.load(open(f"{stem}.lm.json"))
hdr = json.load(open(f"{stem}.json"))
want = hdr.get("prompts")
got, st = side["events"], side["stats"]
if want is None or got == want:
    sys.exit(0)
how = ("--keep-preroll" if got == want - st["preroll"] else "--drop-preroll")
print(f"error: {stem}.lm.npy holds {got:,} events but the header counts "
      f"{want:,} prompts ({got - want:+,}).", file=sys.stderr)
if abs(got - want) == st["preroll"]:
    print(f"  That is exactly the {st['preroll']:,} pre-roll events. This "
          f"decoder chose keep_preroll={side['keep_preroll']} for the whole "
          f"exam; the policy is per bed.\n"
          f"  Rebuild the image so gerdf.listmode.resolve_preroll is present:\n"
          f"    docker build -t d710:full -f D710/Dockerfile .\n"
          f"  or decode this bed with {how}.", file=sys.stderr)
sys.exit(1)
PY
    fi
  done
fi

echo
echo "decoded into --out:"
ls -1 /out | sed 's/^/  /'
