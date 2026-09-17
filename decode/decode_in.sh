#!/usr/bin/env bash
set -euo pipefail

LISTMODE="${1:-0}"; FORCE="${2:-0}"; TOF="${3:-}"; LM_FORMAT="${4:-npy}"
TOOL=/opt/custom_tool/ge_rdf_tool.py
PYTHON=python3

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
    bed=$((10#$n + 1))
    stem="/out/bed${bed}"
    case "$LM_FORMAT" in npy) ext=lm.npy ;; petsird|prd) ext=prd ;;
      *) echo "error: --format must be npy or petsird" >&2; exit 2 ;; esac
    if [[ $FORCE -eq 0 && -f "$stem.$ext" ]]; then
      echo "=== bed $bed  listmode : cached"; continue
    fi
    echo "=== bed $bed  listmode <- $(basename "$blf")  -> .$ext"
    "$PYTHON" "$TOOL" listmode-decode "$blf" -o "$stem.$ext"
    rm -rf /out/.gerdf_lm

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
