#!/usr/bin/env bash
# Move the old flat output layout out of the source tree and into $D710_OUT.
#
#   tools/migrate_out.sh --from ../D710_reconstruction --to ~/UET/d710_out
#   tools/migrate_out.sh --from ../D710_reconstruction --to ~/UET/d710_out --apply
#
# Dry run by default; nothing moves until --apply.
#
# Every step is `mv`, never `cp`.  Source and destination have to be on the
# same filesystem, which makes each move a rename: instant, and no moment where
# 19 GB exists twice.  A cross-device --apply is refused rather than silently
# turning into a multi-hour copy.
#
#   raw_prompt/<ca>             ->  <out>/<ca>/decoded
#   vendor/out/<ca>_bed<n>      ->  <out>/<ca>/vendor/bed<n>
#   work/<ca>_bed<n>            ->  <out>/<ca>/work/bed<n>
#   out/<ca>_bqml.nii.gz  &c    ->  <out>/<ca>/export/
#   out/<ca>_dicom/             ->  <out>/<ca>/export/dicom
#
# The case name is whatever `raw_prompt/` holds, so `<ca>_bed<n>` is split on
# the LAST `_bed`.  Anything that does not match a rule is left where it is and
# listed at the end -- this script never deletes and never guesses.
set -euo pipefail

FROM=""; TO=""; APPLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)    FROM="$2"; shift 2 ;;
    --to)      TO="$2"; shift 2 ;;
    --apply)   APPLY=1; shift ;;
    --dry-run) APPLY=0; shift ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$FROM" && -n "$TO" ]] || { echo "error: --from and --to are both required" >&2; exit 2; }
[[ -d "$FROM" ]] || { echo "error: no such directory: $FROM" >&2; exit 2; }

FROM="$(cd "$FROM" && pwd)"
mkdir -p "$TO"
TO="$(cd "$TO" && pwd)"

if [[ "$(stat -c %d "$FROM")" != "$(stat -c %d "$TO")" ]]; then
  echo "error: $FROM and $TO are on different filesystems." >&2
  echo "  Every move would become a full copy of ~19 GB.  Pick a --to on the" >&2
  echo "  same filesystem, or copy by hand and accept the wait." >&2
  exit 3
fi

(( APPLY )) || echo "== DRY RUN -- nothing is moved.  Add --apply to do it."
echo "   from $FROM"
echo "   to   $TO"
echo

moved=0
claimed=()
move() {           # move <src> <dst>
  local src="$1" dst="$2"
  [[ -e "$src" ]] || return 0
  claimed+=("$src")
  if [[ -e "$dst" ]]; then
    echo "  SKIP  $dst already exists"
    return 0
  fi
  printf '  mv    %s\n        -> %s\n' "${src#$FROM/}" "${dst#$TO/}"
  if (( APPLY )); then
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
  fi
  moved=$((moved + 1))
}

shopt -s nullglob

# ---------------------------------------------------------------- decoded
for d in "$FROM"/raw_prompt/*/; do
  ca="$(basename "$d")"
  move "${d%/}" "$TO/$ca/decoded"
done

# ------------------------------------------------- vendor terms, per bed
for d in "$FROM"/vendor/out/*_bed*/; do
  b="$(basename "$d")"
  ca="${b%_bed*}"; n="${b##*_bed}"
  move "${d%/}" "$TO/$ca/vendor/bed$n"
done

# -------------------------------------------------- STIR terms, per bed
for d in "$FROM"/work/*_bed*/; do
  b="$(basename "$d")"
  ca="${b%_bed*}"; n="${b##*_bed}"
  move "${d%/}" "$TO/$ca/work/bed$n"
done

# ---------------------------------------------------------------- export
for f in "$FROM"/out/*_bqml.nii.gz "$FROM"/out/*_suvbw.nii.gz; do
  b="$(basename "$f")"
  move "$f" "$TO/${b%%_*}/export/$b"
done
for d in "$FROM"/out/*_dicom/; do
  b="$(basename "${d%/}")"
  move "${d%/}" "$TO/${b%_dicom}/export/dicom"
done

echo
echo "== $moved move(s)$( (( APPLY )) || echo ' would be made')"

# --------------------------------------------------------------- leftovers
echo
echo "== not claimed by any rule (left exactly where it is):"
left=0
for p in "$FROM"/out/* "$FROM"/vendor/out/* "$FROM"/work/* "$FROM"/raw_prompt/*; do
  [[ -e "$p" ]] || continue
  for c in ${claimed[@]+"${claimed[@]}"}; do
    [[ "$c" == "$p" ]] && continue 2
  done
  printf '  %8s  %s\n' "$(du -sh "$p" 2>/dev/null | cut -f1)" "${p#$FROM/}"
  left=$((left + 1))
done
(( left )) || echo "  (none)"
echo
echo "Nothing was deleted.  Once you are happy, remove what is left by hand."
