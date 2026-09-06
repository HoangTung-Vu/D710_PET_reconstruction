#!/usr/bin/env bash
# Test driver for the D710 pipeline.
#
#   tests/run_tests.sh              everything that can run here
#   tests/run_tests.sh --no-data    synthetic only, skip the decoded exams
#   tests/run_tests.sh --case ped   only that exam's beds
#
# The normal environment is `petct_recon`, built from `environment.yml`:
#
#   conda activate petct_recon && tests/run_tests.sh
#
# It deliberately has no SIRF, so the STIR-backed tests skip and say so;
# nothing fails.  To actually RUN those, use the environment SIRF was built
# into from source -- and note a bare interpreter path is not enough there,
# the loader needs that environment's LD_LIBRARY_PATH:
#
#   conda activate petct_reconstruction && tests/run_tests.sh
#
# The data-backed tests read `$D710_OUT/<exam>/decoded/` and
# `$D710_OUT/<exam>/work/bed<n>/`, which are patient-derived and live outside
# this tree.  They skip when $D710_OUT is unset or those are not built; build
# them with `d710 exam`.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
py="${D710_PYTHON:-${PYTHON:-python}}"

args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-data)  args+=(--ignore="$here/test_pipeline_data.py"); shift ;;
        --case)     export D710_CASE="$2"; shift 2 ;;
        --case=*)   export D710_CASE="${1#--case=}"; shift ;;
        -h|--help)  sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)          args+=("$1"); shift ;;
    esac
done

cd "$root"

echo "=== environment ==="
"$py" - <<'EOF'
import importlib
for name in ("numpy", "scipy", "pydicom", "nibabel", "stir", "sirf.STIR"):
    try:
        importlib.import_module(name)
        print(f"  [ok]   {name}")
    except Exception as e:
        print(f"  [skip] {name}: {type(e).__name__}")
EOF
echo "  D710_OUT = ${D710_OUT:-(unset -- the data-backed tests will skip)}"

echo
echo "=== decoded exams on disk ==="
PYTHONPATH="$root:$root/vendor${PYTHONPATH:+:$PYTHONPATH}" "$py" - <<'EOF'
import sys
sys.path.insert(0, "tests")
from cases import decoded_beds
beds = decoded_beds()
if not beds:
    print("  (none -- the data-backed tests will skip; run `d710 exam`)")
for b in beds:
    print(f"  {b['case']} bed {b['bed']}: {b['terms']}")
EOF

echo
echo "=== pytest ==="
"$py" -m pytest "${args[@]+"${args[@]}"}" -q
rc=$?

echo
if [ "$rc" = 0 ]; then echo "ALL PASSED"; else echo "FAILURES -- see above"; fi
exit "$rc"
