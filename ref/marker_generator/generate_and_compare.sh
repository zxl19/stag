#!/usr/bin/env bash
#
# One-click STag marker generation + ground-truth comparison.
#
# Generates the markers for the given HD family with marker_generator.py, then
# compares them against the ground-truth folder (HD<hd>) using
# compare_markers.py and writes side-by-side diff images.
#
# Usage:
#   ./generate_and_compare.sh --HD=23         # generate HD23 and compare
#   ./generate_and_compare.sh --HD=all        # generate every HD family and compare
#   ./generate_and_compare.sh --HD=11,15,23   # several families at once
#   ./generate_and_compare.sh                 # same as --HD=all
#
# Requirements:
#   - marker_generator.py and compare_markers.py in the same directory
#   - a ground-truth folder named HD<hd> next to this script
#   - Python 3 with numpy, opencv-python, pillow, tqdm

set -euo pipefail

# resolve the directory containing this script (works regardless of CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# coloured, severity-tagged logging helpers
# ---------------------------------------------------------------------------

# disable colour when stdout is not a TTY (piped / redirected)
if [ -t 1 ]; then
    C_RED=$'\033[31m'
    C_YELLOW=$'\033[33m'
    C_GREEN=$'\033[32m'
    C_BLUE=$'\033[34m'
    C_RESET=$'\033[0m'
else
    C_RED="" C_YELLOW="" C_GREEN="" C_BLUE="" C_RESET=""
fi

# info  -> green   (normal progress)
# warn  -> yellow  (non-fatal anomaly)
# error -> red     (fatal / failure)
info()  { printf '%b[INFO]%b  %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn()  { printf '%b[WARN]%b  %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
error() { printf '%b[ERROR]%b %s\n' "$C_RED" "$C_RESET" "$*" >&2; }

# ---------------------------------------------------------------------------
# all supported HD families
# ---------------------------------------------------------------------------
ALL_HD=(11 13 15 17 19 21 23)

# parse the --HD=... option (defaults to "all")
HD_VALUE="all"
for arg in "$@"; do
    case "$arg" in
        --HD=*)
            HD_VALUE="${arg#--HD=}"
            ;;
        --HD)
            error "use --HD=<value> (with '='); e.g. --HD=23"
            exit 1
            ;;
        *)
            error "unknown argument '$arg'"
            exit 1
            ;;
    esac
done

# expand the --HD value into a list of families
HD_LIST=()
if [ "$HD_VALUE" = "all" ]; then
    HD_LIST=("${ALL_HD[@]}")
else
    IFS=',' read -r -a HD_LIST <<< "$HD_VALUE"
fi

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
info "Processing ${#HD_LIST[@]} HD family/families: ${HD_LIST[*]}"

for hd in "${HD_LIST[@]}"; do
    echo "=============================================================="
    info "Generating markers for HD${hd} ..."
    python3 marker_generator.py --HD="$hd" || {
        error "marker generation failed for HD${hd}, aborting."
        exit 1
    }

    TRUTH_DIR="HD${hd}"
    if [ ! -d "$TRUTH_DIR" ]; then
        warn "ground-truth folder '${TRUTH_DIR}' not found, skipping comparison."
        continue
    fi

    info "Comparing against '${TRUTH_DIR}' ..."
    python3 compare_markers.py \
        --generated "HD${hd}_generated" \
        --truth "$TRUTH_DIR" \
        --out "HD${hd}_diff" || {
        error "comparison failed for HD${hd}, aborting."
        exit 1
    }
done

echo "=============================================================="
info "Done."
