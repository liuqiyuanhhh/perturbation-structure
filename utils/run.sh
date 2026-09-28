# Shared part of the stage scripts <N>_<stage>/run.sh, sourced after
# `set -euo pipefail` with the script's arguments.  Parses the options and
# defines the command helpers.  Every Python script gets the non-path
# parameters of the paper runs; the scripts read data/ and write results/
# (utils/paths.py).
#
# Activate the analysis environment first.  Set
# PYTHON to use another interpreter for the analysis scripts.  py_corum runs a
# script in the conda environment $CORUM_ENV (7_complex_enrichment).

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
STAGE_SCRIPT=${BASH_SOURCE[1]}
HERE=$(cd "$(dirname "$STAGE_SCRIPT")" && pwd -P)
STAGE=$(basename "$HERE")
PYTHON=${PYTHON:-python}

DATA="$REPO/data"
RESULTS="$REPO/results"
CORUM_ENV=${CORUM_ENV:-perturbation-structure-corum}

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# set by the paper jobs of 2_global_trend, 3_cross_dataset_similarity and 6_magnitude_structure
export HDF5_USE_FILE_LOCKING=FALSE

usage() { sed -n '2,/^$/s/^# \{0,1\}//p' "$STAGE_SCRIPT"; }

while (($#)); do
  case $1 in
    -h|--help) usage; exit 0 ;;
    *) echo "$STAGE/run.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# section TITLE: log the start of a group of commands
section() { echo "== [$(date '+%F %T')] $STAGE: $*"; }

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

# py SCRIPT [ARGS...]: run a script in the active (analysis) environment
py() { run "$PYTHON" "$@"; }

# py_corum SCRIPT [ARGS...]: run a script in the conda environment $CORUM_ENV
py_corum() {
  printf '+ [conda activate %s]' "$CORUM_ENV"
  printf ' %q' python "$@"
  printf '\n'
  (
    eval "$(conda shell.bash hook)"
    set +u
    conda activate "$CORUM_ENV"
    set -u
    python "$@"
  )
}

# thread counts of the paper jobs (OMP/OpenBLAS/MKL = CPUs per task)
threads() { export OMP_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 MKL_NUM_THREADS=$1; }

skip() { echo "   skipped: $*"; }
