#!/usr/bin/env bash
# Prediction benchmark (Fig 1d, S2): GEARS, scGPT-ft, PRESAGE (+Perturb-seq),
# Weighted, the linear models, the training mean and the evaluation, submitted as Slurm jobs
# (slurm/*.sbatch) chained with afterok dependencies, in the order of
# README.md "Running".
#
# Usage: bash 4_prediction_benchmark/run.sh
#
# TARGETS="VCC Feng-gw" limits the run to those targets and the GEARS builds
# they need (default: the 12 targets of targets.py).  Memory and time per job
# are those of the paper runs (README.md "Compute").  sbatch takes the account
# and partition from SBATCH_ACCOUNT and SBATCH_PARTITION; GPU_PARTITION, if
# set, is used for the GEARS and scGPT-ft jobs instead.  Conda environments:
# GEARS_ENV, SCGPT_ENV, PRESAGE_ENV (default perturbation-structure-gears,
# -scgpt, -presage); JOB_SETUP, if set, holds shell lines each job runs before
# conda activate.  Logs go to results/logs/slurm.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../utils/run.sh" "$@"

# the job scripts import paths.py and targets.py as top-level modules
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export REPO_ROOT=$REPO
JOB_SETUP=${JOB_SETUP:-}
GEARS_ENV=${GEARS_ENV:-perturbation-structure-gears}
SCGPT_ENV=${SCGPT_ENV:-perturbation-structure-scgpt}
PRESAGE_ENV=${PRESAGE_ENV:-perturbation-structure-presage}
export JOB_SETUP GEARS_ENV SCGPT_ENV PRESAGE_ENV
LOGS=$RESULTS/logs/slurm
mkdir -p "$LOGS"

# Memory and time of the paper runs (README.md "Compute"; ranges at their upper
# end).  Other jobs use the defaults in their template.
declare -A PREP_MEM=(
  [VCC]=500G [Pan-GW-hESC]=500G
  [Replogle-E-k562]=200G [Replogle-GW-k562]=500G [Replogle-E-rpe1]=200G
  [Nadig-HEPG2]=200G [Nadig-JURKAT]=200G [Feng-ts]=500G
  [Feng-gw]=300G [Feng-gw-control-cap]=200G [Nourreddine-GW-ipsc]=600G
  [Huang-HCT116]=900G [Huang-HEK293T]=1000G
)
declare -A SUBSET_MEM=(
  [Huang-HCT116]=350G [Nourreddine-GW-ipsc]=350G [Huang-HEK293T]=550G
)
# target: GEARS memory and time, scGPT-ft memory and time, PRESAGE memory per fold
declare -A TRAIN=(
  [VCC]="500G 50:00:00 500G 50:00:00 120G"
  [Pan-GW-hESC]="400G 40:00:00 320G 72:00:00 160G"
  [Replogle-E-k562]="200G 50:00:00 200G 50:00:00 120G"
  [Replogle-GW-k562]="480G 50:00:00 400G 120:00:00 200G"
  [Replogle-E-rpe1]="200G 50:00:00 200G 50:00:00 120G"
  [Nadig-HEPG2]="200G 25:00:00 200G 25:00:00 64G"
  [Nadig-JURKAT]="200G 25:00:00 200G 25:00:00 64G"
  [Feng-ts]="480G 50:00:00 400G 72:00:00 200G"
  [Feng-gw]="120G 24:00:00 150G 48:00:00 140G"
  [Nourreddine-GW-ipsc]="340G 24:00:00 300G 60:00:00 260G"
  [Huang-HCT116]="340G 24:00:00 300G 60:00:00 260G"
  [Huang-HEK293T]="550G 36:00:00 450G 72:00:00 440G"
)

# after ID...: an afterok dependency on these jobs (empty without any)
after() {
  local IFS=:
  (($#)) && echo "afterok:$*"
  return 0
}

# submit NAME TEMPLATE DEPENDENCY VAR=VALUE [SBATCH_OPTION...]: submit one
# template and print its job ID
submit() {
  local name=$1 template=$2 dependency=$3 variable=$4 log id
  shift 4
  log="$LOGS/${template%.sbatch}_${name}_%j.out"
  grep -q '^#SBATCH --array' "$HERE/slurm/$template" && log="${log%_%j.out}_%A_%a.out"
  local cmd=(sbatch --parsable --export="ALL,$variable" --output="$log")
  [[ $template == train_* && -n ${GPU_PARTITION:-} ]] && cmd+=(--partition="$GPU_PARTITION")
  [[ -n $dependency ]] && cmd+=(--dependency="$dependency")
  cmd+=("$@" "$HERE/slurm/$template")
  echo "+ ${cmd[*]}" >&2
  id=$("${cmd[@]}")
  echo "  job ${id%%;*}" >&2
  echo "${id%%;*}"
}

# The jobs, from targets.py and gears_scgpt/prepare_data.py (- = none):
#   prep BUILD                 a GEARS build and its folds
#   subset PARENT              the five per-fold reduced builds of PARENT
#   target NAME GEARS_BUILD GEARS_JOB PRESAGE_JOBS CONTROL_BUILD CONTROL_JOB
# where *_JOB(S) name the job(s) that write the builds a target reads.
PLAN=$("$PYTHON" - "$HERE" "${TARGETS:-}" <<'EOF'
import sys

sys.path.insert(0, f"{sys.argv[1]}/gears_scgpt")
import prepare_data as gears
import targets

preps, subsets = [], []


def job(build):
    """Add the job that writes GEARS build `build` to the plan; return its key."""
    spec = gears.DATASETS[build.format(fold=1)]
    parent = spec.get("fold_subset_of")
    if parent:
        job(parent)
        if parent not in subsets:
            subsets.append(parent)
        return f"subset:{parent}"
    if build not in preps:
        preps.append(build)
    return f"prep:{build}"


rows = []
for name in sys.argv[2].split() or targets.TARGETS:
    t = targets.get(name)
    presage = sorted({job(targets.presage_build(name)), job(targets.split_build(name))})
    rows.append([name, t.gears_build or "-", job(t.gears_build) if t.gears_build else "-",
                 ",".join(presage), t.control_build or "-",
                 job(t.control_build) if t.control_build else "-"])
for build in preps:
    print("prep", build)
for parent in subsets:
    print("subset", parent)
for row in rows:
    print("target", *row)
EOF
)
mapfile -t ROWS <<<"$PLAN"
declare -A JOB     # plan key -> job ID
declare -A WAIT    # target -> job IDs its evaluation waits for

section "1. GEARS builds and folds"
for row in "${ROWS[@]}"; do
  read -r kind name _ <<<"$row"
  case $kind in
    prep)
      JOB[prep:$name]=$(submit "$name" gears_prep.sbatch "" "DATASET=$name" \
        --mem="${PREP_MEM[$name]}") ;;
    subset)
      JOB[subset:$name]=$(submit "$name" gears_fold_subset.sbatch "$(after "${JOB[prep:$name]}")" \
        "PARENT=$name" --mem="${SUBSET_MEM[$name]}") ;;
  esac
done

section "2. GEARS and scGPT-ft"
for row in "${ROWS[@]}"; do
  read -r kind name build build_job _ <<<"$row"
  [[ $kind == target && $build != - ]] || continue
  read -r gears_mem gears_time scgpt_mem scgpt_time _ <<<"${TRAIN[$name]}"
  dependency=$(after "${JOB[$build_job]}")
  WAIT[$name]="$(submit "$name" train_gears.sbatch "$dependency" "BUILD=$build" \
    --mem="$gears_mem" --time="$gears_time")"
  WAIT[$name]+=" $(submit "$name" train_scgpt.sbatch "$dependency" "BUILD=$build" \
    --mem="$scgpt_mem" --time="$scgpt_time")"
done

section "3. PRESAGE (+Perturb-seq)"
for row in "${ROWS[@]}"; do
  read -r kind name _ _ build_jobs _ <<<"$row"
  [[ $kind == target ]] || continue
  read -r _ _ _ _ presage_mem <<<"${TRAIN[$name]}"
  ids=()
  for key in ${build_jobs//,/ }; do ids+=("${JOB[$key]}"); done
  dependency=$(after "${ids[@]}")
  # singleton: one presage_prep at a time, as the Perturb-seq source files
  # written by step 01 are shared across targets
  prep=$(submit "$name" presage_prep.sbatch "${dependency:+$dependency,}singleton" "PP_DATASET=$name")
  # fold 1 writes the caches that folds 2-5 reuse
  fold1=$(submit "$name-f1" presage_train.sbatch "$(after "$prep")" "PP_DATASET=$name" \
    --array=1 --mem="$presage_mem")
  folds=$(submit "$name-f2-5" presage_train.sbatch "$(after "$fold1")" "PP_DATASET=$name" \
    --array=2-5 --mem="$presage_mem")
  submit "$name" presage_collect.sbatch "$(after "$folds")" "PP_DATASET=$name" >/dev/null
  WAIT[$name]="${WAIT[$name]:-} $fold1 $folds"
done

section "4. evaluation (Weighted, the linear models, the training mean and the pooled control profiles are computed here)"
for row in "${ROWS[@]}"; do
  read -r kind name _ _ _ control control_job <<<"$row"
  [[ $kind == target ]] || continue
  # the pooled control profile is read from the control build
  if [[ $control != - ]]; then WAIT[$name]+=" ${JOB[$control_job]}"; fi
  # shellcheck disable=SC2086  # WAIT holds space-separated job IDs
  submit "$name" evaluate.sbatch "$(after ${WAIT[$name]})" "TARGET=$name" >/dev/null
done
