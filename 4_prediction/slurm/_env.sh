# Sourced by the job templates: paths, conda activation, thread count.
# 4_prediction/run.sh submits the templates with REPO_ROOT and JOB_SETUP
# (shell lines run before conda activate) exported.  To submit one by hand,
# run sbatch from the repository root (mkdir -p logs first).  The conda
# environment of a job is GEARS_ENV / SCGPT_ENV / PRESAGE_ENV if set, else
# perturbation-structure-gears / -scgpt / -presage (built from envs/*.txt).
# sbatch exports the submitting shell's environment.
REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PRED="${REPO}/4_prediction"
if [ ! -f "${PRED}/paths.py" ]; then
    echo "submit from the repository root, or set REPO_ROOT" >&2
    exit 1
fi
# paths.py and targets.py are imported as top-level modules; paths.py needs utils
export PYTHONPATH="${PRED}:${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

# conda_env KEY: $<KEY>_ENV, else perturbation-structure-KEY
conda_env() {
    local override="${1^^}_ENV"
    echo "${!override:-perturbation-structure-$1}"
}

activate() {
    # JOB_SETUP and conda's activation hooks are not `set -u` safe
    set +u
    eval "${JOB_SETUP:-}"
    local target
    target=$(conda_env "$1")
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${target}"
    set -u
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
