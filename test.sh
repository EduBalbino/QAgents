#!/bin/bash

#=======================================================================
# SBATCH -- Directives for the Slurm Scheduler
#=======================================================================
#SBATCH --job-name=qab_uv_abtest
#SBATCH --output=/home/eduardo.andrade/job_output_%j.out
#SBATCH --error=/home/eduardo.andrade/job_error_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH -p cpuq
#SBATCH --mem=20G

#=======================================================================
# Bash options for safety and debugging
#=======================================================================
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u

#=======================================================================
# Environment Variables to Prevent Deadlocks in Numerical Libraries
#=======================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

#=======================================================================
# JOB SETUP
#=======================================================================

echo "========================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Job started: $(date)"
echo "CPU threads per task: ${SLURM_CPUS_PER_TASK:-1}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "========================================================"

# --- 1. Navigate to Project Directory ---
cd ${SLURM_SUBMIT_DIR}
echo "Working directory: $(pwd)"

# --- 2. Load System Modules ---
echo "--------------------------------------------------------"
echo "Loading system modules..."
module load python
echo "Modules loaded."

# --- 3. Sync Python Environment ---
echo "--------------------------------------------------------"
echo "Verifying environment with uv..."

uv pip sync --offline requirements.txt

echo "Environment is in sync."

#=======================================================================
# STAGE 1: RUN DEBUG SCRIPT TO CHECK IMPORTS
# If this stage fails, 'set -e' will stop the entire job.
#=======================================================================
echo "--------------------------------------------------------"
echo "--- STAGE 1: RUNNING DEBUG SCRIPT ---"
echo "This checks for hangs during library imports."
echo "--------------------------------------------------------"

uv run python scripts/debug_imports.py

echo "--------------------------------------------------------"
echo "--- STAGE 1: DEBUG SCRIPT PASSED SUCCESSFULLY! ---"
echo "--------------------------------------------------------"


#=======================================================================
# STAGE 2: RUN MAIN COMPUTATIONAL SCRIPT
# This stage only runs if Stage 1 was successful.
#=======================================================================
echo "--------------------------------------------------------"
echo "--- STAGE 2: RUNNING MAIN SCRIPT ---"
echo "Now executing the primary computational task."
echo "--------------------------------------------------------"

# Ensure logs dir exists at project root
mkdir -p logs

echo "Executing scripts/QML_ML-EdgeIIoT-Binario.py ..."
# Run from within scripts/ so its '../data/...' path resolves correctly,
# then sync generated logs back to project root.
pushd scripts >/dev/null
PYTHONPATH=.. uv run -q python QML_ML-EdgeIIoT-Binario.py
popd >/dev/null

# Sync logs if the script created them under scripts/logs
if [ -d scripts/logs ]; then
  echo "Syncing logs from scripts/logs to logs/ ..."
  shopt -s nullglob
  for f in scripts/logs/*.log; do
    base=$(basename "$f")
    mv -f "$f" "logs/$base"
  done
  rmdir scripts/logs 2>/dev/null || true
fi


#=======================================================================
# JOB COMPLETION
#=======================================================================
echo "========================================================"
echo " Main script finished."
echo "✅ Main script finished."
echo "Job finished successfully at: $(date)"
echo "========================================================"
