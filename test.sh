#!/bin/bash

#=======================================================================
# SBATCH -- Directives for the Slurm Scheduler
#=======================================================================
#SBATCH --job-name=test_job_uv_final # Nome do seu job
#SBATCH --output=/home/eduardo.andrade/job_output_%j.out   # Arquivo para salvar a saída padrão
#SBATCH --error=/home/eduardo.andrade/job_error_%j.err     # Arquivo para salvar o erro padrão
#SBATCH --time=01:00:00             # Tempo máximo de execução (hh:mm:ss)
#SBATCH --nodes=1                   # Número de nós
#SBATCH --ntasks=32                 # Número de tarefas
#SBATCH -p cpuq                     # Partição (fila) a ser utilizada
#SBATCH --mem=40G                   # Memória solicitada

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

# --- FINAL FIX: Explicitly provide requirements.txt for offline sync ---
# The --offline flag requires a source file to be named.
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

uv run python notebooks/debug_imports.py

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

uv run python notebooks/QML_CICIDS2017.py


#=======================================================================
# JOB COMPLETION
#=======================================================================
echo "========================================================"
echo "✅ Main script finished."
echo "Job finished successfully at: $(date)"
echo "========================================================"
