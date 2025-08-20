#!/bin/bash

#=======================================================================
# SBATCH -- Directives for the Slurm Scheduler
#=======================================================================
#SBATCH --job-name=qab_uv_abtest
#SBATCH --output=/home/eduardo.andrade/job_output_%j.out
#SBATCH --error=/home/eduardo.andrade/job_error_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=3
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

# Run A/B tests in parallel across 3 nodes x 48 tasks (144 shards)
echo "Launching parallel A/B shards via srun..."
TOTAL_SHARDS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
echo "Total shards: ${TOTAL_SHARDS}"

# Ensure logs directory exists on all nodes (shared FS assumed)
mkdir -p logs

# Launch shards
srun --ntasks=${TOTAL_SHARDS} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} bash -c '
  export AB_NUM_SHARDS='${TOTAL_SHARDS}';
  export AB_SHARD_INDEX=${SLURM_PROCID};
  export AB_MAX_WORKERS=12;
  export AB_SAMPLE=1000;
  uv run -q python -m scripts.ab_test > logs/ab_shard_${SLURM_PROCID}.out 2>&1
'

# Merge JSON results
echo "Merging shard results..."
uv run -q python - <<'PY'
import json, glob
files = sorted(glob.glob('logs/ab_results_shard_*.json'))
merged = []
for f in files:
    try:
        with open(f) as fh:
            data = json.load(fh)
            if isinstance(data, list):
                merged.extend(data)
    except Exception as e:
        print('Failed to read', f, e)
with open('logs/ab_results_merged.json', 'w') as out:
    json.dump(merged, out, indent=2)
print('Merged', len(files), 'shards into logs/ab_results_merged.json with', len(merged), 'rows')
PY

# Pretty summary
echo "Generating consolidated summary..."
uv run -q python - <<'PY'
import json, os
def pretty_row(cols, widths):
    return ' | '.join(str(c).ljust(w) for c, w in zip(cols, widths))
with open('logs/ab_results_merged.json') as f:
    rows = json.load(f)
headers = [
    'Encoder','Hadamard','Reupload','AngleScale','Meas','Layers','Acc','Prec','Rec','F1','Log'
]
tbl = []
for r in rows:
    enc_opts = r.get('encoder_opts', {})
    meas = r.get('measurement', {})
    tbl.append([
        r.get('encoder',''),
        str(enc_opts.get('hadamard', False)),
        str(enc_opts.get('reupload', False)),
        str(enc_opts.get('angle_range', enc_opts.get('angle_scale','-'))),
        f"{meas.get('name')}:{','.join(map(str, meas.get('wires',[])))}",
        r.get('layers',''),
        f"{r.get('metrics',{}).get('accuracy',0):.4f}",
        f"{r.get('metrics',{}).get('precision',0):.4f}",
        f"{r.get('metrics',{}).get('recall',0):.4f}",
        f"{r.get('metrics',{}).get('f1',0):.4f}",
        os.path.basename(r.get('log_path',''))
    ])
widths = [max(len(h), *(len(str(row[i])) for row in tbl)) for i, h in enumerate(headers)]
sep = '-+-'.join('-'*w for w in widths)
print('\n===== Consolidated A/B Summary =====')
print(pretty_row(headers, widths))
print(sep)
for row in tbl:
    print(pretty_row(row, widths))
print('===================================\n')
PY


#=======================================================================
# JOB COMPLETION
#=======================================================================
echo "========================================================"
echo "✅ Main script finished."
echo "Job finished successfully at: $(date)"
echo "========================================================"
