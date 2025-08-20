#!/bin/bash

#=======================================================================
# SBATCH -- GPU job for A/B QML tests with PennyLane
#=======================================================================
#SBATCH --job-name=qab_uv_abtest_gpu
#SBATCH --output=/home/eduardo.andrade/job_output_gpu_%j.out
#SBATCH --error=/home/eduardo.andrade/job_error_gpu_%j.err
#SBATCH --time=01:00:00
#SBATCH -p gpuq
#SBATCH --nodes=1
#SBATCH -p gpuq
#SBATCH --gpus=2
#SBATCH --mem=256G

#=======================================================================
# Bash options
#=======================================================================
set -e
set -u

#=======================================================================
# Threading limits for BLAS stacks
#=======================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "========================================================"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Running on: $(hostname)"
echo "Job started: $(date)"
echo "========================================================"

cd ${SLURM_SUBMIT_DIR:-$(pwd)}
echo "Working directory: $(pwd)"

echo "Loading system modules..."
module load python
echo "Modules loaded."

echo "Syncing Python environment with uv..."
uv pip sync --offline requirements.txt
echo "Environment synced."

echo "Running import sanity check..."
uv run python scripts/debug_imports.py

#=======================================================================
# A/B GPU RUN
#=======================================================================
echo "Launching GPU A/B shards via srun..."

# Derive shard count from GPUs to avoid oversubscription
GPUS=${SLURM_GPUS_ON_NODE:-0}
if [[ "$GPUS" -eq 0 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L | wc -l)
  fi
fi
SHARDS_PER_GPU=${SHARDS_PER_GPU:-8}
if [[ "$GPUS" -gt 0 ]]; then
  TOTAL_SHARDS=$(( GPUS * SHARDS_PER_GPU ))
else
  TOTAL_SHARDS=${SLURM_NTASKS:-64}
fi
echo "GPUs detected: ${GPUS} | Total shards: ${TOTAL_SHARDS} (per GPU: ${SHARDS_PER_GPU})"

mkdir -p logs

# Select GPU device for PennyLane (Lightning GPU if available)
export QML_DEVICE=${QML_DEVICE:-lightning.gpu}

srun --ntasks=${TOTAL_SHARDS} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} bash -c '
  export AB_NUM_SHARDS='${TOTAL_SHARDS}';
  export AB_SHARD_INDEX=${SLURM_PROCID};
  export AB_MAX_WORKERS=1;  # one worker per shard to favor GPU utilization
  export AB_SAMPLE=${AB_SAMPLE:-20000};
  uv run -q python -m scripts.ab_test > logs/ab_gpu_shard_${SLURM_PROCID}.out 2>&1
'

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
with open('logs/ab_results_merged_gpu.json', 'w') as out:
    json.dump(merged, out, indent=2)
print('Merged', len(files), 'shards into logs/ab_results_merged_gpu.json with', len(merged), 'rows')
PY

echo "Generating consolidated summary..."
uv run -q python - <<'PY'
import json, os
def pretty_row(cols, widths):
    return ' | '.join(str(c).ljust(w) for c, w in zip(cols, widths))
with open('logs/ab_results_merged_gpu.json') as f:
    rows = json.load(f)
headers = ['Encoder','Hadamard','Reupload','AngleScale','Meas','Layers','Acc','Prec','Rec','F1','Log']
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
print('\n===== Consolidated A/B GPU Summary =====')
print(pretty_row(headers, widths))
print(sep)
for row in tbl:
    print(pretty_row(row, widths))
print('=======================================\n')
PY

echo "✅ GPU job finished at: $(date)"

