QML EdgeIIoT Benchmark — Quickstart

Main entrypoint: `scripts/QML_ML-EdgeIIoT-benchmark.py`
Uses all 61 raw features → quantile uniformization → supervised PLS → 8 components for both RF and QML. Models are saved to `models/` per run.

Recommended explore sweep (≈100k train / 20k test)

```bash
EDGE_PHASE=explore EDGE_SWEEP_SAMPLE=120000 EDGE_TEST_SIZE=0.1666667 EDGE_STRATIFY=1 EDGE_EXPLORE_COUNT=8 EDGE_EXPAND_COUNT=0 PYTHONPATH=/home/pichau/QAgents uv run python /home/pichau/QAgents/scripts/QML_ML-EdgeIIoT-benchmark.py
```

Other useful runs
- Grid sanity check (defaults): `PYTHONPATH=/home/pichau/QAgents EDGE_MODE=grid uv run python /home/pichau/QAgents/scripts/QML_ML-EdgeIIoT-benchmark.py`
- RF baseline (all features → PLS-8): `PYTHONPATH=/home/pichau/QAgents EDGE_MODE=rf EDGE_STRATIFY=1 uv run python -m scripts.QML_ML-EdgeIIoT-benchmark`

Key env overrides
- Disable W&B: `WANDB_DISABLED=1` (skips login/init/logging for all runs)
- Data split: `EDGE_SWEEP_SAMPLE` (default 120000), `EDGE_TEST_SIZE` (fraction, e.g. 0.1667), `EDGE_STRATIFY=1|0`
- Sweep counts: `EDGE_EXPLORE_COUNT` (default 8), `EDGE_EXPAND_COUNT` (default 16)
- Train hyperparams: `EDGE_LR`, `EDGE_BATCH` (default 256 fixed in run), `EDGE_EPOCHS`, `EDGE_CLASS_WEIGHTS`, `EDGE_SEED`
- W&B: `WANDB_PROJECT` (default qml-edgeiiot), `WANDB_ENTITY`, `WANDB_GROUP`, `WANDB_API_KEY`, `WANDB_BASE_URL`

Defaults baked in sweeps
- Sample 120k; encoder `angle_embedding_y` with hadamard; ansatz `strongly_entangling`; reupload sweeps over {False, True}; layers ∈ {4,5,7}; lr ∈ {0.01, 0.04, 0.07, 0.10}; epochs explore=3, expand=6; seeds explore=[42], expand=[42,1337,2024].

Environment setup with uv
- Create venv: `uv venv .venv`
- Activate: `source .venv/bin/activate`
- Install deps: `uv pip install -r requirements.txt`
- Run (examples):
  - Explore sweep: `EDGE_PHASE=explore EDGE_SWEEP_SAMPLE=120000 EDGE_TEST_SIZE=0.1666667 EDGE_STRATIFY=1 EDGE_EXPLORE_COUNT=8 EDGE_EXPAND_COUNT=0 PYTHONPATH=/home/pichau/QAgents uv run python /home/pichau/QAgents/scripts/QML_ML-EdgeIIoT-benchmark.py`
  - Grid: `PYTHONPATH=/home/pichau/QAgents EDGE_MODE=grid uv run python /home/pichau/QAgents/scripts/QML_ML-EdgeIIoT-benchmark.py`
  - RF baseline: `PYTHONPATH=/home/pichau/QAgents EDGE_MODE=rf EDGE_STRATIFY=1 uv run python -m scripts.QML_ML-EdgeIIoT-benchmark`