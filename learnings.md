# Learnings (EdgeIIoT QML): Training Stability, Readout Geometry, and Catalyst Constraints

Date context: this note reflects work in this repo up to **2026-02-10** (America/Fortaleza).

This started as “why is the QML model not learning / too slow” and ended up being two intertwined threads:

1) **Correctness + learning dynamics**: loss plateaus, degenerate thresholds, non-functional readouts, LR schedule instability, and “A/B tests” that were not actually varying what we thought.

2) **Catalyst / circuit compilation constraints**: batching rules, adjoint support, and what is (and is not) differentiable under `qjit` in the pinned stack.

If you only remember one line: most early “model is bad” symptoms were self-inflicted by objective/measurement plumbing, and most “trainable readout” ideas were blocked by Catalyst autodiff limitations (Hamiltonian coefficient gradients).

---

## Terminology and Metrics (so later sections are unambiguous)

**Model score pipeline (as implemented)**

- Quantum forward produces an expval-like scalar `s(x)` (depending on measurement mode).
- Classical head produces a logit `ℓ(x)`:
  - `ℓ(x) = α * (s(x) + bias)` (this is the critical convention to keep training and eval consistent)
- Probability is `p(x) = sigmoid(ℓ(x))`.

**AUC**  
Rank-based; invariant to strictly monotone transforms of `ℓ(x)`. AUC is therefore “mostly threshold-free” and is the primary metric for selecting a checkpoint.

**sep (separation)**  
As logged, `sep = mean(score | y=1) - mean(score | y=0)` with “score” being the model score used for ranking (typically logit-ish quantity). sep is a margin-ish proxy; it can move differently from AUC.

**Degenerate threshold / [DEGEN]**  
This was triggered when the *chosen* threshold produced extreme `pred_pos_rate` (very close to 0 or 1) or when scores were nearly constant, which makes any threshold-based metric brittle and can cause sweep aborts if not handled carefully.

---

## Goals

1. Make short, repeatable A/B tests possible (micro-runs like `epochs=1`, `sample=100000`, optionally `fixed_num_batches=100`).
2. Stop run-killers (no more sweep aborts from `[DEGEN]`, no silent ignoring of overrides, no compiled-cache shape reuse hazards).
3. Identify which knobs actually move **val AUC** and **sep** (primary), while keeping thresholded metrics as secondary.
4. Provide a *trainable readout* that actually learns under Catalyst today.

---

## Initial Symptoms and Diagnosis

### 1) “Loss plateaus at ~0.173” is a focal-loss fixed point (observed + explained)

**Observed:** loss hovering around `0.173`, `sep ~ 0.005`, and outputs behaving like “everything is ~0” (logits near 0 ⇒ `p≈0.5`) is consistent with a focal-loss stall.

For BCE-with-logits at `ℓ=0`:
- `CE(ℓ=0) = log(2) ≈ 0.6931`

For focal with `γ=2`, with `p_t = 0.5` at `ℓ=0`:
- focal factor `(1 - p_t)^γ = (0.5)^2 = 0.25`

So the focal-weighted loss near zero logits is:
- `0.25 * log(2) ≈ 0.1733`

**Inference:** when the model sits near `ℓ≈0` everywhere, focal can reduce gradient signal enough that the optimizer never escapes, especially if other constraints also cap margin (see measurement geometry).

**Important confounder:** when `balanced_batches=1` with `pos_frac=0.5`, the code forces `focal_gamma -> 0.0`. That means any “focal plateau” symptom cannot coexist with balanced batching in those runs; if you see both, you’re looking at a mismatch between what you think ran and what actually ran.

---

### 2) Output geometry was a bottleneck (bounded scalar pretending to be a logit)

With `measurement=mean_z`, the circuit returns a single expval in `[-1, 1]`.

If you treat this as “the logit,” then even with a bias, the raw pre-α score has limited dynamic range. You *can* recover larger logit magnitudes via `α`, but that introduces its own failure mode:

- `α` becomes a **temperature** parameter that can dominate behavior and mask whether the circuit is learning anything meaningful.
- If `α` is allowed to go negative (direct training), it can **invert ranking** (AUC catastrophe) by flipping sign of logits.

This is why getting `α` semantics correct (and constrained) is not optional; it’s core plumbing.

---

### 3) `z_vec` “trainable readout via Hamiltonian coefficients” is gradient-dead under this qjit/Catalyst stack

We tried `measurement=z_vec` with “trainable readout weights” `w_ro`, using:
- `s(x) = ⟨Σ_i w_ro[i] Z_i⟩`

Two probes were decisive:

**(A) Forward sensitivity:** changing `w_ro` changes outputs (so `w_ro` is *used* in the forward).  
**(B) Gradient identity check (must hold mathematically):**
- `∂/∂w_i ⟨Σ_j w_j Z_j⟩ = ⟨Z_i⟩`

**Observed under `qjit`:**
- `grad_w_ro = [0,0,0,0,0,0,0,0]`
- but `⟨Z_i⟩` values are nonzero

**Conclusion (fact for this pinned environment):** Catalyst is not propagating gradients through Hamiltonian coefficients here. So “trainable readout via Hamiltonian coeffs” is *dead*; it can change the forward pass but will not learn.

Diagnostics (created in `/tmp`, not in repo history):
- `/tmp/diag_zvec_wro.py` (forward sensitivity + gradient identity)

---

### 4) Throughput ceiling is imposed at the compiler boundary (and why it matters)

Catalyst cannot `vmap` a `qjit` QNode because the batching rule for `qinst` is missing. That means:

- The “natural” approach (`jax.vmap(qnode)(batch)`) fails.
- We are forced into **serial per-example evaluation inside compiled code** using `jax.lax.scan`.

This sets a hard throughput ceiling: increasing batch size mostly increases serial work, not vectorized speed, and the main knob becomes “how fast can the compiled QNode execute repeatedly,” not “how well is XLA batching.”

**Practical effect:** training is stable/compilable but behaves like an inner loop over batch elements. Speed is therefore bounded even on GPU.

More notes: `catalyst-learnings.md`.

---

## Key Repo Fixes (plumbing that makes experiments real instead of imaginary)

### 1) Make overrides actually apply (micro-runs, grids, and reproducibility)

**Problem:** `train_params` overrides were silently ignored in some harness paths (notably `run_one()`), so “A/B tests” were not actually changing what we thought.

**Fix:** expanded allowlists in `scripts/QML_ML-EdgeIIoT-benchmark.py` so test harness overrides propagate end-to-end:

- micro-run knobs: `fixed_num_batches`, `preflight_compile`, `eval_every_epochs`
- stability knobs: `abort_on_degen`
- distribution knobs: `balanced_batches`, `balanced_pos_frac`, `focal_gamma`
- regularization / head: `weight_decay`, `weight_decay_ro`, LR multipliers
- alpha: `alpha_mode`, `alpha_train`, `lr_mult_alpha`
- selection: `val_objective` (AUC/sep)

**Outcome:** when logs show `[INFO] Using fixed_num_batches=...`, you can trust the run is genuinely in “micro-run mode.”

---

### 2) Stop `[DEGEN]` from killing sweeps (preserve metrics, preserve sweep continuity)

**Problem:** evaluation aborted when threshold selection yielded extreme `pred_pos_rate` or scores were near-constant. This destroyed sweeps mid-grid and made A/B impossible.

**Fix in `scripts/core/builders.py`:**
- `abort_on_degen` is now optional.
- On degeneracy: fall back to a safe default threshold (`thr=0.0` in logit space), set `degen=1`, continue.
- AUC/sep/val_loss are still valid under degeneracy; thresholded metrics are the ones that become untrustworthy.

**Operational rule:** if `degen=1`, trust AUC/sep/val_loss; treat bacc/f1/precision as “diagnostic only.”

---

### 3) Fix compiled-cache hazards (shape must be part of the cache key)

**Problem:** compiled-core caching did not include `batch_size`/`feature_dim`. Reusing a compiled function across shapes is a correctness risk (wrong assumptions baked into IR, potential silent misuse).

**Fix in `scripts/core/compiled_core.py`:**
- cache key includes `batch_size`, `feature_dim`
- also includes `alpha_mode` and measurement signature (so loss semantics cannot accidentally mismatch).

**Outcome:** “parallel” runs or mixed-shape sweeps won’t accidentally reuse the wrong compiled core.

---

## Model / Objective Changes That Actually Matter

### 1) Constrain alpha (`alpha_mode`) and keep training/eval identical

We introduced `alpha_mode`:

- `direct`: `α` trained directly → can cross zero → can invert ranking
- `softplus`: `α = softplus(alpha_param) + 1e-3` → `α > 0` by construction

**Why this matters:** if `α` flips sign, `ℓ(x)` flips sign for every sample, which can invert ranking and trash AUC without any other change. Enforcing `α>0` prevents this pathological degree of freedom.

**Implementation must be consistent in both places:**
- compiled loss (`scripts/core/compiled_core.py`)
- evaluation / reporting (`scripts/core/builders.py`)

Also added `alpha_train=false` for “trust runs” so α cannot compensate for modeling issues (i.e., you’re measuring the circuit + readout, not temperature games).

---

### 2) Provide a trainable readout that works under Catalyst: `mean_z_readout`

Because Hamiltonian-coefficient readout is gradient-dead, the working escape hatch is: **put readout parameters into gates**, where adjoint differentiation is supported.

New measurement mode: `mean_z_readout`

Behavior:
- encoder + ansatz as usual
- then apply a per-wire readout rotation (trainable)
- then measure fixed mean-Z across wires

Implementation details (important for stability):
- used `Rot`-equivalent decomposition (`RZ/RY/RZ`) because Lightning adjoint under Catalyst complained when using `qml.Rot` directly in this stack.
- `readout_dim = 3 * len(measurement.wires)` (each measured wire gets 3 parameters)

Touched files:
- `scripts/core/compiled_core.py`: implements mode + dim
- `scripts/predict.py`: loads/sanity-checks `mean_z_readout` at inference
- `scripts/core/builders.py`: logs `grad_norm_w_ro`, `param_norm_w_ro`, `delta_w_ro` (so you can audit that the head is actually learning)
- `scripts/QML_ML-EdgeIIoT-benchmark.py`: adds `mean_z_readout` to allowed measurements

**Observed diagnostic:** `grad_norm_w_ro` becomes nonzero and `delta_w_ro` accumulates across epochs. This is the minimal sanity check that “the readout is not dead.”

---

## Experiments and What They Showed

### 1) Epoch 1 / sample 100k grid (6 runs): balanced batching dominates, focal hurts under true prior

Log: `logs/grid_epoch1_sample100k.out`  
Setup: `epochs=1`, `sample=100000`, `batch=256`, `seed=42`.

#### Grid 1: `balanced_batches=1` (and therefore focal forced to 0)
Runs 1–4 were identical across:
- `mean_z` vs `z_vec`
- `alpha_mode=direct` vs `softplus`

Metrics:
- `val_auc=0.82134`, `test_auc=0.82727`, `val_sep=0.15289`

**Interpretation (inference consistent with diagnostics):**
- in balanced mode, `z_vec` behaves like a fixed scalar head because `w_ro` cannot learn.
- focal is not being exercised here at all (forced to 0), so it cannot explain any differences.

#### Grid 2: `balanced_batches=0` (true class prior; focal tested)
- Run 5 (`focal_gamma=0`): `val_auc=0.87195`, `test_auc=0.87661`, `val_sep=0.16941`
- Run 6 (`focal_gamma=2`): `val_auc=0.85201`, `test_auc=0.85720`, `val_sep=0.17722`

Numeric implications:
- Turning balanced sampling off (with focal 0):
  - `Δval_auc = +0.05061`, `Δtest_auc = +0.04934`
- Adding focal γ=2 under unbalanced:
  - `Δval_auc = -0.01994`, `Δtest_auc = -0.01941` (sep increases slightly)

**Interpretation:**
- `balanced_batches=1` is not a harmless “stabilizer” when true `pos_rate ~ 0.855`; it changes the effective training distribution/objective and can hurt AUC materially.
- `focal_gamma=2` is too aggressive here under the true prior; it degrades ranking (AUC) even if it increases a margin proxy (sep). This is consistent with focal emphasizing “hard” cases in a way that can distort global ordering.

---

### 2) 5 seeds × 5 epochs micro-run (one-cycle; fixed batches): overshoot signature

Log: `logs/seed5_epoch5.out`

Summary:
- `VAL_AUC mean=0.908089 stdev=0.021066 over 5 seeds`
- with one-cycle, `best_epoch=1` for all 5 seeds by val AUC, and later epochs often degraded.

**Inference:** the one-cycle schedule (with a relatively high peak LR) is inducing overshoot / early overfit in this setup. The “always best at epoch 1” pattern is a red flag that the schedule is too aggressive or mismatched to the model’s effective curvature/conditioning under qjit+scan evaluation.

---

### 3) Paired mean_z vs mean_z_readout (constant LR, unbalanced): started, not yet concluded

Log: `logs/ablation_mean_z_vs_readout.out`

We started the paired ablation under settings designed to remove the one-cycle pathology:
- A: `measurement=mean_z`
- B: `measurement=mean_z_readout`
- `lr_schedule=constant`, `lr=1e-3`
- `balanced_batches=0`, `focal_gamma=0`
- `fixed_num_batches=100`, select by `val_auc`

Seed 1 completed:
- A: `best_val_auc=0.928712`, `test_auc=0.926636`
- B: `best_val_auc=0.928336`, `test_auc=0.926249`
- delta (B − A): `Δval_auc=-0.000376`, `Δtest_auc=-0.000386`

Readout learning signal exists in B:
- `grad_norm_w_ro=4.16e-4`
- `delta_w_ro=0.0405`

**Conclusion so far:** `mean_z_readout` is learnable and moving, but for this single seed it did not improve AUC/sep. Multiple seeds are required to decide whether the added head capacity helps consistently.

---

## What We Believe Now (Actionable Conclusions)

1. Do not use `measurement=z_vec` as “trainable readout” under `qjit`/Catalyst in this repo right now; `w_ro` gradients through Hamiltonian coefficients are dead.
2. Use `measurement=mean_z_readout` if you want a trainable readout head under Catalyst today (trainable gate parameters are in the supported adjoint path).
3. Keep `balanced_batches=0` and `focal_gamma=0` as current default for AUC/sep on this dataset, based on the epoch-1 / 100k grid.
4. Prefer constant LR (or gentle decay) over one-cycle for “trust runs”; one-cycle produced a consistent “best_epoch=1 then degrade” signature in the 5-seed micro-run.
5. Use AUC and sep (and val_loss) as primary selection criteria; thresholded metrics are secondary and fragile under heavy imbalance and degeneracy.

---

## If We Must Use Balanced Batches (how to make it statistically honest)

Balanced sampling changes the effective training prior. If you *need* balanced batches for optimization stability, restore the true-risk objective via importance weights.

Let batch prior be `π_b = 0.5` and true prior be `π_t ≈ pos_rate`.

Weights (up to a common scale factor):
- `w_pos = π_t / π_b`
- `w_neg = (1 - π_t) / (1 - π_b)`

With `π_t ≈ 0.855`:
- `w_pos ≈ 1.71`
- `w_neg ≈ 0.29`

Implementation note: your compiled loss already accepts `wb` per-example weights. The “statistically honest balanced-batches” mode is therefore: keep sampling balanced, but set `wb` according to class label using the weights above (and optionally normalize so mean weight is ~1 to keep loss scale stable).

---

## Repro Commands (all runs via uv)

Common environment:

```bash
cd /home/devpod/QAgents
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

Epoch-1 / 100k grid:

uv run python /tmp/run_ab_grid.py | tee logs/grid_epoch1_sample100k.out

5 seeds × 5 epochs micro-run (historical log):

uv run python /tmp/run_5seed_5epoch_check.py | tee logs/seed5_epoch5.out

Next Minimal Steps (to turn “learnings” into a decision) Finish the paired ablation across 5–10 seeds: mean_z vs mean_z_readout balanced_batches=0, focal_gamma=0 constant LR (or gentle cosine decay) checkpoint by val_objective=auc report mean±stdev of best-val AUC and matched test AUC If readout does not consistently beat baseline, stop spending complexity there and move upstream: encoder scaling / feature mapping ansatz depth and entanglement topology regularization and head-only LR multipliers (if the readout is learning too slowly relative to circuit parameters) Add importance-weighted loss path so balanced_batches=1 becomes testable without changing the objective. Before trusting “high AUC quickly,” run a leakage stress test: grouped split, time-based split, or entity-based split (whatever is appropriate for EdgeIIoT provenance) Appendix: Catalyst Performance Microbench Notes (what matters operationally) The core constraints that shaped the implementation: qjit QNodes cannot be vmap’d due to missing batching rule for qinst; batching must be emulated with lax.scan.catalyst.value_and_grad on the batched quantum loss graph has triggered brittle MLIR lowering failures; using catalyst.grad + separate loss evaluation is more stable. Some gate abstractions (e.g., qml.Rot) have adjoint support pitfalls under Catalyst+Lightning; explicit RZ/RY/RZ decomposition is safer in this stack. As a result, the system behaves like “compiled inner loop + serial batch evaluation,” which bounds throughput even on GPU. See catalyst-learnings.md for the deeper compiler-side notes. If you want this to read more like an engineering incident report (root cause → contributing factors → fixes → verification evidence), I can rewrite the same content into that structure without changing any claims.
