# Catalyst Learnings

## Snapshot
- Date context: this note reflects work done in this repo on 2026-02-06.
- Goal: stabilize/accelerate the compiled training path (`scripts/core/builders.py`, `scripts/core/compiled_core.py`) and avoid Catalyst/JAX integration pitfalls.
- Most important outcome: we now have a working path again, but the biggest speed ceiling is still architectural (`jax.jit`/`jax.vmap` driving `qjit` calls) rather than one-line tuning.

## Hard findings (important for next LLM)

### 1) `Rot` + adjoint + lightning.gpu is a real blocker
- Error observed repeatedly:
  - `AdjointJacobianGPU.hpp ... operation is not supported using the adjoint differentiation method`
- Trigger: `qml.Rot` directly or via `qml.StronglyEntanglingLayers`.
- Reliable workaround: decompose `Rot(phi, theta, omega)` into `RZ(phi) -> RY(theta) -> RZ(omega)` in circuit order.
- Implemented in `scripts/core/compiled_core.py`.

### 2) Catalyst docs explicitly warn about `vmap` placement
- From `catalyst-ref/doc/dev/sharp_bits.rst`:
  - `jax.vmap(qjit(circuit))` can be used from Python as a workaround.
  - `qjit(jax.vmap(circuit))` is not supported for quantum ops (`Batching rule for 'qinst' not implemented`).
- This matches what we saw: batching at the JAX layer around qjit does not become a fused quantum batch kernel.

### 3) MLIR/LLVM inspection confirmed batch is outside kernel
- Generated files in `qnode_forward/` (with `keep_intermediate=True`) show:
  - single-sample kernel launch signature (`tensor<8xf32> -> tensor<f64>`) in `qnode_forward/0_qnode_forward.mlir`.
  - many scalarized `slice/extract` ops (heavy unrolling), not true batched quantum execution.
  - runtime init/release calls visible in lowered LLVM (`device_init`, `device_release`) in `qnode_forward/6_AfterLLVMIRTranslation.ll`.
- Practical implication: `jax.vmap`/`lax.map` over qjit still means repeated kernel invocation overhead.

### 4) Forcing `jax_enable_x64=False` caused Catalyst compile failure here
- Attempting global toggle in `builders.run` caused MLIR type mismatch:
  - `tensor.extract ... tensor<i32> -> i64` verification failure.
- We reverted that toggle.
- Keep float32 arrays and device dtype tuning, but do not force this global flag in current env.

### 5) Best-practice from Catalyst docs/benchmarks
- `catalyst-ref/benchmark/batchrun.py` is just a runner.
- Real patterns are in:
  - `catalyst-ref/benchmark/catalyst_benchmark/measurements.py`
  - `catalyst-ref/benchmark/catalyst_benchmark/test_cases/*.py`
- Guidance consistent across docs:
  - compile once, run many.
  - use Catalyst control flow (`for_loop`, `while_loop`) and `catalyst.grad` inside qjit for fully compiled hybrid loops.
  - avoid JAX transforms over quantum processing inside qjit.

### 6) `catalyst.value_and_grad` over batched quantum loss is the crash trigger (reproduced)
- Minimal reproducer added: `scripts/debug/catalyst_batch_ad_repro.py`.
- Cases that compile/run (both `lightning.qubit` and `lightning.gpu`):
  - `sample_grad`
  - `batch_grad_map` (`catalyst.grad` over a `lax.map` batch loss)
  - `batch_grad_dynamic_slice`
  - `batch_grad_map_optax`
- Case that hard-crashes compiler:
  - `batch_vg_map` (`catalyst.value_and_grad` over the same batch loss)
- Crash signature:
  - `memref::SubViewOp::inferResultType ... staticOffsets length mismatch`
  - abort inside one-shot bufferization (`InsertSliceOpInterface::bufferize`).
- Practical rule:
  - Prefer `catalyst.grad` for batched loss in qjit loops.
  - Do not use `catalyst.value_and_grad` on that batched quantum loss path in this environment/version.

## What was changed in repo

### `scripts/core/compiled_core.py`
- Added GPU dtype/runtime hints when device is `lightning.gpu`:
  - `c_dtype=np.complex64`
  - optional `use_async` via `EDGE_LIGHTNING_ASYNC`.
- Kept QNode compiled with `qjit(..., autograph=False)`.
- Added configurable batch method for experimentation:
  - `EDGE_COMPILED_BATCH_MODE=vmap|map` (`jax.vmap` vs `jax.lax.map`).
- Loss updated to stable BCE-with-logits:
  - `softplus(logit) - y*logit`.
- Params are `(weights, bias, alpha)` and optimizer is `optax.adam`.
- Added epoch-level JAX scan helper (`train_epoch_compiled`) with on-device permutation and donated state.
- Training kernel now uses batch-level AD via `catalyst.grad(_batch_loss_map, ...)` per step (no per-sample gradient accumulation loop).
- Avoids the `value_and_grad` crash path while reducing gradient-call overhead.
- NOTE: this is still a JAX-compiled classical loop calling qjit QNode, not full Catalyst-native optimization loop yet.

### `scripts/core/builders.py`
- Deterministic preprocessing/labels/sampling improvements:
  - deterministic binary label coercion, deterministic CSV row sampling seed handling.
- Batch size now respects config.
- Real validation split for threshold selection (`val_size`).
- Threshold persisted and used consistently on load/eval.
- Preprocessing/coercion states persisted and re-applied at inference.
- Logging path simplified:
  - removed expensive per-5-iter duplicate batch forward.
  - moved to epoch-level reporting.
  - added `losses.block_until_ready()` for honest timing when async dispatch exists.
- Device creation now passes `c_dtype=np.complex64` for `lightning.gpu`.

### Eval/CLI/supporting files
- `scripts/eval_models_edge_pls8.py`:
  - deterministic label coercion.
  - random seeded subset for `--limit` (no `head` bias).
  - thresholded prediction from `decision_function`.
- `scripts/train_edge_pls8_binary.py`:
  - `--batch` actually passed through.
- `scripts/specs.py`:
  - measurement mismatch fixed to include `"z0"`.
- `scripts/export_pls_full.py`:
  - deterministic label coercion + leakage warning.

## Validation notes
- Syntax checks passed with:
  - `.venv/bin/python -m py_compile ...`
- Smoke train run passed after reverting `jax_enable_x64=False`:
  - `EDGE_WANDB_LIVE=0 QML_DEVICE=lightning.qubit .venv/bin/python scripts/train_edge_pls8_binary.py --sample 300 --epochs 1 --batch 64 --lr 0.01 --seed 123 --out models/_smoke_perf_patch.pt`
- Eval run on raw CSV still showed NaN issues through PLS transform in one smoke path, indicating remaining data-cleaning/coercion edge cases to tighten.

## Known open risks
- Main throughput bottleneck likely remains:
  - JAX loop + qjit callback/kernel boundary per sample/batch element.
- `jax.vmap` vs `lax.map` may differ by hardware/backend; benchmark required per target GPU.
- The fully Catalyst-native train loop (`@qjit` + `catalyst.for_loop` + `catalyst.grad`) has not yet been implemented.

## Recommended next step for next LLM
1. Build a minimal fully-Catalyst training kernel:
- `@qjit(autograph=False)`
- fixed-shape batch and fixed `num_steps`
- inner loop with `catalyst.for_loop`
- per-sample circuit calls inside Catalyst loop (no JAX `vmap`)
- gradients via `catalyst.grad`
- simple optimizer first (SGD/momentum), then consider optax compatibility.

2. Benchmark three modes with `block_until_ready`:
- current `vmap(qjit(qnode))`
- current `lax.map(qjit(qnode))`
- fully Catalyst-native train loop

3. Keep `Rot` decomposition hardcoded in compiled path until upstream adjoint support is confirmed for target device/version.

4. Preserve current deployability fixes (preprocessing + coercion + threshold persistence) regardless of training backend refactor.

## Note on your x64 caveat (MLIR i32→i64 errors)

Plausible, and consistent with the kind of type strictness you hit in MLIR pipelines: turning `jax_enable_x64=False` can change default integer promotion/casting behavior in ways that expose `i32`/`i64` mismatches during compilation. The robust mitigation is to **make index dtypes explicit** (e.g., force permutation/indices to the expected width once) and keep all “count/shape” values in one consistent integer dtype end-to-end inside the compiled region.
