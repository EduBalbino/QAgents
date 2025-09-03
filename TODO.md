# TODO: Align implementation with the article (Angle vs Amplitude Encoding)

This plan lists concrete, implementable divergences between the paper in `article/main.tex` and the current code in `scripts/core/` and `scripts/ab_test.py`. Each item has steps and acceptance criteria.

References (by file/section):
- Article
  - Angle variants table: `article/main.tex` lines 531–568
  - Re-uploading: lines 484–491
  - Measurement (binary vs multi-class + softmax): lines 494–502
  - Preprocessing: lines 521–529, 770–777
  - Strongly Entangling layers: lines 746–766
  - Training protocol: lines 770–783, 775–777
- Codebase
  - Encoders: `scripts/core/builders.py` lines 108–143
  - Ansatz: `scripts/core/builders.py` lines 145–162
  - Runner: `scripts/core/builders.py` `run()` lines 188–481
  - A/B harness: `scripts/ab_test.py`

---

## Goals
- Enable the full Angle-encoding search space from the article (not just RX/RY/RZ single-axis).
- Make AmplitudeEmbedding correct and fair (PCA to power-of-two + qubit count = log2).
- Use Strongly Entangling layers and the article’s layer schedule {2,4,6,8,10}.
- Align preprocessing to encoding type (Angle in [0,π], Amplitude relies on L2 norm).
- Support multi-class measurement with softmax and the article’s training protocol.

---

## 1) Full Angle-encoding search space (20 circuit variants)

Current gap:
- Only `qml.AngleEmbedding` with single-axis X/Y/Z (+ optional global Hadamard). No multi-rotation sequences.

Plan:
- Implement the 20 Angle variants from the table (RX, RY, RX–RY, RX–RZ, …, H–RY–RX–RZ, etc.), deduping equivalents.
- Register encoders with descriptive names (e.g., `angle_combo_rx`, `angle_combo_rx_ry`, `angle_combo_h_ry_rx_rz`).
- Update `scripts/ab_test.py` to enumerate these encoders for A/B runs.

Steps:
- In `scripts/core/builders.py`:
  - Add a helper `def _apply_angle_combo(x, wires, gates: List[str])` to apply per-feature rotations in order.
  - Create and `@register_encoder(...)` the 20 variants, internally calling the helper; prepend H where needed.
- In `scripts/ab_test.py`:
  - Extend `encoders` list to include representative subset/all variants from the article’s Table (lines 549–568).

Acceptance criteria:
- Running `ab_test.py` over the new encoder names completes without exceptions.
- Logs indicate distinct encoder names and results for each variant.

---

## 2) Correct AmplitudeEmbedding wiring + fair PCA alignment

Current gap:
- Device wires = number of features, but `qml.AmplitudeEmbedding` expects length = 2**n_qubits and device wires = n_qubits.
- Optional PCA step exists (`dataset.pca_pow2`) but A/B harness doesn’t use it; fairness rules from the article aren’t applied.

Plan:
- Auto-adjust for Amplitude:
  - If D (features) is not a power-of-two, PCA to D' = 2**k ≤ D.
  - Set `num_qubits = int(log2(D'))` for the device (not D').
- Support fairness for Angle models: optional flag to reduce features to `k = int(log2(D_original))` via PCA so Angle vs Amplitude use the same qubits.

Steps (in `builders.run()`):
- Detect `enc_name == "amplitude_embedding"`:
  - Apply PCA to nearest power-of-two ≤ D if needed.
  - Compute `num_qubits = int(log2(D'))` and set device wires accordingly.
- For Angle encoders, honor a config flag (e.g., `enc_cfg["fair_qubits"]`) to run PCA to `k = int(log2(D_orig))`.
- Keep existing `pca_to_pow2(...)` step but allow automatic handling per encoder when not explicitly requested.

Acceptance criteria:
- Amplitude runs do not crash on non-power-of-two feature counts and use `log2` wires.
- With `fair_qubits=True` for Angle encoders, Amplitude and Angle runs report the same `num_qubits` in logs.

---

## 3) Strongly Entangling layers + article layer schedule

Current gap:
- `scripts/ab_test.py` uses `ring_rot_cnot` and layers `[2,3]`.

Plan:
- Switch harness to `ansatz("strongly_entangling", layers=L)` with `layers_list = [2,4,6,8,10]`.
- Optionally keep `ring_rot_cnot` as a baseline toggle.

Steps:
- In `scripts/ab_test.py`:
  - Change `ansatz("ring_rot_cnot", ...)` to `ansatz("strongly_entangling", ...)`.
  - Set `layers_list = [2,4,6,8,10]`.

Acceptance criteria:
- A/B runs complete across the expanded layer grid and record metrics per layer.

---

## 4) Preprocessing aligned to encoding

Current gap:
- Always `MinMaxScaler(0,1)`; Angle scaling to [0,π] is optional; Amplitude also gets min-max then L2-normalized in the embedding (double scaling).

Plan:
- Angle encoders:
  - Default to angle range [0,π] (or configurable to [-π/2, π/2] for Bloch analyses) and pass `angle_scale` accordingly.
- Amplitude encoders:
  - Skip MinMax scaling entirely; rely on `normalize=True` of `AmplitudeEmbedding` (and optional standardization if needed).

Steps (in `builders.run()`):
- Branch scaling by encoder type:
  - If `enc_name.startswith("angle_")` or in angle combos: MinMax to [0,1] then scale to [0,π] in-circuit (existing `angle_range="0_pi"`), or center to [-π/2, π/2] when requested.
  - If `enc_name == "amplitude_embedding"`: pass raw (or standardized) features; do not MinMax.

Acceptance criteria:
- Angle runs show consistent angle scaling; Amplitude runs no longer apply MinMax.
- No runtime errors due to input ranges; accuracy is reproducible across runs.

---

## 5) Multi-class measurement + training protocol parity

Current gap:
- Only binary-style `expval(PauliZ)` or mean of a few Z’s; labels coerced to {-1,1}.
- No softmax or cross-entropy; device runs analytic (no shots); training uses lr=0.1, epochs=1, batch large; no repeated trials.

Plan:
- Measurement:
  - Add `measurement={"name":"multiclass","wires":[...]} -> qml.probs(wires=...)`.
  - Apply softmax and cross-entropy loss on one-hot labels.
- Training protocol:
  - Device: `default.qubit` with `shots=1000`.
  - Optimizer: Adam, lr=0.01, epochs=30, batch=10.
  - Repeats: run N=10 times and aggregate mean/std of metrics.

Steps:
- In `builders.run()`:
  - Extend QNode to return probabilities for multiclass.
  - Use one-hot labels and cross-entropy when in multiclass mode; preserve current path for binary.
  - Allow device config `device(name="default.qubit", shots=1000)`.
  - Add a `repeats` parameter that loops training/eval N times and aggregates results and std in the returned dict.
- In `scripts/ab_test.py`:
  - Pass device config, training params (lr, epochs, batch), and repeats; include std in printed table.

Acceptance criteria:
- Wine (multi-class) runs complete and report mean ± std for Accuracy (and optionally other metrics), using softmax outputs.
- Diabetes (binary) continues to work with the existing expval pathway.

---

## Execution order (recommended)
1) Implement item 2 (Amplitude + PCA fairness) to unblock correctness and comparability.
2) Implement item 3 (Strongly Entangling + layers) to match article’s ansatz/layer schedule.
3) Implement item 4 (Preprocessing) to align input pipelines by encoder.
4) Implement item 1 (Angle variants) to explore the full search space.
5) Implement item 5 (Multiclass + training parity) to support Wine and statistical reporting.

---

## Risks & Mitigations
- Amplitude dimension mismatch: mitigated by auto-PCA to 2**k and device wires = log2.
- Performance/runtime: expanded encoder/layer grid increases jobs; use `AB_NUM_SHARDS` and `AB_MAX_WORKERS` to shard/limit concurrency.
- API churn: keep defaults backward-compatible; guard new behavior behind explicit flags where needed (e.g., `fair_qubits`, `angle_range`).

---

## Quick harness usage (examples)
- Shard A/B jobs:
  - `AB_NUM_SHARDS=4 AB_SHARD_INDEX=0 python scripts/ab_test.py`
- Limit concurrency:
  - `AB_MAX_WORKERS=2 python scripts/ab_test.py`
- Smaller sample for smoke tests (if dataset is large):
  - `AB_SAMPLE=5000 python scripts/ab_test.py`
