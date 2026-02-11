from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import catalyst
from catalyst import qjit


@dataclass(frozen=True)
class Backend:
    device_name: str
    dtype: Any
    compile_opts: Dict[str, Any]


_CORE_CACHE: Dict[tuple, Dict[str, Callable]] = {}


def get_compiled_core(
    num_qubits: int,
    num_layers: int,
    backend: Backend,
    *,
    batch_size: int,
    feature_dim: int,
    encoder_name: str = "angle_embedding_y",
    ansatz_name: str = "strongly_entangling",
    measurement_name: str = "z0",
    measurement_wires: tuple[int, ...] = (0,),
    hadamard: bool = False,
    reupload: bool = False,
    focal_gamma: float = 0.0,
    alpha_mode: str = "softplus",
) -> Dict[str, Callable]:
    cache_key = (
        num_qubits,
        num_layers,
        int(batch_size),
        int(feature_dim),
        backend.device_name,
        str(backend.dtype),
        tuple(sorted(backend.compile_opts.items())),
        encoder_name,
        ansatz_name,
        measurement_name,
        tuple(measurement_wires),
        bool(hadamard),
        bool(reupload),
        float(focal_gamma),
        str(alpha_mode),
    )
    cached = _CORE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    compiled = build_compiled_core(
        num_qubits,
        num_layers,
        backend,
        encoder_name=encoder_name,
        ansatz_name=ansatz_name,
        measurement_name=measurement_name,
        measurement_wires=measurement_wires,
        hadamard=hadamard,
        reupload=reupload,
        focal_gamma=focal_gamma,
        alpha_mode=alpha_mode,
    )
    _CORE_CACHE[cache_key] = compiled
    return compiled


def build_compiled_core(
    num_qubits: int,
    num_layers: int,
    backend: Backend,
    encoder_name: str,
    ansatz_name: str,
    measurement_name: str,
    measurement_wires: tuple[int, ...],
    hadamard: bool,
    reupload: bool,
    focal_gamma: float,
    alpha_mode: str,
) -> Dict[str, Callable]:
    dev_kwargs: Dict[str, Any] = {}
    if backend.device_name.startswith("lightning."):
        # Use complex64 deterministically; do not depend on env var overrides.
        dev_kwargs["c_dtype"] = np.complex64
    if backend.device_name == "lightning.gpu":
        # Deterministic default: async execution enabled.
        dev_kwargs["use_async"] = True
    dev = qml.device(backend.device_name, wires=num_qubits, **dev_kwargs)

    def _rot_as_rz_ry_rz(phi, theta, omega, wire: int) -> None:
        # qml.Rot(phi, theta, omega) decomposition in circuit order.
        qml.RZ(phi, wires=wire)
        qml.RY(theta, wires=wire)
        qml.RZ(omega, wires=wire)

    all_wires = tuple(range(num_qubits))
    meas_ws = tuple(int(w) for w in measurement_wires) if measurement_wires else (0,)
    meas_ws = tuple(w for w in meas_ws if 0 <= w < num_qubits) or all_wires

    if num_qubits > 1:
        se_ranges = tuple((l % (num_qubits - 1)) + 1 for l in range(num_layers))
    else:
        se_ranges = (0,) * num_layers

    def _apply_encoder(x) -> None:
        if hadamard and encoder_name != "amplitude_embedding":
            for w in all_wires:
                qml.Hadamard(wires=w)
        if encoder_name == "angle_embedding_y":
            qml.AngleEmbedding(x, wires=all_wires, rotation="Y")
        elif encoder_name == "angle_pair_xy":
            for i, w in enumerate(all_wires):
                qml.RX(x[i], wires=w)
                qml.RY(x[i], wires=w)
        elif encoder_name == "amplitude_embedding":
            qml.AmplitudeEmbedding(x, wires=all_wires, normalize=True)
        else:
            raise ValueError(f"Unsupported encoder for compiled core: {encoder_name}")

    def _apply_ansatz_layer(weights, l: int) -> None:
        if ansatz_name == "strongly_entangling":
            for i, w in enumerate(all_wires):
                _rot_as_rz_ry_rz(
                    weights[..., l, i, 0],
                    weights[..., l, i, 1],
                    weights[..., l, i, 2],
                    w,
                )
            if num_qubits > 1:
                r = se_ranges[l]
                for i, w in enumerate(all_wires):
                    qml.CNOT(wires=[w, all_wires[(i + r) % num_qubits]])
            return

        if ansatz_name == "ring_rot_cnot":
            for i, w in enumerate(all_wires):
                _rot_as_rz_ry_rz(
                    weights[..., l, i, 0],
                    weights[..., l, i, 1],
                    weights[..., l, i, 2],
                    w,
                )
            if num_qubits > 1:
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[all_wires[i], all_wires[i + 1]])
                qml.CNOT(wires=[all_wires[-1], all_wires[0]])
            return

        raise ValueError(f"Unsupported ansatz for compiled core: {ansatz_name}")

    def _apply_readout_layer_rot(w_ro) -> None:
        # Trainable post-processing layer using gate parameters (supported by adjoint).
        #
        # Per-wire Rot makes the measured axis an arbitrary Bloch direction, i.e.
        # U^\dagger Z U = a X + b Y + c Z.
        for i, w in enumerate(meas_ws):
            j = 3 * i
            # Avoid qml.Rot here; some Lightning adjoint paths (esp under Catalyst)
            # have incomplete support for it. Use the explicit RZ/RY/RZ decomposition
            # already used by the ansatz.
            _rot_as_rz_ry_rz(w_ro[j + 0], w_ro[j + 1], w_ro[j + 2], w)

    @qml.qnode(dev, interface="jax", diff_method="adjoint")
    def qnode_forward(weights, x, w_ro):
        if reupload:
            for l in range(num_layers):
                _apply_encoder(x)
                _apply_ansatz_layer(weights, l)
        else:
            _apply_encoder(x)
            for l in range(num_layers):
                _apply_ansatz_layer(weights, l)

        if measurement_name == "z0":
            return qml.expval(qml.PauliZ(0))
        if measurement_name == "mean_z":
            coeffs = [1.0 / float(len(meas_ws))] * len(meas_ws)
            observables = [qml.PauliZ(w) for w in meas_ws]
            return qml.expval(qml.Hamiltonian(coeffs, observables))
        if measurement_name == "mean_z_readout":
            _apply_readout_layer_rot(w_ro)
            coeffs = [1.0 / float(len(meas_ws))] * len(meas_ws)
            observables = [qml.PauliZ(w) for w in meas_ws]
            return qml.expval(qml.Hamiltonian(coeffs, observables))
        if measurement_name == "z_vec":
            # Project the Z-vector via Hamiltonian coefficients. This keeps the QNode output
            # scalar (important for Catalyst stability). Note: some Catalyst versions have
            # limited support for differentiating w.r.t. Hamiltonian coefficients, so callers
            # should not rely on `w_ro` being learnable; initialize it sensibly.
            coeffs = w_ro
            observables = [qml.PauliZ(w) for w in meas_ws]
            return qml.expval(qml.Hamiltonian(coeffs, observables))
        raise ValueError(f"Unsupported measurement for compiled core: {measurement_name}")

    qnode_compiled = qjit(qnode_forward, **backend.compile_opts)

    focal_gamma = float(focal_gamma or 0.0)
    alpha_mode = str(alpha_mode or "softplus").strip().lower()
    if alpha_mode not in ("direct", "softplus"):
        alpha_mode = "softplus"

    def _scan_expvals(weights, w_ro, xb):
        # NOTE: Catalyst does not implement a batching rule for `qinst`, so `jax.vmap`
        # over a qjit-compiled QNode fails with:
        #   NotImplementedError: Batching rule for 'qinst' not implemented
        # Use `lax.scan` instead (sequential over batch) to stay within supported transforms.
        def _step(carry, x):
            ev = qnode_compiled(weights, x, w_ro)
            ev = jnp.asarray(ev, dtype=backend.dtype)
            return carry, ev

        carry0 = jnp.asarray(0, dtype=backend.dtype)
        _, evs = jax.lax.scan(_step, carry0, xb)
        return evs

    def _batch_logits(weights, w_ro, bias, xb):
        xb = jnp.asarray(xb, dtype=backend.dtype)
        w_ro = jnp.asarray(w_ro, dtype=backend.dtype)
        evs = _scan_expvals(weights, w_ro, xb)  # (B,)
        bias = jnp.asarray(bias, dtype=backend.dtype)  # ()
        logits_raw = evs + bias  # (B,)
        return jnp.asarray(logits_raw, dtype=backend.dtype)

    def batched_forward(weights, w_ro, X):
        X = jnp.asarray(X, dtype=backend.dtype)
        w_ro = jnp.asarray(w_ro, dtype=backend.dtype)
        return _scan_expvals(weights, w_ro, X)

    @qjit(**backend.compile_opts)
    def batch_loss_and_grad(weights, w_ro, bias, alpha_raw, Xb, yb, wb):
        # Expect already-correct dtypes from the Python training loop.
        # Avoid dtype-casting inside qjit; it tends to introduce extra IR and has
        # triggered brittle MLIR lowering bugs in this repo's Catalyst versions.

        def _loss_fn(wq, wlin, b, a_raw):
            logits_raw = _batch_logits(wq, wlin, b, Xb)  # (proj_expval) + bias
            if alpha_mode == "direct":
                a = a_raw
            else:
                # Constrain alpha > 0 to avoid sign flips and gain blow-ups.
                a = jax.nn.softplus(a_raw) + jnp.asarray(1e-3, dtype=backend.dtype)
            logits = a * logits_raw  # match eval: alpha * (expval + bias)
            y = yb  # {0,1} float
            ce = jax.nn.softplus(logits) - y * logits  # stable BCE-with-logits

            # Optional focal factor to emphasize hard examples under class imbalance.
            # Keep gamma as a constant for compilation stability (passed via get_compiled_core()).
            if focal_gamma > 0.0:
                p = jax.nn.sigmoid(logits)
                pt = y * p + (1.0 - y) * (1.0 - p)
                focal = jnp.power(1.0 - pt, jnp.asarray(focal_gamma, dtype=backend.dtype))
                loss = focal * ce
            else:
                loss = ce

            return jnp.mean(wb * loss)

        # IMPORTANT: In this environment/version, `catalyst.value_and_grad` over a
        # batched quantum loss has been observed to hard-crash the compiler
        # (`memref.subview ... staticOffsets length mismatch`). Use `catalyst.grad`
        # and compute the loss value separately.
        loss = _loss_fn(weights, w_ro, bias, alpha_raw)
        gwq, gwlin, gb, ga = catalyst.grad(_loss_fn, argnums=(0, 1, 2, 3))(weights, w_ro, bias, alpha_raw)
        return loss, gwq, gwlin, gb, ga

    return {
        "batched_forward": batched_forward,
        "batch_loss_and_grad": batch_loss_and_grad,
        # For callers to size `w_ro`.
        "readout_dim": int(
            len(meas_ws) if measurement_name == "z_vec" else (3 * len(meas_ws) if measurement_name == "mean_z_readout" else 1)
        ),
    }
