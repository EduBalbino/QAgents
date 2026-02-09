from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import os
import optax
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
    spec_hash: str,
    shape_key: tuple,
    encoder_name: str = "angle_embedding_y",
    ansatz_name: str = "strongly_entangling",
    measurement_name: str = "z0",
    measurement_wires: tuple[int, ...] = (0,),
    hadamard: bool = False,
    reupload: bool = False,
    num_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Callable]:
    if num_batches is None:
        num_batches = 1
    if batch_size is None:
        batch_size = int(shape_key[0]) if len(shape_key) > 0 else 1
    cache_key = (
        num_qubits,
        num_layers,
        backend.device_name,
        str(backend.dtype),
        tuple(sorted(backend.compile_opts.items())),
        spec_hash,
        shape_key,
        encoder_name,
        ansatz_name,
        measurement_name,
        tuple(measurement_wires),
        bool(hadamard),
        bool(reupload),
        num_batches,
        batch_size,
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
        num_batches=num_batches,
        batch_size=batch_size,
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
    num_batches: int,
    batch_size: int,
) -> Dict[str, Callable]:
    dev_kwargs: Dict[str, Any] = {}
    if backend.device_name.startswith("lightning."):
        # lightning simulators default to complex128; complex64 is usually faster for training.
        c_dtype_env = os.environ.get("EDGE_LIGHTNING_C_DTYPE", "complex64").strip().lower()
        dev_kwargs["c_dtype"] = np.complex128 if c_dtype_env == "complex128" else np.complex64
    if backend.device_name == "lightning.gpu":
        dev_kwargs["use_async"] = os.environ.get("EDGE_LIGHTNING_ASYNC", "1") != "0"
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

    @qml.qnode(dev, interface="jax", diff_method="adjoint")
    def qnode_forward(weights, x):
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
        raise ValueError(f"Unsupported measurement for compiled core: {measurement_name}")

    qnode_compiled = qjit(qnode_forward, **backend.compile_opts)

    def _batch_logits(weights, bias, alpha, xb):
        def _scan_body(_, x):
            logit = jnp.asarray(alpha * qnode_forward(weights, x) + bias, dtype=backend.dtype)
            return None, logit

        _, logits = jax.lax.scan(_scan_body, None, xb)
        return logits

    def batched_forward(weights, X_batch):
        def _scan_body(_, x):
            return None, qnode_compiled(weights, x)

        _, preds = jax.lax.scan(_scan_body, None, X_batch)
        return preds

    def bce_with_logits(logits, targets01, sample_weights):
        logits = jnp.asarray(logits, dtype=backend.dtype)
        y = jnp.asarray(targets01, dtype=backend.dtype)
        sample_weights = jnp.asarray(sample_weights, dtype=backend.dtype)
        # Numerically stable BCE with logits: softplus(logit) - y*logit
        loss = jax.nn.softplus(logits) - y * logits
        return jnp.mean(sample_weights * loss)

    adam_tx = optax.scale_by_adam()

    def init_opt_state(params):
        return adam_tx.init(params)

    def _batch_loss_map(weights, bias, alpha, xb, yb, wb):
        xb = jnp.asarray(xb, dtype=backend.dtype)
        yb = jnp.asarray(yb, dtype=backend.dtype)
        wb = jnp.asarray(wb, dtype=backend.dtype)
        logits = _batch_logits(weights, bias, alpha, xb)
        return bce_with_logits(logits, yb, wb)

    _batch_grad = catalyst.grad(_batch_loss_map, argnums=(0, 1, 2))

    @qjit(**backend.compile_opts)
    def train_epoch_compiled(train_state, key, X_steps, y01_steps, w_steps, lr_t):
        params, opt_state = train_state
        weights, bias, alpha = params
        lr_t = jnp.asarray(lr_t, dtype=backend.dtype)

        @catalyst.for_loop(0, num_batches, 1)
        def _batch_loop(i, carry):
            cur_weights, cur_bias, cur_alpha, cur_opt_state = carry
            Xb = X_steps[i]
            yb = y01_steps[i]
            wb = w_steps[i]
            grad_w, grad_b, grad_a = _batch_grad(cur_weights, cur_bias, cur_alpha, Xb, yb, wb)
            grads = (grad_w, grad_b, grad_a)
            params_now = (cur_weights, cur_bias, cur_alpha)
            updates, new_opt_state = adam_tx.update(grads, cur_opt_state, params_now)
            updates = jax.tree_util.tree_map(lambda u: -lr_t * u, updates)
            new_weights, new_bias, new_alpha = optax.apply_updates(params_now, updates)
            return (new_weights, new_bias, new_alpha, new_opt_state)

        weights, bias, alpha, opt_state = _batch_loop(
            (
                weights,
                bias,
                alpha,
                opt_state,
            )
        )
        final_i = num_batches - 1
        final_loss = _batch_loss_map(
            weights, bias, alpha,
            X_steps[final_i], y01_steps[final_i], w_steps[final_i],
        )
        loss_stats = jnp.asarray([final_loss, final_loss], dtype=backend.dtype)
        return ((weights, bias, alpha), opt_state), key, loss_stats

    def assert_no_python_callback_ir(
        train_state,
        key,
        X_steps,
        y01_steps,
        w_steps,
        lr_t,
    ) -> None:
        """Fail fast if compiled IR still contains Python callback boundaries."""
        _ = train_epoch_compiled(train_state, key, X_steps, y01_steps, w_steps, lr_t)
        mlir_txt = str(getattr(train_epoch_compiled, "mlir", ""))
        if "xla_ffi_python" in mlir_txt or "CpuCallback" in mlir_txt:
            raise RuntimeError("Compiled train epoch still contains Python callback boundary.")

    return {
        "batched_forward": batched_forward,
        "init_opt_state": init_opt_state,
        "train_epoch_compiled": train_epoch_compiled,
        "assert_no_python_callback_ir": assert_no_python_callback_ir,
    }
