from __future__ import annotations

import argparse
import os

import catalyst
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml
from catalyst import qjit


def make_device(name: str, n_qubits: int):
    kwargs = {}
    if name == "lightning.gpu":
        kwargs["c_dtype"] = np.complex64
    return qml.device(name, wires=n_qubits, **kwargs)


def build_qnode(dev, n_qubits: int, n_layers: int):
    def _rot_as_rz_ry_rz(phi, theta, omega, wire: int) -> None:
        qml.RZ(phi, wires=wire)
        qml.RY(theta, wires=wire)
        qml.RZ(omega, wires=wire)

    def _strongly_entangling_no_rot(weights) -> None:
        if n_qubits > 1:
            ranges = tuple((l % (n_qubits - 1)) + 1 for l in range(n_layers))
        else:
            ranges = (0,) * n_layers

        wires = list(range(n_qubits))
        for l in range(n_layers):
            for i in range(n_qubits):
                _rot_as_rz_ry_rz(
                    weights[l, i, 0],
                    weights[l, i, 1],
                    weights[l, i, 2],
                    wires[i],
                )
            if n_qubits > 1:
                r = ranges[l]
                for i in range(n_qubits):
                    qml.CNOT(wires=[wires[i], wires[(i + r) % n_qubits]])

    @qml.qnode(dev, interface="jax", diff_method="adjoint")
    def qnode_forward(weights, x):
        qml.AngleEmbedding(jnp.pi * x, wires=range(n_qubits), rotation="Y")
        _strongly_entangling_no_rot(weights)
        return qml.expval(qml.PauliZ(0))

    return qnode_forward


def bce_with_logits(logits, targets01, weights):
    logits = jnp.asarray(logits, dtype=jnp.float32)
    targets01 = jnp.asarray(targets01, dtype=jnp.float32)
    weights = jnp.asarray(weights, dtype=jnp.float32)
    loss = jax.nn.softplus(logits) - targets01 * logits
    return jnp.asarray(jnp.mean(weights * loss), dtype=jnp.float32)


def build_data(steps: int, batch: int, n_features: int):
    n = steps * batch
    rng = np.random.default_rng(7)
    x = rng.random((n, n_features), dtype=np.float32)
    y = rng.integers(0, 2, size=(n,), dtype=np.int32).astype(np.float32)
    w = np.ones((n,), dtype=np.float32)
    x_steps = x.reshape(steps, batch, n_features)
    y_steps = y.reshape(steps, batch)
    w_steps = w.reshape(steps, batch)
    return (
        jnp.asarray(x, dtype=jnp.float32),
        jnp.asarray(y, dtype=jnp.float32),
        jnp.asarray(w, dtype=jnp.float32),
        jnp.asarray(x_steps, dtype=jnp.float32),
        jnp.asarray(y_steps, dtype=jnp.float32),
        jnp.asarray(w_steps, dtype=jnp.float32),
    )


def run_case(case: str, device_name: str, steps: int, batch: int, qubits: int, layers: int):
    # Reproducer stability: Catalyst control-flow lowering can hit i32/i64 index
    # mismatches when x64 is disabled in some builds.
    jax.config.update("jax_enable_x64", True)

    dev = make_device(device_name, qubits)
    qnode_forward = build_qnode(dev, n_qubits=qubits, n_layers=layers)

    x_flat, y_flat, w_flat, x_steps, y_steps, w_steps = build_data(
        steps=steps, batch=batch, n_features=qubits
    )

    inv_b = jnp.asarray(1.0 / float(batch), dtype=jnp.float32)
    w0 = jnp.zeros((layers, qubits, 3), dtype=jnp.float32)
    b0 = jnp.asarray(0.0, dtype=jnp.float32)
    a0 = jnp.asarray(1.0, dtype=jnp.float32)

    def sample_loss(weights, bias, alpha, x_i, y_i, w_i):
        logit_i = jnp.asarray(alpha * qnode_forward(weights, x_i) + bias, dtype=jnp.float32)
        y_i = jnp.asarray(y_i, dtype=jnp.float32)
        w_i = jnp.asarray(w_i, dtype=jnp.float32)
        return jnp.asarray(w_i * (jax.nn.softplus(logit_i) - y_i * logit_i), dtype=jnp.float32)

    sample_grad = catalyst.grad(sample_loss, argnums=(0, 1, 2))

    def batch_loss_map(weights, bias, alpha, xb, yb, wb):
        logits = jax.lax.map(
            lambda x: jnp.asarray(alpha * qnode_forward(weights, x) + bias, dtype=jnp.float32), xb
        )
        return bce_with_logits(logits, yb, wb)

    batch_grad_map = catalyst.grad(batch_loss_map, argnums=(0, 1, 2))
    batch_vg_map = catalyst.value_and_grad(batch_loss_map, argnums=(0, 1, 2))
    adam_tx = optax.adam(learning_rate=0.01)

    @qjit(autograph=False)
    def epoch_sample_grad(weights, bias, alpha, xsteps, ysteps, wsteps):
        @catalyst.for_loop(0, steps, 1)
        def _batch_loop(i, carry):
            cw, cb, ca, sum_loss = carry
            xb = xsteps[i]
            yb = ysteps[i]
            wb = wsteps[i]

            @catalyst.for_loop(0, batch, 1)
            def _sample_loop(j, sc):
                gW, gB, gA, lsum = sc
                li = sample_loss(cw, cb, ca, xb[j], yb[j], wb[j])
                dW, dB, dA = sample_grad(cw, cb, ca, xb[j], yb[j], wb[j])
                return (gW + dW, gB + dB, gA + dA, lsum + li)

            gW0 = jnp.zeros_like(cw)
            gB0 = jnp.asarray(0.0, dtype=jnp.float32)
            gA0 = jnp.asarray(0.0, dtype=jnp.float32)
            gW, gB, gA, lsum = _sample_loop((gW0, gB0, gA0, jnp.asarray(0.0, dtype=jnp.float32)))
            lr = jnp.asarray(0.01, dtype=jnp.float32)
            nw = cw - lr * gW * inv_b
            nb = cb - lr * gB * inv_b
            na = ca - lr * gA * inv_b
            return (nw, nb, na, sum_loss + lsum * inv_b)

        return _batch_loop((weights, bias, alpha, jnp.asarray(0.0, dtype=jnp.float32)))

    @qjit(autograph=False)
    def epoch_batch_grad_map(weights, bias, alpha, xsteps, ysteps, wsteps):
        @catalyst.for_loop(0, steps, 1)
        def _batch_loop(i, carry):
            cw, cb, ca, sum_loss = carry
            xb = xsteps[i]
            yb = ysteps[i]
            wb = wsteps[i]
            li = batch_loss_map(cw, cb, ca, xb, yb, wb)
            dW, dB, dA = batch_grad_map(cw, cb, ca, xb, yb, wb)
            lr = jnp.asarray(0.01, dtype=jnp.float32)
            nw = cw - lr * dW
            nb = cb - lr * dB
            na = ca - lr * dA
            return (nw, nb, na, sum_loss + li)

        return _batch_loop((weights, bias, alpha, jnp.asarray(0.0, dtype=jnp.float32)))

    @qjit(autograph=False)
    def epoch_batch_vg_map(weights, bias, alpha, xsteps, ysteps, wsteps):
        @catalyst.for_loop(0, steps, 1)
        def _batch_loop(i, carry):
            cw, cb, ca, sum_loss = carry
            xb = xsteps[i]
            yb = ysteps[i]
            wb = wsteps[i]
            li, (dW, dB, dA) = batch_vg_map(cw, cb, ca, xb, yb, wb)
            lr = jnp.asarray(0.01, dtype=jnp.float32)
            nw = cw - lr * dW
            nb = cb - lr * dB
            na = ca - lr * dA
            return (nw, nb, na, sum_loss + li)

        return _batch_loop((weights, bias, alpha, jnp.asarray(0.0, dtype=jnp.float32)))

    @qjit(autograph=False)
    def epoch_batch_grad_dynamic_slice(weights, bias, alpha, xtrain, ytrain, wtrain):
        @catalyst.for_loop(0, steps, 1)
        def _batch_loop(i, carry):
            cw, cb, ca, sum_loss = carry
            start = i * batch
            xb = jax.lax.dynamic_slice(xtrain, (start, 0), (batch, qubits))
            yb = jax.lax.dynamic_slice(ytrain, (start,), (batch,))
            wb = jax.lax.dynamic_slice(wtrain, (start,), (batch,))
            li = batch_loss_map(cw, cb, ca, xb, yb, wb)
            dW, dB, dA = batch_grad_map(cw, cb, ca, xb, yb, wb)
            lr = jnp.asarray(0.01, dtype=jnp.float32)
            nw = cw - lr * dW
            nb = cb - lr * dB
            na = ca - lr * dA
            return (nw, nb, na, sum_loss + li)

        return _batch_loop((weights, bias, alpha, jnp.asarray(0.0, dtype=jnp.float32)))

    @qjit(autograph=False)
    def epoch_batch_grad_map_optax(train_state, xsteps, ysteps, wsteps):
        @catalyst.for_loop(0, steps, 1)
        def _batch_loop(i, carry):
            params, opt_state, sum_loss = carry
            cw, cb, ca = params
            xb = xsteps[i]
            yb = ysteps[i]
            wb = wsteps[i]
            li = batch_loss_map(cw, cb, ca, xb, yb, wb)
            grads = batch_grad_map(cw, cb, ca, xb, yb, wb)
            updates, new_opt_state = adam_tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_opt_state, sum_loss + li)

        return _batch_loop((train_state[0], train_state[1], jnp.asarray(0.0, dtype=jnp.float32)))

    if case == "sample_grad":
        w1, b1, a1, lsum = epoch_sample_grad(w0, b0, a0, x_steps, y_steps, w_steps)
    elif case == "batch_grad_map":
        w1, b1, a1, lsum = epoch_batch_grad_map(w0, b0, a0, x_steps, y_steps, w_steps)
    elif case == "batch_vg_map":
        w1, b1, a1, lsum = epoch_batch_vg_map(w0, b0, a0, x_steps, y_steps, w_steps)
    elif case == "batch_grad_dynamic_slice":
        w1, b1, a1, lsum = epoch_batch_grad_dynamic_slice(w0, b0, a0, x_flat, y_flat, w_flat)
    elif case == "batch_grad_map_optax":
        train_state0 = ((w0, b0, a0), adam_tx.init((w0, b0, a0)))
        (w1, b1, a1), _, lsum = epoch_batch_grad_map_optax(train_state0, x_steps, y_steps, w_steps)
    else:
        raise ValueError(case)

    print(
        f"[ok] case={case} device={device_name} "
        f"loss_sum={float(np.asarray(lsum)):.6f} "
        f"|W|={float(np.linalg.norm(np.asarray(w1))):.6f} "
        f"b={float(np.asarray(b1)):.6f} a={float(np.asarray(a1)):.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "sample_grad",
            "batch_grad_map",
            "batch_vg_map",
            "batch_grad_dynamic_slice",
            "batch_grad_map_optax",
        ),
    )
    parser.add_argument("--device", default=os.environ.get("QML_DEVICE", "lightning.gpu"))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()

    run_case(
        case=args.case,
        device_name=args.device,
        steps=int(args.steps),
        batch=int(args.batch),
        qubits=int(args.qubits),
        layers=int(args.layers),
    )


if __name__ == "__main__":
    main()
