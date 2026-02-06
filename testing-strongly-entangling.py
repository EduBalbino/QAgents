import os
import jax
from jax import numpy as jnp
import pennylane as qml
from catalyst import grad, qjit

DEVICE = os.environ.get("QML_DEVICE", "lightning.qubit")
NUM_QUBITS = 4
LAYERS = 2
BATCH = 256


def _device():
    return qml.device(DEVICE, wires=NUM_QUBITS)


def _init_weights(key):
    # StronglyEntanglingLayers expects (L, M, 3)
    return jax.random.normal(key, (LAYERS, NUM_QUBITS, 3), dtype=jnp.float32)

def build_qnode():
    dev = _device()

    @qml.qnode(dev, interface="jax")
    def qnode_forward(weights, x):
        qml.AngleEmbedding(x, wires=range(NUM_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
        return qml.expval(qml.PauliZ(0))

    return qnode_forward


def run_forward():
    qnode_forward = build_qnode()
    forward_compiled = qjit(qnode_forward)
    key = jax.random.PRNGKey(0)
    weights = _init_weights(key)
    x = jax.random.normal(key, (NUM_QUBITS,), dtype=jnp.float32)
    return forward_compiled(weights, x)


def run_grad_single():
    qnode_forward = build_qnode()

    def loss_fn(weights, x, y):
        pred = qnode_forward(weights, x)
        return (pred - y) * (pred - y)

    train_step = qjit(lambda W, x, y: grad(loss_fn)(W, x, y))
    key = jax.random.PRNGKey(1)
    weights = _init_weights(key)
    x = jax.random.normal(key, (NUM_QUBITS,), dtype=jnp.float32)
    y = jnp.array(0.25, dtype=jnp.float32)
    return train_step(weights, x, y)


def run_grad_batch():
    qnode_forward = build_qnode()
    qnode_compiled = qjit(qnode_forward)

    def loss_fn(weights, X, y):
        preds = jax.vmap(qnode_compiled, in_axes=(None, 0), out_axes=0)(weights, X)
        diff = preds - y
        return jnp.mean(diff * diff)

    grad_fn = jax.grad(loss_fn)

    key = jax.random.PRNGKey(2)
    weights = _init_weights(key)
    X = jax.random.normal(key, (BATCH, NUM_QUBITS), dtype=jnp.float32)
    y = jax.random.normal(key, (BATCH,), dtype=jnp.float32)

    return grad_fn(weights, X, y)


def run_batch_no_grad():
    qnode_forward = build_qnode()
    qnode_compiled = qjit(qnode_forward)

    def loss_fn(weights, X, y):
        preds = jax.vmap(qnode_compiled, in_axes=(None, 0), out_axes=0)(weights, X)
        diff = preds - y
        return jnp.mean(diff * diff)

    key = jax.random.PRNGKey(3)
    weights = _init_weights(key)
    X = jax.random.normal(key, (BATCH, NUM_QUBITS), dtype=jnp.float32)
    y = jax.random.normal(key, (BATCH,), dtype=jnp.float32)
    return loss_fn(weights, X, y)


def run_batch_vmap_grad_batch1():
    qnode_forward = build_qnode()
    qnode_compiled = qjit(qnode_forward)

    def loss_fn(weights, X, y):
        preds = jax.vmap(qnode_compiled, in_axes=(None, 0), out_axes=0)(weights, X)
        diff = preds - y
        return jnp.mean(diff * diff)

    grad_fn = jax.grad(loss_fn)

    key = jax.random.PRNGKey(4)
    weights = _init_weights(key)
    X = jax.random.normal(key, (1, NUM_QUBITS), dtype=jnp.float32)
    y = jax.random.normal(key, (1,), dtype=jnp.float32)
    return grad_fn(weights, X, y)


if __name__ == "__main__":
    print(f"device={DEVICE}")

    def _print_small_error(name, exc):
        msg = str(exc).splitlines()
        hit = ""
        for line in msg:
            if "BufferizationStage" in line or "BufferHoistingPass" in line:
                hit = line.strip()
                break
        if not hit:
            hit = msg[0] if msg else repr(exc)
        print(f"{name} failed: {hit}")

    def _run_case(name, fn):
        try:
            print(f"{name}:", fn())
        except Exception as exc:
            _print_small_error(name, exc)

    _run_case("forward", run_forward)
    _run_case("grad_single", run_grad_single)
    _run_case("batch_no_grad", run_batch_no_grad)
    _run_case("batch_vmap_grad_batch1", run_batch_vmap_grad_batch1)
    _run_case("grad_batch", run_grad_batch)
