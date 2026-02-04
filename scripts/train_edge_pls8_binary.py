import argparse
import datetime as _dt
import os
import sys

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from scripts.core.jax_preprocess import load_edge_dataset
from scripts.core.jax_train import train_qjit
from scripts.core.builders import _save_model_torch


EDGE_DATASET = "data/ML-EdgeIIoT-dataset-binario.csv"
EDGE_LABEL = "Attack_label"
EDGE_FEATURES = [
    "ip.src_host",
    "ip.dst_host",
    "arp.dst.proto_ipv4",
    "arp.opcode",
    "arp.hw.size",
    "arp.src.proto_ipv4",
    "icmp.checksum",
    "icmp.seq_le",
    "icmp.transmit_timestamp",
    "icmp.unused",
    "http.file_data",
    "http.content_length",
    "http.request.uri.query",
    "http.request.method",
    "http.referer",
    "http.request.full_uri",
    "http.request.version",
    "http.response",
    "http.tls_port",
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.checksum",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.dstport",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",
    "tcp.options",
    "tcp.payload",
    "tcp.seq",
    "tcp.srcport",
    "udp.port",
    "udp.stream",
    "udp.time_delta",
    "dns.qry.name",
    "dns.qry.name.len",
    "dns.qry.qu",
    "dns.qry.type",
    "dns.retransmission",
    "dns.retransmit_request",
    "dns.retransmit_request_in",
    "mqtt.conack.flags",
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msg_decoded_as",
    "mqtt.msg",
    "mqtt.msgtype",
    "mqtt.proto_len",
    "mqtt.protoname",
    "mqtt.topic",
    "mqtt.topic_len",
    "mqtt.ver",
    "mbtcp.len",
    "mbtcp.trans_id",
    "mbtcp.unit_id",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an 8-qubit PLS model on ML-EdgeIIoT binario (4 layers)."
    )
    p.add_argument("--sample", type=int, default=60000, help="Row sample size")
    p.add_argument("--epochs", type=int, default=4, help="Training epochs")
    p.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    p.add_argument("--batch", type=int, default=512, help="Batch size")
    p.add_argument("--seed", type=int, default=1337, help="Random seed")
    p.add_argument(
        "--out",
        default="",
        help="Optional output model path (default: models/edgeiiot_bin_pls8_L4_<ts>.pt)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or f"models/edgeiiot_bin_pls8_L4_{ts}.pt"
    device_name = os.environ.get("QML_DEVICE", "lightning.gpu")

    prep = load_edge_dataset(
        path=EDGE_DATASET,
        sample_size=args.sample,
        features=EDGE_FEATURES,
        label=EDGE_LABEL,
        seed=args.seed,
        test_size=0.2,
        stratify=True,
        quantile_n=1000,
        quantile_output="uniform",
        pls_components=8,
    )

    batch_size = min(256, len(prep.X_train))
    num_it = max(1, len(prep.X_train) // batch_size)

    result = train_qjit(
        X_train=prep.X_train,
        y_train01=prep.y_train01,
        n_qubits=prep.feature_dim,
        layers=4,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=batch_size,
        seed=args.seed,
        device_name=device_name,
    )

    total_iters = int(args.epochs * num_it)
    print(
        f"Training finished in {result.train_time_s:.2f}s over {args.epochs} epoch(s), {total_iters} iters."
    )

    import jax
    import numpy as _np

    weights_np = _np.asarray(jax.device_get(result.params[0]))
    bias_np = _np.asarray(jax.device_get(result.params[1]))
    alpha_np = _np.asarray(jax.device_get(result.params[2]))

    _save_model_torch(
        path=out_path,
        created_at=ts,
        dataset=os.path.basename(EDGE_DATASET).replace(".csv", ""),
        device_name=device_name,
        num_qubits=prep.feature_dim,
        encoder_name="angle_embedding_y",
        encoder_opts={"hadamard": True, "angle_range": "0_pi", "reupload": True},
        ansatz_name="strongly_entangling",
        layers=4,
        measurement={"name": "z0", "wires": [0]},
        features=EDGE_FEATURES,
        label=EDGE_LABEL,
        scaler=prep.scaler,
        quantile=prep.quantile,
        pls=prep.pls,
        pca=None,
        weights=weights_np,
        bias=bias_np,
        alpha=alpha_np,
        train_cfg={"lr": args.lr, "batch": batch_size, "epochs": args.epochs},
        metrics={"last_loss": result.last_loss},
    )


if __name__ == "__main__":
    main()
