import argparse
import datetime as _dt
import os
import sys

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from scripts.core import builders


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
    p.add_argument("--batch", type=int, default=256, help="Mini-batch size")
    p.add_argument("--seed", type=int, default=1337, help="Random seed")
    p.add_argument(
        "--out",
        default="",
        help="Optional output model path (default: models/edgeiiot_bin_pls8_L4_<ts>.pt)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    recipe = builders.Recipe() | builders.csv(EDGE_DATASET, sample_size=args.sample)
    recipe = (
        recipe
        | builders.select(EDGE_FEATURES, EDGE_LABEL)
        | builders.quantile_uniform(n_quantiles=1000, output_distribution="uniform")
        | builders.pls_to_pow2(components=8)
        | builders.device(name=os.environ.get("QML_DEVICE", "lightning.qubit"))
        | builders.encoder("angle_embedding_y")
        | builders.ansatz("strongly_entangling", layers=4)
        | builders.train(lr=args.lr, batch=args.batch, epochs=args.epochs, seed=args.seed, test_size=0.2, stratify=True)
    )

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or f"models/edgeiiot_bin_pls8_L4_{ts}.pt"
    recipe = recipe | builders.save(out_path)

    builders.run(recipe)


if __name__ == "__main__":
    main()
