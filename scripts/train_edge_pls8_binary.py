import argparse
import datetime as _dt
import os
import re
import subprocess
import sys
import time

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from scripts.core import builders


EDGE_BATCH_SIZE = 64
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
    p.add_argument("--seed", type=int, default=1337, help="Random seed")
    p.add_argument("--no-save", action="store_true", help="Skip writing model file.")
    p.add_argument(
        "--out",
        default="",
        help="Optional output model path (default: models/edgeiiot_bin_pls8_L4_<ts>.pt)",
    )
    p.add_argument("--sweep", action="store_true", help="Run a small speed sweep.")
    return p.parse_args()

_RE_COMPILE_TIME = re.compile(r"Compile\\(preflight\\) time:\\s*([0-9]+\\.[0-9]+)s")
_RE_CPU_FUSED_TIME = re.compile(r"CPU fused done .*\\| Time:\\s*([0-9]+\\.[0-9]+)s")
_RE_EPOCH_TIME = re.compile(r"Epoch\\s+\\d+/\\d+\\s+done .*\\| Epoch Time:\\s*([0-9]+\\.[0-9]+)s")


def _parse_times(stdout: str) -> dict:
    out: dict = {}
    m = _RE_COMPILE_TIME.search(stdout)
    if m:
        out["compile_s"] = float(m.group(1))
    m = _RE_CPU_FUSED_TIME.search(stdout)
    if m:
        out["train_s"] = float(m.group(1))
    else:
        ep = [float(x) for x in _RE_EPOCH_TIME.findall(stdout)]
        if ep:
            out["train_s"] = float(sum(ep))
    return out


def _run_one(*, sample: int, epochs: int, lr: float, seed: int) -> dict:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--sample",
        str(int(sample)),
        "--epochs",
        str(int(epochs)),
        "--lr",
        str(float(lr)),
        "--seed",
        str(int(seed)),
        "--no-save",
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.perf_counter() - t0
    stdout = p.stdout or ""
    parsed = _parse_times(stdout)
    return {
        "returncode": int(p.returncode),
        "wall_s": float(dt),
        **parsed,
        "tail": "\n".join(stdout.splitlines()[-40:]),
    }


def _run_sweep(args: argparse.Namespace) -> None:
    epochs = 4  # fixed for sweep
    r = _run_one(
        sample=int(args.sample),
        epochs=epochs,
        lr=float(args.lr),
        seed=int(args.seed),
    )
    status = "OK" if r["returncode"] == 0 else "FAIL"
    compile_s = r.get("compile_s")
    train_s = r.get("train_s")
    print(
        f"[{status}] impl=scan batch={EDGE_BATCH_SIZE} wall={r['wall_s']:.2f}s "
        f"compile={compile_s if compile_s is not None else 'NA'} "
        f"train={train_s if train_s is not None else 'NA'}"
    )
    if r["returncode"] != 0:
        print(r["tail"])


def main() -> None:
    args = parse_args()
    if args.sweep:
        _run_sweep(args)
        return
    recipe = builders.Recipe() | builders.csv(EDGE_DATASET, sample_size=args.sample)
    recipe = (
        recipe
        | builders.select(EDGE_FEATURES, EDGE_LABEL)
        | builders.quantile_uniform(n_quantiles=1000, output_distribution="uniform")
        | builders.pls_to_pow2(components=8)
        | builders.device(name=os.environ.get("QML_DEVICE", "lightning.qubit"))
        | builders.encoder("angle_embedding_y")
        | builders.ansatz("strongly_entangling", layers=4)
        | builders.train(lr=args.lr, batch=EDGE_BATCH_SIZE, epochs=args.epochs, seed=args.seed, test_size=0.2, stratify=True)
    )

    if not args.no_save:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = args.out or f"models/edgeiiot_bin_pls8_L4_{ts}.pt"
        recipe = recipe | builders.save(out_path)

    builders.run(recipe)


if __name__ == "__main__":
    main()
