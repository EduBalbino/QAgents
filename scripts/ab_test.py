import concurrent.futures
import itertools
import json
import os
from typing import Any, Dict, List

from scripts.core.builders import Recipe, csv, select, device, encoder, ansatz, train, run, save
import datetime as _dt


def build_recipe(dataset: str, features: List[str], label: str, sample: int,
                 enc_name: str, enc_opts: Dict[str, Any], layers: int,
                 meas: Dict[str, Any]) -> Recipe:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    ds_base = os.path.basename(dataset).replace(".csv", "")
    had = "H1" if enc_opts.get("hadamard", False) else "H0"
    reu = "R1" if enc_opts.get("reupload", False) else "R0"
    meas_name = meas.get("name", "z0")
    save_path = f"models/ab/{ds_base}_{enc_name}_{had}_{reu}_{meas_name}_L{layers}_{ts}.pt"

    r = (Recipe()
         | csv(dataset, sample_size=sample)
         | select(features, label=label)
         | device("lightning.qubit", wires_from_features=True)
         | encoder(enc_name, **enc_opts)
         | ansatz("ring_rot_cnot", layers=layers)
         | train(lr=0.1, batch=64, epochs=1, class_weights="balanced", seed=42)
         | save(save_path))
    # measurement is encoded via encoder options inside builders.run config
    # so we inject it at the end by mutating the last step list element
    r.parts.append(type(r.parts[0])("measurement", meas))
    return r


def pretty_row(cols: List[str], widths: List[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cols, widths))


def run_job(job):
    dataset, features, label, sample, enc_name, enc_opts, meas, layers = job
    recipe = build_recipe(dataset, features, label, sample, enc_name, enc_opts, layers, meas)
    summary = run(recipe)  # builders.run returns a dict
    # attach job definition
    summary.update({"enc_name": enc_name, "enc_opts": enc_opts, "meas": meas, "layers": layers})
    return summary


def main():
    dataset = "data/PCA_CICIDS2017.csv"
    features = [f"PC_{i}" for i in range(1, 9)]
    label = "Label"
    sample = int(os.environ.get("AB_SAMPLE", "50000"))

    encoders = [
        ("angle_embedding_y", {"hadamard": False, "reupload": False, "angle_range": None}),
        ("angle_embedding_y", {"hadamard": True,  "reupload": False, "angle_range": None}),
        ("angle_embedding_x", {"hadamard": False, "reupload": False, "angle_range": None}),
        ("angle_embedding_y", {"hadamard": False, "reupload": True,  "angle_range": "0_pi"}),
    ]

    measurements = [
        {"name": "z0", "wires": [0]},
        {"name": "mean_z", "wires": [0, 1]},
    ]

    layers_list = [2, 3]

    jobs = []
    for (enc_name, enc_opts), meas, layers in itertools.product(encoders, measurements, layers_list):
        jobs.append((dataset, features, label, sample, enc_name, enc_opts, meas, layers))

    # Shard support via env
    num_shards = int(os.environ.get("AB_NUM_SHARDS", "1"))
    shard_index = int(os.environ.get("AB_SHARD_INDEX", "0"))
    shard_jobs = jobs[shard_index::num_shards]

    results: List[Dict[str, Any]] = []
    max_workers_env = os.environ.get("AB_MAX_WORKERS")
    max_workers = int(max_workers_env) if max_workers_env else min(os.cpu_count() or 4, len(shard_jobs) or 1)
    exec_mode = os.environ.get("AB_EXEC", "process").lower()  # process | thread | sequential
    timeout_s = int(os.environ.get("AB_TIMEOUT_S", "0"))  # 0 means no timeout

    def _run_pool(executor_cls):
        nonlocal results
        with executor_cls(max_workers=max_workers) as ex:
            future_to_job = {ex.submit(run_job, job): job for job in shard_jobs}
            for future in concurrent.futures.as_completed(future_to_job, timeout=None if timeout_s <= 0 else timeout_s):
                job = future_to_job[future]
                try:
                    res = future.result(timeout=None if timeout_s <= 0 else max(1, timeout_s))
                    results.append(res)
                except Exception as e:
                    print(f"Job failed for {job}: {e}")

    print(
        f"Running shard {shard_index+1}/{num_shards} with {len(shard_jobs)} jobs on {max_workers} workers (mode={exec_mode})..."
    )
    if shard_jobs:
        try:
            if exec_mode == "thread":
                _run_pool(concurrent.futures.ThreadPoolExecutor)
            elif exec_mode == "sequential":
                for job in shard_jobs:
                    try:
                        results.append(run_job(job))
                    except Exception as e:
                        print(f"Job failed for {job}: {e}")
            else:
                _run_pool(concurrent.futures.ProcessPoolExecutor)
        except concurrent.futures.process.BrokenProcessPool as e:
            print(f"Process pool broke ({e}). Falling back to threads...")
            results = []
            _run_pool(concurrent.futures.ThreadPoolExecutor)
        except Exception as e:
            print(f"Executor failed ({e}). Falling back to sequential...")
            results = []
            for job in shard_jobs:
                try:
                    results.append(run_job(job))
                except Exception as ex:
                    print(f"Job failed for {job}: {ex}")

    # Pretty print summary table
    headers = [
        "Encoder", "Hadamard", "Reupload", "AngleScale", "Meas", "Layers",
        "Acc", "Prec", "Rec", "F1", "Log"
    ]
    rows = []
    for r in results:
        enc_opts = r.get("encoder_opts", {})
        meas = r.get("measurement", {})
        rows.append([
            r.get("encoder", ""),
            str(enc_opts.get("hadamard", False)),
            str(enc_opts.get("reupload", False)),
            str(enc_opts.get("angle_range", enc_opts.get("angle_scale", "-"))),
            f"{meas.get('name')}:{','.join(map(str, meas.get('wires', [])))}",
            str(r.get("layers", "")),
            f"{r['metrics']['accuracy']:.4f}",
            f"{r['metrics']['precision']:.4f}",
            f"{r['metrics']['recall']:.4f}",
            f"{r['metrics']['f1']:.4f}",
            os.path.basename(r.get("log_path", "")),
        ])

    # Robust width calculation without star-unpacking (handles empty rows)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                w = len(cell)
                if w > widths[i]:
                    widths[i] = w
    sep = "-+-".join("-" * w for w in widths)
    print("\n===== A/B Test Summary =====")
    print(pretty_row(headers, widths))
    print(sep)
    for row in rows:
        print(pretty_row(row, widths))
    print("===========================\n")

    # Also dump JSON for downstream tools
    out_json = f"logs/ab_results_shard_{shard_index}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved shard results to {out_json}")


if __name__ == "__main__":
    main()


