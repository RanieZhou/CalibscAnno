from pathlib import Path
import argparse
import json
import shutil
import subprocess
import time

import anndata as ad


def utc_now_iso():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_nvidia_smi_query():
    query = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(query, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                values.append(float(line))
            except ValueError:
                pass
    return max(values) if values else None


def wait_with_gpu_monitor(process):
    peak_gpu_memory_mb = run_nvidia_smi_query()
    while process.poll() is None:
        current = run_nvidia_smi_query()
        if current is not None:
            if peak_gpu_memory_mb is None:
                peak_gpu_memory_mb = current
            else:
                peak_gpu_memory_mb = max(peak_gpu_memory_mb, current)
        time.sleep(5)
    current = run_nvidia_smi_query()
    if current is not None:
        peak_gpu_memory_mb = max(peak_gpu_memory_mb or current, current)
    return process.returncode, peak_gpu_memory_mb


def validate_h5ad(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"h5ad not found: {path}")

    adata = ad.read_h5ad(path, backed="r")
    try:
        shape = tuple(int(v) for v in adata.shape)
        obs_cols = list(adata.obs.columns)
    finally:
        adata.file.close()

    if shape[1] != 19264:
        raise ValueError(f"Expected 19264 genes for scFoundation input, got {shape[1]} in {path}")

    return shape, obs_cols


def expected_official_output(save_dir: Path, task_name: str, ckpt_name: str, input_type: str, output_type: str, tgthighres: str):
    return save_dir / f"{task_name}_{ckpt_name}_{input_type}_{output_type}_embedding_{tgthighres}_resolution.npy"


def build_command(args, input_path: Path, save_dir: Path, task_name: str):
    cmd = [
        args.python,
        "get_embedding.py",
        "--task_name",
        task_name,
        "--input_type",
        "singlecell",
        "--output_type",
        "cell",
        "--pool_type",
        args.pool_type,
        "--tgthighres",
        args.tgthighres,
        "--data_path",
        str(input_path.resolve()),
        "--save_path",
        str(save_dir.resolve()),
        "--pre_normalized",
        args.pre_normalized,
        "--version",
        args.version,
        "--ckpt_name",
        args.ckpt_name,
    ]

    if args.demo:
        cmd.append("--demo")

    if args.version == "noversion":
        cmd.extend(["--model_path", str(args.model_path.resolve())])

    return cmd


def extract_dataset(args, dataset: str):
    model_dir = args.scfoundation_model_dir.resolve()
    get_embedding_py = model_dir / "get_embedding.py"
    if not get_embedding_py.exists():
        raise FileNotFoundError(f"get_embedding.py not found: {get_embedding_py}")

    default_ckpt = model_dir / "models" / "models.ckpt"
    checkpoint_path = args.model_path if args.version == "noversion" else default_ckpt
    checkpoint_exists = checkpoint_path.exists()

    if not args.dry_run and args.version != "noversion" and not default_ckpt.exists():
        raise FileNotFoundError(
            "scFoundation checkpoint not found. Put models.ckpt at "
            f"{default_ckpt}, or run with --version noversion --model_path <checkpoint>."
        )

    if not args.dry_run and args.version == "noversion" and not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    input_path = args.processed_dir / dataset / "adata.h5ad"
    shape, obs_cols = validate_h5ad(input_path)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = args.tmp_dir / "scfoundation_raw_outputs"
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset}_scfoundation_{args.pool_type}_{args.tgthighres}_{args.version}.npy"
    if args.demo:
        output_path = output_dir / f"{dataset}_scfoundation_demo_{args.pool_type}_{args.tgthighres}_{args.version}.npy"

    metadata_path = output_path.with_suffix(".metadata.json")
    task_name = f"{dataset}_scfoundation"
    official_output = expected_official_output(
        save_dir=save_dir,
        task_name=task_name,
        ckpt_name=args.ckpt_name,
        input_type="singlecell",
        output_type="cell",
        tgthighres=args.tgthighres,
    )
    cmd = build_command(args, input_path, save_dir, task_name)

    metadata = {
        "dataset": dataset,
        "input_path": str(input_path),
        "input_shape": shape,
        "obs_columns": obs_cols,
        "output_path": str(output_path),
        "official_output_path": str(official_output),
        "metadata_path": str(metadata_path),
        "scfoundation_model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": bool(checkpoint_exists),
        "command": cmd,
        "pool_type": args.pool_type,
        "tgthighres": args.tgthighres,
        "pre_normalized": args.pre_normalized,
        "version": args.version,
        "ckpt_name": args.ckpt_name,
        "demo": bool(args.demo),
        "dry_run": bool(args.dry_run),
        "started_at_utc": utc_now_iso(),
    }

    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    if args.dry_run:
        metadata["finished_at_utc"] = utc_now_iso()
        metadata["status"] = "dry_run_ok"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        return metadata

    if output_path.exists() and not args.force:
        metadata["finished_at_utc"] = utc_now_iso()
        metadata["status"] = "skipped_existing_output"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        print(f"Skip existing output: {output_path}")
        return metadata

    if official_output.exists() and args.force:
        official_output.unlink()

    start = time.perf_counter()
    process = subprocess.Popen(cmd, cwd=model_dir)
    returncode, peak_gpu_memory_mb = wait_with_gpu_monitor(process)
    runtime_seconds = time.perf_counter() - start

    metadata["finished_at_utc"] = utc_now_iso()
    metadata["runtime_seconds"] = float(runtime_seconds)
    metadata["peak_gpu_memory_mb_nvidia_smi"] = peak_gpu_memory_mb
    metadata["returncode"] = int(returncode)

    if returncode != 0:
        metadata["status"] = "failed"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        raise RuntimeError(f"scFoundation extraction failed for {dataset} with return code {returncode}")

    if not official_output.exists():
        metadata["status"] = "missing_official_output"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        raise FileNotFoundError(f"Expected scFoundation output not found: {official_output}")

    shutil.copy2(official_output, output_path)
    metadata["status"] = "completed"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved standardized embedding to: {output_path}")
    print(f"Saved metadata to: {metadata_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["zheng68k", "segerstolpe"])
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--tmp_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--scfoundation_model_dir", type=Path, default=Path("external/scFoundation/model"))
    parser.add_argument("--model_path", type=Path, default=Path("external/scFoundation/model/models/models.ckpt"))
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--pool_type", choices=["all", "max"], default="all")
    parser.add_argument("--tgthighres", type=str, default="t4")
    parser.add_argument("--pre_normalized", choices=["F", "T", "A"], default="F")
    parser.add_argument("--version", type=str, default="ce")
    parser.add_argument("--ckpt_name", type=str, default="01B-resolution")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    all_metadata = []
    for dataset in args.datasets:
        all_metadata.append(extract_dataset(args, dataset))

    print("\nCompleted dataset checks/runs:")
    for metadata in all_metadata:
        print(f"- {metadata['dataset']}: {metadata['status']}")


if __name__ == "__main__":
    main()
