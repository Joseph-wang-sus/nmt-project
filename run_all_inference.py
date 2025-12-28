#!/usr/bin/env python3
"""Batch runner that evaluates every checkpoint under checkpoints/."""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def deslug(value: str) -> str:
    """Inverse of the slug used in training run names."""
    return value.replace("m", "-").replace("p", ".")


def parse_float(token: str) -> float:
    try:
        return float(deslug(token))
    except ValueError as exc:
        raise ValueError(f"Could not parse float from '{token}'") from exc


def parse_rnn_config(inner_name: str) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    for chunk in inner_name.split("_"):
        if chunk.startswith("type"):
            cfg["rnn_type"] = chunk[4:]
        elif chunk.startswith("attn"):
            cfg["attention_type"] = chunk[4:]
        elif chunk.startswith("hid"):
            cfg["hidden_size"] = int(chunk[3:])
        elif chunk.startswith("emb"):
            cfg["embed_size"] = int(chunk[3:])
        elif chunk.startswith("drop"):
            cfg["dropout"] = parse_float(chunk[4:])
    required = {"rnn_type", "attention_type", "hidden_size", "embed_size", "dropout"}
    missing = sorted(required - cfg.keys())
    if missing:
        raise ValueError(f"Missing RNN config fields {missing} in '{inner_name}'")
    return cfg


def discover_checkpoints(root: Path) -> List[Path]:
    return sorted(root.glob("*/*/model_best.pt"))


def build_common_args(args: argparse.Namespace) -> List[str]:
    extra: List[str] = []
    if args.test_file:
        extra += ["--test_file", args.test_file]
    if args.data_path:
        extra += ["--data_path", args.data_path]
    if args.limit is not None:
        extra += ["--limit", str(args.limit)]
    if args.beam_size is not None:
        extra += ["--beam_size", str(args.beam_size)]
    return extra


def run_single_checkpoint(
    ckpt_path: Path,
    repo_root: Path,
    output_root: Path,
    common_args: List[str],
    skip_existing: bool,
) -> bool:
    outer_name = ckpt_path.parents[1].name
    inner_name = ckpt_path.parent.name

    if outer_name.startswith("transformer"):
        model_type = "transformer"
        script_path = repo_root / "inference_transformer.py"
        output_file = output_root / model_type / f"{outer_name}.csv"
        cmd = [
            sys.executable,
            str(script_path),
            "--checkpoint_path",
            str(ckpt_path),
            "--output_file",
            str(output_file),
        ] + common_args
    elif outer_name.startswith("rnn"):
        model_type = "rnn"
        script_path = repo_root / "inference_rnn.py"
        cfg = parse_rnn_config(inner_name)
        output_file = output_root / model_type / f"{outer_name}.csv"
        cmd = [
            sys.executable,
            str(script_path),
            "--checkpoint_path",
            str(ckpt_path),
            "--output_file",
            str(output_file),
            "--rnn_type",
            cfg["rnn_type"],
            "--attention_type",
            cfg["attention_type"],
            "--hidden_size",
            str(cfg["hidden_size"]),
            "--embed_size",
            str(cfg["embed_size"]),
            "--dropout",
            str(cfg["dropout"]),
        ] + common_args
    else:
        raise ValueError(f"Unknown checkpoint family for '{outer_name}'")

    if skip_existing and output_file.exists():
        print(f"Skipping {outer_name} because {output_file} already exists.")
        return True

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print("\n=== Running", outer_name, "===")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=repo_root)
    if completed.returncode != 0:
        print(f"Run failed for {outer_name} (exit code {completed.returncode}).")
    return completed.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for every checkpoint.")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints")
    parser.add_argument("--output_root", type=str, default="batch_inference_outputs")
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    checkpoint_root = (repo_root / args.checkpoint_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(checkpoint_root)
    if not checkpoints:
        print(f"No model_best.pt files under {checkpoint_root}")
        sys.exit(1)

    common_args = build_common_args(args)
    total = len(checkpoints)
    successes = 0
    failures: List[str] = []

    for ckpt in checkpoints:
        try:
            ok = run_single_checkpoint(
                ckpt_path=ckpt,
                repo_root=repo_root,
                output_root=output_root,
                common_args=common_args,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:
            ok = False
            print(f"Error while preparing run for {ckpt}: {exc}")
        if ok:
            successes += 1
        else:
            failures.append(str(ckpt))

    print(f"\nFinished {total} runs: {successes} succeeded, {total - successes} failed.")
    if failures:
        print("Failures:")
        for item in failures:
            print(f"- {item}")


if __name__ == "__main__":
    main()
