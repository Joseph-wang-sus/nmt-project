#!/usr/bin/env python3
import argparse
import subprocess
import sys
import pandas as pd
import sacrebleu
from pathlib import Path
from typing import Dict, List, Any

def deslug(value: str) -> str:
    """Inverse of the slug used in training run names."""
    return value.replace("m", "-").replace("p", ".")

def parse_float(token: str) -> float:
    try:
        return float(deslug(token))
    except ValueError:
        return 0.0

def parse_transformer_config(inner_name: str) -> Dict[str, Any]:
    cfg = {}
    # Example: tfm_d256_h4_l6_bs128_lr0p0005_drop0p1_ls0p05_wd0p0001_ms64_mt70_posrelative_normlayernorm
    parts = inner_name.split("_")
    for part in parts:
        if part == "tfm": continue
        if part.startswith("d") and part[1:].isdigit(): cfg["d_model"] = int(part[1:])
        elif part.startswith("h") and part[1:].isdigit(): cfg["n_heads"] = int(part[1:])
        elif part.startswith("l") and part[1:].isdigit(): cfg["n_layers"] = int(part[1:])
        elif part.startswith("bs"): cfg["batch_size"] = int(part[2:])
        elif part.startswith("lr"): cfg["lr"] = parse_float(part[2:])
        elif part.startswith("drop"): cfg["dropout"] = parse_float(part[4:])
        elif part.startswith("ls"): cfg["label_smoothing"] = parse_float(part[2:])
        elif part.startswith("wd"): cfg["weight_decay"] = parse_float(part[2:])
        elif part.startswith("pos"): cfg["pos_type"] = part[3:]
        elif part.startswith("norm"): cfg["norm_type"] = part[4:]
    return cfg

def calculate_bleu(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        # Check columns based on inference_transformer.py output
        if "Reference" not in df.columns:
            print(f"Warning: {csv_path} missing 'Reference' column.")
            return 0.0
            
        refs = [df["Reference"].tolist()]
        beam_hyps = df["Beam"].fillna("").tolist() if "Beam" in df.columns else []
        
        bleu_beam = sacrebleu.corpus_bleu(beam_hyps, refs).score if beam_hyps else 0.0
        return bleu_beam
    except Exception as e:
        print(f"Error calculating BLEU for {csv_path}: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Run inference for Transformer checkpoints.")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints")
    parser.add_argument("--output_root", type=str, default="batch_inference_outputs/transformer")
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    checkpoint_root = (repo_root / args.checkpoint_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Find Transformer checkpoints
    checkpoints = sorted(checkpoint_root.glob("transformer_*/*/model_best.pt"))
    
    if not checkpoints:
        print(f"No Transformer model_best.pt files found under {checkpoint_root}")
        sys.exit(0)

    results = []
    
    for ckpt in checkpoints:
        outer_name = ckpt.parents[1].name
        inner_name = ckpt.parent.name
        
        print(f"\n=== Processing {outer_name} ===")
        
        params = parse_transformer_config(inner_name)
        output_file = output_root / f"{outer_name}.csv"
        
        # Build command
        cmd = [
            sys.executable,
            str(repo_root / "inference_transformer.py"),
            "--checkpoint_path", str(ckpt),
            "--output_file", str(output_file),
        ]
        
        if args.test_file: cmd.extend(["--test_file", args.test_file])
        if args.data_path: cmd.extend(["--data_path", args.data_path])
        if args.limit: cmd.extend(["--limit", str(args.limit)])
        if args.beam_size: cmd.extend(["--beam_size", str(args.beam_size)])

        # Run inference
        if args.skip_existing and output_file.exists():
            print(f"Skipping inference (file exists): {output_file}")
        else:
            print("Running inference...")
            ret = subprocess.run(cmd, cwd=repo_root)
            if ret.returncode != 0:
                print(f"Inference failed for {outer_name}")
                continue

        # Calculate BLEU
        if output_file.exists():
            bb = calculate_bleu(output_file)
            print(f"  >> BLEU Beam: {bb:.2f}")
            
            res_entry = {
                "run_name": outer_name,
                "bleu_beam": bb,
                **params
            }
            results.append(res_entry)
        else:
            print(f"Output file not found: {output_file}")

    # Save Summary
    if results:
        df = pd.DataFrame(results)
        # Reorder columns
        first_cols = ["run_name", "bleu_beam", "d_model", "n_layers", "n_heads"]
        cols = first_cols + [c for c in df.columns if c not in first_cols]
        df = df[cols]
        
        summary_path = output_root / "transformer_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        print(df.to_string())

if __name__ == "__main__":
    main()
