import argparse
import json
import math
import os
import numpy as np
from tqdm import tqdm

# Import your actual tokenizers from data_utils
from data_utils import ChineseTokenizerHanLP, BPETokenizerEn, clean_text

def compute_percentiles(lengths):
    """Compute basic statistics for a list of lengths."""
    if not lengths:
        return {
            "count": 0, "p95": 0, "p99": 0, 
            "mean": 0, "median": 0, "max": 0
        }
    
    return {
        "count": len(lengths),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "mean": float(np.mean(lengths)),
        "median": int(np.median(lengths)),
        "max": int(np.max(lengths))
    }

def propose_max_len(stats, margin_ratio=0.25, min_len=64, ceiling=512):
    """
    Propose a reasonable max_len based on P95 stats.
    Formula: max_len = round_up(P95 * (1 + margin))
    """
    p95 = stats['p95']
    proposal = int(p95 * (1 + margin_ratio))
    
    # Ensure minimum length
    if proposal < min_len:
        proposal = min_len
    
    # Cap at a reasonable ceiling (e.g. 512 for Transformer absolute pos encoding)
    if proposal > ceiling:
        print(f"Warning: Proposed length {proposal} exceeds ceiling {ceiling}. Capping at {ceiling}.")
        proposal = ceiling
        
    return proposal

def main():
    parser = argparse.ArgumentParser(description="Analyze token length distributions")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL data (e.g., train.jsonl)")
    parser.add_argument("--src_field", type=str, default="zh", help="Field name for source text")
    parser.add_argument("--tgt_field", type=str, default="en", help="Field name for target text")
    parser.add_argument("--vocab_dir", type=str, default="data_processed", help="Directory where tokenizer vocab exists")
    parser.add_argument("--output_json", type=str, default="length_stats.json", help="Output stats file")
    parser.add_argument("--output_config", type=str, default="length_config.json", help="Output config file for training")
    
    args = parser.parse_args()
    
    print("Initializing tokenizers...")
    # 1. Chinese (HanLP)
    zh_tokenizer = ChineseTokenizerHanLP()
    
    # 2. English (BPE)
    # Assumes 'tokenizer_en.json' exists in vocab_dir. Adjust path if necessary.
    tokenizer_en_path = os.path.join(args.vocab_dir, "tokenizer_en.json")
    if os.path.exists(tokenizer_en_path):
        en_tokenizer = BPETokenizerEn(vocab_size=10000) # Size is placeholder; load overrides it
        en_tokenizer.load(tokenizer_en_path)
    else:
        print(f"Warning: BPE tokenizer file not found at {tokenizer_en_path}. Using un-trained BPE (counts may be inaccurate).")
        en_tokenizer = BPETokenizerEn() 

    src_lengths = []
    tgt_lengths = []
    
    print(f"Reading data from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            obj = json.loads(line)
            src_text = clean_text(obj.get(args.src_field, ""))
            tgt_text = clean_text(obj.get(args.tgt_field, ""))
            
            if not src_text or not tgt_text:
                continue
                
            # Tokenize and count
            src_tokens = zh_tokenizer.encode(src_text)
            tgt_tokens = en_tokenizer.encode(tgt_text)
            
            src_lengths.append(len(src_tokens))
            tgt_lengths.append(len(tgt_tokens))
            
    # Compute Statistics
    src_stats = compute_percentiles(src_lengths)
    tgt_stats = compute_percentiles(tgt_lengths)
    
    stats_report = {
        "source": src_stats,
        "target": tgt_stats
    }
    
    print("\n=== Length Statistics ===")
    print(json.dumps(stats_report, indent=4))
    
    # Propose max_len
    # We use a margin of 25% over P95
    prop_src = propose_max_len(src_stats, margin_ratio=0.25, ceiling=512)
    prop_tgt = propose_max_len(tgt_stats, margin_ratio=0.25, ceiling=512)
    
    # Count truncated samples
    trunc_src = sum(1 for l in src_lengths if l > prop_src)
    trunc_tgt = sum(1 for l in tgt_lengths if l > prop_tgt)
    
    print(f"\nProposed max_len_src: {prop_src} (Truncates {trunc_src}/{len(src_lengths)} samples)")
    print(f"Proposed max_len_tgt: {prop_tgt} (Truncates {trunc_tgt}/{len(tgt_lengths)} samples)")
    
    config_out = {
        "max_len_src": prop_src,
        "max_len_tgt": prop_tgt,
        "src_p95": src_stats['p95'],
        "tgt_p95": tgt_stats['p95']
    }
    
    # Save outputs
    with open(args.output_json, 'w') as f:
        json.dump(stats_report, f, indent=4)
        
    with open(args.output_config, 'w') as f:
        json.dump(config_out, f, indent=4)
        
    print(f"\nSaved stats to {args.output_json}")
    print(f"Saved config to {args.output_config}")

if __name__ == "__main__":
    main()