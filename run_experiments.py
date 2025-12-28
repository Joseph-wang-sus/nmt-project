import argparse
import itertools
import subprocess
import os
import json
import sys
import glob
import pandas as pd
from datetime import datetime

# ==========================================
# 1. Hyperparameter Grids
# ==========================================

# Grid for RNN Experiments
RNN_GRID = {
    "rnn_type": ["gru", "lstm"],
    "attention_type": ["dot", "general", "additive"],
    "hidden_size": [256, 512],
    "learning_rate": [1e-3, 5e-4],
    "teacher_forcing_ratio": [0.0, 0.5, 1.0],
}

# Grid for Transformer Experiments
TRANSFORMER_GRID = {
    "pos_type": ["absolute", "relative"],
    "norm_type": ["layernorm", "rmsnorm"],
    "d_model": [256, 512],
    "n_layers": [4, 6],
    "lr": [1e-3, 5e-4],
    "batch_size": [32, 64],
}

# ==========================================
# 2. Helper Functions
# ==========================================

def format_val(x):
    """Formats float/numbers for compact filenames."""
    if isinstance(x, float):
        # 0.001 -> 1e-3, 1.0 -> 1.0
        return f"{x:.0e}" if x < 0.01 else f"{x}"
    return str(x)

def make_run_name(model_type, params):
    """Generates a descriptive run name based on parameters."""
    if model_type == "rnn":
        # e.g. rnn_gru_dot_h512_lr1e-3_tf0.5
        return (f"rnn_{params['rnn_type']}_{params['attention_type']}_"
                f"h{params['hidden_size']}_lr{format_val(params['learning_rate'])}_"
                f"tf{format_val(params['teacher_forcing_ratio'])}")
    
    elif model_type == "transformer":
        # e.g. tfm_abs_ln_d256_L4_lr1e-4_bs32
        p_short = "abs" if params['pos_type'] == "absolute" else "rel"
        n_short = "ln" if params['norm_type'] == "layernorm" else "rms"
        return (f"tfm_{p_short}_{n_short}_d{params['d_model']}_"
                f"L{params['n_layers']}_lr{format_val(params['lr'])}_"
                f"bs{params['batch_size']}")
    return "unknown_run"

def run_training(script_name, run_name, params, args):
    """Constructs command and runs subprocess with real-time output logging."""
    
    # 1. Setup Directories
    output_dir = os.path.join(args.runs_dir, args.model_type, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Save Config for record keeping
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({**params, "model_type": args.model_type}, f, indent=4)
        
    # 3. Construct Command
    # add "-u" to force unbuffered output, so you see prints immediately
    cmd = [sys.executable, "-u", script_name] 
    
    # Add Data Args
    cmd.extend(["--data_path", os.path.join(args.data_dir, "processed_data.pt")])
    cmd.extend(["--save_dir", output_dir]) 
    cmd.extend(["--epochs", str(args.epochs)])
    
    # Explicitly pass 512 for Transformer safety
    if args.model_type == "transformer":
        cmd.extend(["--max_len", "512"])

    if args.do_test_eval:
        cmd.append("--do_test_eval")
        cmd.extend(["--test_file", os.path.join(args.data_dir, "test.jsonl")])
        cmd.extend(["--decode_strategy", args.decode_strategy])
        cmd.extend(["--beam_size", str(args.beam_size)])

    for key, val in params.items():
        flag = f"--{key}"
        if key == "learning_rate": flag = "--lr"
        if key == "teacher_forcing_ratio": flag = "--teacher_forcing"
        cmd.extend([flag, str(val)])
        
    log_file = os.path.join(output_dir, "train.log")
    print(f"[{datetime.now().strftime('%H:%M')}] Starting: {run_name}")
    print(f"   > Streaming Output to Console & {log_file} ...\n")
    
    # === [修改核心] 使用 Popen 实现双向输出 ===
    with open(log_file, "w") as f_log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 把错误信息也合并到标准输出
            text=True,                # 以文本形式读取
            bufsize=1                 # 行缓冲
        )
        
        # 逐行读取输出
        for line in process.stdout:
            # 1. 打印到屏幕 (sys.stdout.write 不会自动加换行符，因为 line 里已经有了)
            sys.stdout.write(line) 
            sys.stdout.flush() # 强制刷新缓冲区，确保实时显示
            
            # 2. 写入到文件
            f_log.write(line)
            
        # 等待进程结束
        return_code = process.wait()
        
        if return_code != 0:
            print(f"\n   [FAILED]  Error in {run_name}. Check logs above.")
        else:
            print(f"\n   [SUCCESS] Finished {run_name}")

def summarize_results(args):
    base_path = os.path.join(args.runs_dir, args.model_type)
    records = []
    
    metric_files = glob.glob(os.path.join(base_path, "*", "metrics.json"))
    for m_file in metric_files:
        run_dir = os.path.dirname(m_file)
        run_name = os.path.basename(run_dir)
        try:
            with open(m_file, 'r') as f: metrics = json.load(f)
            config_file = os.path.join(run_dir, "config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f: config = json.load(f)
                records.append({**config, **metrics, "run_name": run_name})
        except: pass

    if records:
        df = pd.DataFrame(records)
        out_csv = os.path.join(args.runs_dir, f"{args.model_type}_summary.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nSummary saved to: {out_csv}")
        print(df[["run_name", "test_bleu", "best_val_loss"]].head())

# ==========================================
# 3. Main Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["rnn", "transformer"])
    parser.add_argument("--data_dir", type=str, default="data_processed")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    
    # Test Evaluation Settingsssssssssssssssssssssssssssssss
    parser.add_argument("--do_test_eval", action="store_true")
    parser.add_argument("--decode_strategy", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam_size", type=int, default=3)
    
    parser.add_argument("--summarize_only", action="store_true")
    
    args = parser.parse_args()
    if args.summarize_only:
        summarize_results(args)
        return

    grid = RNN_GRID if args.model_type == "rnn" else TRANSFORMER_GRID
    script = "train_rnn.py" if args.model_type == "rnn" else "train_transformer.py"

    keys, values = grid.keys(), grid.values()
    combinations = list(itertools.product(*values))
    
    if args.max_runs: combinations = combinations[:args.max_runs]
    
    print(f"Starting {len(combinations)} experiments for {args.model_type}...")

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        run_name = make_run_name(args.model_type, params)
        run_training(script, run_name, params, args)

    summarize_results(args)

if __name__ == "__main__":
    main()