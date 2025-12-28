
import json
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path("/data/250010144/nmt-project/checkpoints")  # 根据需要修改

def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None

def parse_config_from_folder(folder: Path) -> Dict[str, Any]:
    """
    解析文件夹名称中的超参数。
    支持 RNN (rnn_type...) 和 Transformer (tfm_...) 格式。
    """
    name = folder.name
    cfg: Dict[str, Any] = {"exp_name": name}
    parts = name.split("_")

    # === Transformer ===
    # 示例: tfm_d256_h4_l6_bs128_lr0p001_drop0p1_ls0p05_wd0p0001_ms64_mt70_posabsolute_normlayernorm
    if name.startswith("tfm"):
        cfg["model_type"] = "transformer"
        for p in parts:
            if p.startswith("d") and p[1:].isdigit():
                cfg["d_model"] = int(p[1:])
            elif p.startswith("h") and p[1:].isdigit():
                cfg["n_heads"] = int(p[1:])
            elif p.startswith("l") and p[1:].isdigit():
                cfg["n_layers"] = int(p[1:])
            elif p.startswith("bs"):
                cfg["batch_size"] = int(p[2:])
            elif p.startswith("lr"):
                cfg["learning_rate"] = float(p[2:].replace("p", "."))
            elif p.startswith("drop"):
                cfg["dropout"] = float(p[4:].replace("p", "."))
            elif p.startswith("pos"):
                cfg["pos_type"] = p[3:]
            elif p.startswith("norm"):
                cfg["norm_type"] = p[4:]
            elif p.startswith("ls"):
                cfg["label_smoothing"] = float(p[2:].replace("p", "."))
            elif p.startswith("wd"):
                cfg["weight_decay"] = float(p[2:].replace("p", "."))

    # === RNN ===
    # 示例: rnn_typegru_attndot_hid256_emb256_bs64_lr0p0005_tf0_drop0p1
    else:
        cfg["model_type"] = "rnn"
        for p in parts:
            if p.startswith("rnn_type"):
                cfg["rnn_type"] = p.replace("rnn_type", "")
            elif p.startswith("attn"):
                cfg["attention_type"] = p.replace("attn", "")
            elif p.startswith("hid"):
                cfg["hidden_size"] = int(p.replace("hid", ""))
            elif p.startswith("emb"):
                cfg["embed_size"] = int(p.replace("emb", ""))
            elif p.startswith("bs"):
                cfg["batch_size"] = int(p.replace("bs", ""))
            elif p.startswith("lr"):
                lr_str = p.replace("lr", "").replace("p", ".")
                try:
                    cfg["learning_rate"] = float(lr_str)
                except ValueError:
                    cfg["learning_rate"] = lr_str
            elif p.startswith("tf"):
                tf_str = p.replace("tf", "")
                try:
                    cfg["teacher_forcing_ratio"] = float(tf_str)
                except ValueError:
                    cfg["teacher_forcing_ratio"] = tf_str
            elif p.startswith("drop"):
                drop_str = p.replace("drop", "").replace("p", ".")
                try:
                    cfg["dropout"] = float(drop_str)
                except ValueError:
                    cfg["dropout"] = drop_str

    return cfg

def collect_one_experiment(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """
    exp_dir 类似：
    /data/.../checkpoints/rnn_gru_dot_h256_bs64_lr5e-4_tf0.0/rnn_typegru_attndot_hid256_emb256_bs64_lr0p0005_tf0_drop0p1
    """
    metrics_path = exp_dir / "metrics.json"
    logs_path = exp_dir / "training_logs.json"

    metrics = safe_load_json(metrics_path)
    logs = safe_load_json(logs_path)

    if metrics is None and logs is None:
        # 这个目录不是一个完整实验
        return None

    row: Dict[str, Any] = {}

    # 解析目录层级信息（外层和内层）
    row["outer_dir"] = exp_dir.parent.name
    row.update(parse_config_from_folder(exp_dir))

    # metrics.json 信息
    if metrics is not None:
        for k, v in metrics.items():
            row[f"metrics_{k}"] = v

    # training_logs.json 信息
    if logs is not None:
        # 所有 epoch 的参数可以直接放 history 列，也可以拆开
        history: List[Dict[str, Any]] = logs.get("history", [])
        row["training_history"] = history  # 原样保留
        # 单独取 total_time
        row["total_training_time"] = logs.get("total_time", None)
        # 如果有峰值显存
        if "peak_vram_mb" in logs:
            row["peak_vram_mb"] = logs["peak_vram_mb"]

    return row

def main():
    results: List[Dict[str, Any]] = []

    # 遍历 checkpoints 下所有子目录
    for root, dirs, files in os.walk(ROOT):
        root_path = Path(root)
        # 只在“最内层”目录尝试读取 metrics/training_logs
        if "metrics.json" in files or "training_logs.json" in files:
            exp = collect_one_experiment(root_path)
            if exp is not None:
                results.append(exp)

    # 输出为 JSON
    out_json = ROOT / "experiments_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON summary to {out_json}")

    # 同时输出一个简化版 CSV（只挑部分关键字段）
    out_csv = ROOT / "experiments_summary.csv"
    # 选几个常用列；若某些字段不存在会自动填空
    fieldnames = [
        "outer_dir",
        "exp_name",
        "model_type",
        # RNN params
        "rnn_type",
        "attention_type",
        "hidden_size",
        "embed_size",
        "teacher_forcing_ratio",
        # Transformer params
        "d_model",
        "n_heads",
        "n_layers",
        "pos_type",
        "norm_type",
        # Common params
        "batch_size",
        "learning_rate",
        "dropout",
        "label_smoothing",
        "weight_decay",
        # Metrics
        "metrics_best_val_loss",
        "metrics_peak_vram",
        "metrics_decode_strategy",
        "metrics_beam_size",
        "total_training_time",
        "peak_vram_mb",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            flat = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(flat)

    print(f"Saved CSV summary to {out_csv}")

if __name__ == "__main__":
    main()
