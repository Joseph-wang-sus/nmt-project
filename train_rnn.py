import argparse
import time
import json
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sacrebleu
import functools

# --- Helpers for run naming ---

def _slugify(value):
    if isinstance(value, float):
        value = f"{value:.4g}"
    value = str(value)
    return value.replace('.', 'p').replace('-', 'm')


def _build_run_name(prefix, kv_pairs):
    parts = [prefix]
    for key, val in kv_pairs:
        parts.append(f"{key}{_slugify(val)}")
    return "_".join(parts)

# Imports from your project
from models.rnn_nmt import RNNNMT, RNNConfig
from decoding_utils import greedy_decode, beam_search_decode
from data_utils import clean_text, ChineseTokenizerHanLP

# --- Dataset ---
class NMTDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, src_pad_idx=1, tgt_pad_idx=1, max_len_src=None, max_len_tgt=None):
    # Sort by source sentence length (required by pack_padded_sequence)
    batch.sort(key=lambda x: len(x['zh']), reverse=True)
    
    src_list = [x['zh'] for x in batch]
    tgt_list = [x['en'] for x in batch]
    
    # 1. Truncate (if limits are set)
    if max_len_src:
        src_list = [seq[:max_len_src] for seq in src_list]
    if max_len_tgt:
        tgt_list = [seq[:max_len_tgt] for seq in tgt_list]

    # 2. Get max length in current batch (Dynamic Padding)
    src_lens = [len(s) for s in src_list]
    tgt_lens = [len(t) for t in tgt_list]
    
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    
    # 3. Pad
    src_tensor = torch.full((len(batch), max_src), src_pad_idx, dtype=torch.long)
    tgt_tensor = torch.full((len(batch), max_tgt), tgt_pad_idx, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_list, tgt_list)):
        src_tensor[i, :len(src)] = torch.tensor(src, dtype=torch.long)
        tgt_tensor[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
        
    # Note: RNN needs src_lens for pack_padded_sequence
    return src_tensor, torch.tensor(src_lens), tgt_tensor

# --- Train/Eval Loops ---
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, src_len, tgt in loader:
            src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
            output = model(src, src_len, tgt, teacher_forcing_ratio=1.0)
            loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def train_epoch(model, loader, optimizer, criterion, device, teacher_ratio, log_interval=50):
    model.train()
    total_loss = 0
    epoch_loss_accum = 0
    start_time = time.time()
    for i, (src, src_len, tgt) in enumerate(loader):
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, src_len, tgt, teacher_forcing_ratio=teacher_ratio)
        loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        epoch_loss_accum += loss.item()
        if i % log_interval == 0 and i > 0:
            print(f"| Batch {i} | Loss {total_loss/log_interval:.4f}")
            total_loss = 0
    return epoch_loss_accum / len(loader)

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    # Architecture
    parser.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--attention_type", type=str, default="dot", choices=["dot", "general", "additive"])
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--teacher_forcing", type=float, default=0.0)
    parser.add_argument("--length_config", type=str, default="length_config.json")
    parser.add_argument("--max_len_src", type=int, default=None)
    parser.add_argument("--max_len_tgt", type=int, default=None)


    # Data & Path
    parser.add_argument("--data_path", type=str, default="data_processed/processed_data.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints_rnn")
    # Eval
    parser.add_argument("--do_test_eval", action="store_true")
    parser.add_argument("--test_file", type=str, default="data/test.jsonl")
    parser.add_argument("--decode_strategy", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam_size", type=int, default=5)
    
    args = parser.parse_args()
    base_save_dir = os.path.abspath(args.save_dir)
    os.makedirs(base_save_dir, exist_ok=True)

    run_name = _build_run_name(
        "rnn",
        [
            ("type", args.rnn_type),
            ("attn", args.attention_type),
            ("hid", args.hidden_size),
            ("emb", args.embed_size),
            ("bs", args.batch_size),
            ("lr", args.lr),
            ("tf", args.teacher_forcing),
            ("drop", args.dropout),
        ],
    )
    artifact_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(artifact_dir, exist_ok=True)
    print(f"Artifacts will be saved under {artifact_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # [New] 1. Load length config logic
    if args.max_len_src is None or args.max_len_tgt is None:
        if os.path.exists(args.length_config):
            print(f"Loading length config from {args.length_config}")
            with open(args.length_config, 'r') as f:
                lconf = json.load(f)
                # If not specified in CLI, use config, default to 256
                if args.max_len_src is None: args.max_len_src = lconf.get("max_len_src", 256)
                if args.max_len_tgt is None: args.max_len_tgt = lconf.get("max_len_tgt", 256)
        else:
            # Default strategy when file not found
            if args.max_len_src is None: args.max_len_src = 256
            if args.max_len_tgt is None: args.max_len_tgt = 256
            
    print(f"Training RNN with max_len_src={args.max_len_src}, max_len_tgt={args.max_len_tgt}")
    print(f"Loading data from {args.data_path}...")
    data = torch.load(args.data_path, weights_only=False)
    vocab_zh, vocab_en = data['vocab_zh'], data['vocab_en']
    src_pad_idx = vocab_zh.pad_index
    tgt_pad_idx = vocab_en.pad_index
    my_collate = functools.partial(
        collate_fn, 
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        max_len_src=args.max_len_src, 
        max_len_tgt=args.max_len_tgt
    )
    train_loader = DataLoader(NMTDataset(data['train']), batch_size=args.batch_size, collate_fn=my_collate, shuffle=True)
    valid_loader = DataLoader(NMTDataset(data['valid']), batch_size=args.batch_size, collate_fn=my_collate)
    
    config = RNNConfig(
        vocab_size_src=len(vocab_zh), vocab_size_tgt=len(vocab_en),
        rnn_type=args.rnn_type, attention_type=args.attention_type,
        hidden_size=args.hidden_size,
        embed_size=args.embed_size,
        dropout=args.dropout,
        pad_idx_src=src_pad_idx,
        pad_idx_tgt=tgt_pad_idx
    )
    
    model = RNNNMT(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx_tgt)
    
    training_stats = {"history": [], "total_time": 0, "peak_vram_mb": 0}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_total = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        start_epoch = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.teacher_forcing)
        val_loss = evaluate_loss(model, valid_loader, criterion, device)
        
        elapsed = time.time() - start_epoch
        print(f"Val Loss: {val_loss:.4f} | Time: {elapsed:.2f}s")
        training_stats["history"].append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time": elapsed,
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(artifact_dir, "model_best.pt"))
            print(f"New best checkpoint saved at {artifact_dir}")
            
    training_stats["total_time"] = time.time() - start_total
    if torch.cuda.is_available():
        training_stats["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
    # === Final Test Eval ===
    final_metrics = {
        "best_val_loss": best_val_loss,
        "peak_vram": training_stats["peak_vram_mb"],
        "decode_strategy": args.decode_strategy,
        "beam_size": args.beam_size
    }
    
    if args.do_test_eval:
        print(f"\nRunning Test Eval ({args.decode_strategy})...")
        test_samples = []
        with open(args.test_file, 'r', encoding='utf-8') as f:
            for line in f: test_samples.append(json.loads(line))
        
        zh_tokenizer = ChineseTokenizerHanLP()
        hyps, refs = [], []
        bos_id, eos_id = vocab_en.stoi['<bos>'], vocab_en.stoi['<eos>']
        
        model.eval()
        for item in test_samples:
            src_text_raw = item.get('zh_hy', item.get('zh'))
            if src_text_raw is None:
                raise KeyError("Test file must contain 'zh_hy' (or fallback 'zh').")
            src_text = clean_text(src_text_raw)
            ref_text = item['en']
            
            tokens = zh_tokenizer.encode(src_text)
            src_ids = vocab_zh.encode(tokens)
            src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
            src_len = torch.tensor([len(src_ids)]).to(device)
            
            try:
                if args.decode_strategy == "greedy":
                    out_ids = greedy_decode(model, src_tensor, src_len, 100, bos_id, eos_id)
                else:
                    out_ids = beam_search_decode(model, src_tensor, src_len, 100, bos_id, eos_id, beam_size=args.beam_size)
                
                out_str = vocab_en.decode(out_ids[0].tolist(), remove_special=True)
                sent = " ".join(out_str)
            except: sent = ""
            
            hyps.append(sent)
            refs.append(ref_text)
            
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        print(f"Test BLEU: {bleu.score:.2f}")
        final_metrics["test_bleu"] = bleu.score
        
    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)
    with open(os.path.join(artifact_dir, "training_logs.json"), "w") as f:
        json.dump(training_stats, f, indent=4)
        
    print(f"Done! Saved to {artifact_dir}")

if __name__ == "__main__":
    main()