import argparse
import time
import json
import math
import os
import functools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sacrebleu

# Imports from your project structure
from models.transformer_nmt import TransformerNMT, TransformerConfig
from data_utils import clean_text, ChineseTokenizerHanLP, BPETokenizerEn
from transformer_decoding import greedy_decode, beam_search_decode


def _slugify(value):
    """Convert hyper-parameter values into filesystem-friendly tokens."""
    if isinstance(value, float):
        value = f"{value:.4g}"
    value = str(value)
    return value.replace('.', 'p').replace('-', 'm')


def _build_run_name(prefix, kv_pairs):
    parts = [prefix]
    for key, val in kv_pairs:
        parts.append(f"{key}{_slugify(val)}")
    return "_".join(parts)

# ==========================================
# 1. Dataset Wrapper
# ==========================================

class TranslationDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # data item is like {'zh': [ids...], 'en': [ids...]}
        return self.data[idx]

def collate_fn(batch, src_pad_idx, tgt_pad_idx, max_len_src=None, max_len_tgt=None):
    """
    Collate function with Dynamic Padding.
    Uses specific pad indices for source and target.
    """
    src_list = [x['zh'] for x in batch]
    tgt_list = [x['en'] for x in batch]
    
    # 1. Truncate (if global max_len is set)
    if max_len_src:
        src_list = [seq[:max_len_src] for seq in src_list]
    if max_len_tgt:
        tgt_list = [seq[:max_len_tgt] for seq in tgt_list]
        
    # 2. Dynamic Padding
    batch_max_src = max(len(s) for s in src_list)
    batch_max_tgt = max(len(t) for t in tgt_list)
    
    # Initialize with specific PAD indices
    src_batch = torch.full((len(batch), batch_max_src), src_pad_idx, dtype=torch.long)
    tgt_batch = torch.full((len(batch), batch_max_tgt), tgt_pad_idx, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_list, tgt_list)):
        src_batch[i, :len(src)] = torch.tensor(src, dtype=torch.long)
        tgt_batch[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
        
    return src_batch, tgt_batch

# ==========================================
# 2. Training Logic
# ==========================================

def train_epoch(model, dataloader, optimizer, criterion, device, log_interval=50):
    model.train()
    total_loss = 0
    epoch_loss_accum = 0
    
    step_losses = []
    
    for i, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        
        # tgt_input: <bos> ... w_n
        # tgt_out:   w_1 ... <eos>
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        
        optimizer.zero_grad()
        
        # [Single GPU Mode] Let the model generate the mask internally
        output = model(src, tgt_input)
        
        # Loss calculation
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        epoch_loss_accum += loss.item()
        
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            step_losses.append({'step': i, 'loss': cur_loss})
            print(f"| Batch {i:5d} | Loss {cur_loss:5.4f} | PPL {math.exp(cur_loss):7.2f}", flush=True)
            total_loss = 0
            
    return epoch_loss_accum / len(dataloader), step_losses

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            
            output = model(src, tgt_input)
            
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# ==========================================
# 3. Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    # Data & Path
    parser.add_argument("--data_path", type=str, default="data_processed/processed_data.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints_tfm") 
    parser.add_argument("--save_path", type=str, default=None, help="Alias of save_dir for backward compatibility")
    parser.add_argument("--experiment_name", type=str, default=None, help="Optional custom folder name for this run")
    
    # Model Config
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8) # Recommended 8 heads for 512 d_model
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--norm_type", type=str, default="rmsnorm")
    parser.add_argument("--pos_type", type=str, default="relative")
    
    parser.add_argument("--length_config", type=str, default="length_config.json")
    parser.add_argument("--max_len_src", type=int, default=None)
    parser.add_argument("--max_len_tgt", type=int, default=None)

    # Training Config
    parser.add_argument("--batch_size", type=int, default=128) # Recommended 128 or 256
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=30)
    
    # === [Regularization Parameter Optimization] ===
    parser.add_argument("--dropout", type=float, default=0.1)        # Default 0.1
    parser.add_argument("--label_smoothing", type=float, default=0.05) # Default 0.05
    parser.add_argument("--weight_decay", type=float, default=1e-4)   # Default 1e-4
    
    # Evaluation Config
    parser.add_argument("--do_test_eval", action="store_true")
    parser.add_argument("--test_file", type=str, default="data/test.jsonl")
    parser.add_argument("--decode_strategy", type=str, default="beam", choices=["greedy", "beam"])
    parser.add_argument("--beam_size", type=int, default=5)
    
    args = parser.parse_args()
    if args.save_path:
        args.save_dir = args.save_path

    # 1. Load length config
    if args.max_len_src is None or args.max_len_tgt is None:
        if os.path.exists(args.length_config):
            print(f"Loading length config from {args.length_config}")
            with open(args.length_config, 'r') as f:
                lconf = json.load(f)
                args.max_len_src = args.max_len_src or lconf.get("max_len_src", 256)
                args.max_len_tgt = args.max_len_tgt or lconf.get("max_len_tgt", 256)
        else:
            print("Warning: No length_config.json found. Using defaults (256).")
            args.max_len_src = args.max_len_src or 256
            args.max_len_tgt = args.max_len_tgt or 256

    print(f"Training with max_len_src={args.max_len_src}, max_len_tgt={args.max_len_tgt}")
    
    base_save_dir = os.path.abspath(args.save_dir)
    os.makedirs(base_save_dir, exist_ok=True)

    if args.experiment_name:
        run_name = args.experiment_name
    else:
        run_name = _build_run_name(
            "tfm",
            [
                ("d", args.d_model),
                ("h", args.n_heads),
                ("l", args.n_layers),
                ("bs", args.batch_size),
                ("lr", args.lr),
                ("drop", args.dropout),
                ("ls", args.label_smoothing),
                ("wd", args.weight_decay),
                ("ms", args.max_len_src),
                ("mt", args.max_len_tgt),
                ("pos", args.pos_type),
                ("norm", args.norm_type),
            ],
        )

    artifact_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(artifact_dir, exist_ok=True)
    print(f"Artifacts will be saved under {artifact_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Load Data
    print(f"Loading data from {args.data_path}...")
    data_dict = torch.load(args.data_path, weights_only=False)
    train_ds = TranslationDataset(data_dict['train'])
    valid_ds = TranslationDataset(data_dict['valid'])
    vocab_zh = data_dict['vocab_zh']
    vocab_en = data_dict['vocab_en']
    
    # === [Key] Get correct Pad Indices ===
    src_pad_idx = vocab_zh.pad_index
    tgt_pad_idx = vocab_en.pad_index
    print(f"Source PAD Index: {src_pad_idx}")
    print(f"Target PAD Index: {tgt_pad_idx}")
    
    # Setup Collate Function
    my_collate = functools.partial(
        collate_fn, 
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        max_len_src=args.max_len_src, 
        max_len_tgt=args.max_len_tgt
    )

    # Dataloaders (num_workers=0 for stability)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, collate_fn=my_collate, num_workers=0)
    
    # Config: pass distinct pad indices to the model
    config = TransformerConfig(
        d_model=args.d_model, n_heads=args.n_heads,
        num_encoder_layers=args.n_layers, num_decoder_layers=args.n_layers,
        vocab_size_src=len(vocab_zh), vocab_size_tgt=len(vocab_en),
        pos_embedding_type=args.pos_type, norm_type=args.norm_type,
        pad_idx=tgt_pad_idx,
        max_seq_len=max(args.max_len_src, args.max_len_tgt) + 50,
        dropout=args.dropout
    )
    
    model = TransformerNMT(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # === [Key] Loss Ignore Index uses Target Pad Index ===
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx, label_smoothing=args.label_smoothing)
    
    training_stats = {
        "config": vars(args), "history": [], 
        "total_training_time": 0, "peak_vram_mb": 0
    }
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    total_start_time = time.time()
    best_val_loss = float('inf')
    
    print("Start training...", flush=True)

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---", flush=True)
        epoch_start = time.time()
        
        train_loss_avg, _ = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, valid_loader, criterion, device)
        
        epoch_duration = time.time() - epoch_start
        print(f"Val Loss: {val_loss:.4f} | Time: {epoch_duration:.2f}s", flush=True)
        
        training_stats["history"].append({
            "epoch": epoch + 1, 
            "train_loss": train_loss_avg,
            "val_loss": val_loss,
            "ppl": math.exp(val_loss) if val_loss < 100 else -1,
            "time": epoch_duration
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state': model.state_dict(), 'config': config}, 
                       os.path.join(artifact_dir, "model_best.pt"))
            print(f"New best checkpoint saved at {artifact_dir}")

    training_stats["total_training_time"] = time.time() - total_start_time
    if torch.cuda.is_available():
        training_stats["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # === Final Test Evaluation ===
    final_metrics = {
        "best_val_loss": best_val_loss,
        "peak_vram": training_stats["peak_vram_mb"],
        "decode_strategy": args.decode_strategy,
        "beam_size": args.beam_size
    }

    if args.do_test_eval:
        print(f"\nRunning Test Eval ({args.decode_strategy}, k={args.beam_size})...", flush=True)
        test_samples = []
        if os.path.exists(args.test_file):
            with open(args.test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    test_samples.append(json.loads(line))
        else:
            print(f"Warning: Test file {args.test_file} not found.")

        zh_tokenizer = ChineseTokenizerHanLP()
        en_tokenizer = BPETokenizerEn()
        tok_path = os.path.join(os.path.dirname(args.data_path), "tokenizer_en.json")
        if os.path.exists(tok_path):
            en_tokenizer.load(tok_path)
            
        hyps, refs = [], []
        # Get special token IDs for decoding
        bos_id = vocab_en.bos_index
        eos_id = vocab_en.eos_index
        
        model.eval()
        
        for item in test_samples:
            src_text_raw = item.get('zh_hy', item.get('zh'))
            if src_text_raw is None:
                raise KeyError("Test file must contain 'zh_hy' (or fallback 'zh') field.")
            src_text = clean_text(src_text_raw)
            ref_text = item['en']
            
            # Use fixed short length for inference speed
            gen_max_len = 100 
            
            tokens = zh_tokenizer.encode(src_text)
            src_ids = vocab_zh.encode(tokens)
            src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
            
            # Inference Mask needed for decoding functions
            src_mask = model.make_src_mask(src_tensor)
            
            try:
                if args.decode_strategy == "greedy":
                    out_ids = greedy_decode(model, src_tensor, src_mask, gen_max_len, bos_id, eos_id, device)
                else:
                    out_ids = beam_search_decode(model, src_tensor, src_mask, gen_max_len, bos_id, eos_id, device, beam_size=args.beam_size)
                
                # Decode to text
                if os.path.exists(tok_path):
                    # Use tokenizer for proper BPE decoding
                    sent = en_tokenizer.tokenizer.decode(out_ids.flatten().tolist(), skip_special_tokens=True)
                else:
                    # Fallback to Vocab lookup
                    out_str = vocab_en.decode(out_ids.flatten().tolist(), remove_special=True)
                    sent = " ".join(out_str)

            except Exception as e:
                print(f"Decoding Error: {e}")
                sent = ""
            
            hyps.append(sent)
            refs.append(ref_text)
            
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        print(f"Test BLEU: {bleu.score:.2f}", flush=True)
        final_metrics["test_bleu"] = bleu.score

    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)
    
    with open(os.path.join(artifact_dir, "training_logs.json"), "w") as f:
        json.dump(training_stats, f, indent=4)
        
    print(f"Done! Saved to {artifact_dir}", flush=True)

if __name__ == "__main__":
    main()