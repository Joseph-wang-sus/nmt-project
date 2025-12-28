import torch
import torch.nn.functional as F
import argparse
import json
import os
import pandas as pd
import sacrebleu
from tqdm import tqdm
from models.transformer_nmt import TransformerNMT, TransformerConfig
from data_utils import ChineseTokenizerHanLP, clean_text, BPETokenizerEn # [New] Import BPE class

# ==========================================
# 1. Beam Search Implementation
# ==========================================
# (Beam Search function remains unchanged, omitted for brevity, logic is consistent with before)
def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device, 
                       beam_size=5, length_penalty=1.0, repetition_penalty=1.0):
    batch_size = src.size(0)
    memory = model.encode(src, src_mask)
    memory = memory.repeat_interleave(beam_size, dim=0)
    src_mask = src_mask.repeat_interleave(beam_size, dim=0)
    ys = torch.ones(batch_size * beam_size, 1).fill_(start_symbol).type(torch.long).to(device)
    scores = torch.zeros(batch_size * beam_size).to(device)
    scores[1:] = -1e9 
    finished_seqs = []
    finished_scores = []

    for i in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys).to(device)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        logits = model.generator(out[:, -1])
        
        if repetition_penalty > 1.0:
            for b in range(beam_size):
                prev_tokens = torch.unique(ys[b])
                for t in prev_tokens:
                    if logits[b, t] < 0: logits[b, t] *= repetition_penalty
                    else: logits[b, t] /= repetition_penalty

        log_probs = F.log_softmax(logits, dim=-1)
        next_scores = scores.unsqueeze(1) + log_probs
        next_scores_flat = next_scores.view(-1)
        topk_scores, topk_indices = torch.topk(next_scores_flat, beam_size)
        
        vocab_size = log_probs.size(1)
        beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size
        
        new_scores = []
        active_beams = 0
        next_ys_tensor = torch.zeros(beam_size, ys.size(1) + 1, dtype=torch.long).to(device)
        
        for k in range(beam_size):
            idx = beam_indices[k]
            token = token_indices[k]
            score = topk_scores[k]
            prev_seq = ys[idx]
            if token.item() == end_symbol:
                lp = (len(prev_seq) + 1) ** length_penalty
                finished_seqs.append(torch.cat([prev_seq, torch.tensor([token]).to(device)]))
                finished_scores.append(score / lp)
                new_scores.append(-1e9) 
                next_ys_tensor[k] = torch.cat([prev_seq, torch.tensor([end_symbol]).to(device)]) 
            else:
                new_seq = torch.cat([prev_seq, torch.tensor([token]).to(device)])
                next_ys_tensor[k] = new_seq
                new_scores.append(score)
                active_beams += 1
        
        scores = torch.tensor(new_scores).to(device)
        ys = next_ys_tensor
        if active_beams == 0: break
            
    if len(finished_seqs) == 0:
        best_idx = torch.argmax(scores)
        return ys[best_idx]
    else:
        best_idx = finished_scores.index(max(finished_scores))
        return finished_seqs[best_idx]

# ==========================================
# 2. Main Logic (Beam-Only)
# ==========================================
def calculate_metrics(hypotheses, references):
    return sacrebleu.corpus_bleu(hypotheses, [references], force=True).score

def main():
    parser = argparse.ArgumentParser(description="Evaluate NMT with Proper Detokenization")
    # ... (Arguments remain unchanged)
    parser.add_argument("--test_file", type=str, default="data/test.jsonl")
    parser.add_argument("--data_path", type=str, default="data_processed/processed_data.pt")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="evaluation_results.csv")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.50)
    parser.add_argument("--length_config", type=str, default="length_config.json")
    parser.add_argument("--max_len_src", type=int, default=None)
    parser.add_argument("--max_len_tgt", type=int, default=None)
    parser.add_argument("--vocab_dir", type=str, default="data_processed", help="Dir containing tokenizer_en.json") # [New]

    args = parser.parse_args()

    # (Config processing remains unchanged)
    if args.max_len_src is None or args.max_len_tgt is None:
        if os.path.exists(args.length_config):
            with open(args.length_config, 'r') as f:
                lconf = json.load(f)
                if args.max_len_src is None: args.max_len_src = lconf.get("max_len_src", 64)
                if args.max_len_tgt is None: args.max_len_tgt = lconf.get("max_len_tgt", 80)
    if args.max_len_src is None: args.max_len_src = 64
    if args.max_len_tgt is None: args.max_len_tgt = 80
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Load Resources ---
    print("Loading vocab and model...")
    data_dict = torch.load(args.data_path, weights_only=False)
    vocab_zh = data_dict['vocab_zh']
    vocab_en = data_dict['vocab_en']

    # === [Key Change 1] Load real BPE Tokenizer ===
    print("Loading BPE Tokenizer for detokenization...")
    # Assume tokenizer_en.json and processed_data.pt are in the same folder
    json_path = os.path.join(os.path.dirname(args.data_path), "tokenizer_en.json")
    if os.path.exists(json_path):
        en_tokenizer = BPETokenizerEn(model_path=json_path)
    else:
        print(f"Error: Could not find {json_path}. Decoding might look fragmented.")
        en_tokenizer = BPETokenizerEn()
    # ============================================

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    if 'config' in checkpoint: config = checkpoint['config']
    else: config = TransformerConfig(vocab_size_src=len(vocab_zh), vocab_size_tgt=len(vocab_en), max_seq_len=512)

    model = TransformerNMT(config).to(device)
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # --- 2. Load Data ---
    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f: test_data.append(json.loads(line))
    if args.limit: test_data = test_data[:args.limit]

    zh_tokenizer = ChineseTokenizerHanLP()
    results, refs, hyps_beam = [], [], []
    start_symbol = vocab_en.stoi['<bos>']
    end_symbol = vocab_en.stoi['<eos>']

    print(f"Starting decoding...")
    
    for item in tqdm(test_data):
        src_text = item.get('zh_hy', item.get('zh'))
        if src_text is None:
            raise KeyError("Test file must contain 'zh_hy' (or fallback 'zh').")
        tgt_text_ref = item['en']
        
        # Preprocess
        clean_src = clean_text(src_text)
        src_tokens = zh_tokenizer.encode(clean_src)
        if len(src_tokens) > args.max_len_src: src_tokens = src_tokens[:args.max_len_src]
        src_ids = vocab_zh.encode(src_tokens, add_bos=False, add_eos=False)
        src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src_tensor)

        try:
            beam_out_ids = beam_search_decode(
                model, src_tensor, src_mask, args.max_len_tgt, 
                start_symbol, end_symbol, device, 
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty
            )
            # === [Key Change 2] Use Tokenizer decode ===
            beam_sent = en_tokenizer.tokenizer.decode(beam_out_ids.tolist(), skip_special_tokens=True)
        except Exception as e:
            print(f"Error: {e}")
            beam_sent = ""

        refs.append(tgt_text_ref)
        hyps_beam.append(beam_sent)
        
        results.append({
            "Source": src_text, "Reference": tgt_text_ref,
            "Beam": beam_sent
        })

    # Metrics
    print("\n" + "="*30)
    bleu_beam = calculate_metrics(hyps_beam, refs)
    print(f"BLEU-4 (Beam): {bleu_beam:.2f}")
    
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()