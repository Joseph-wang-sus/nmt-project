import torch
import argparse
import json
import pandas as pd
import sacrebleu
from tqdm import tqdm
from models.rnn_nmt import RNNNMT, RNNConfig
from decoding_utils import greedy_decode, beam_search_decode
from data_utils import ChineseTokenizerHanLP, clean_text, BPETokenizerEn
import os

def calculate_bleu(hypotheses, references):
    """
    Calculates BLEU score using SacreBLEU.
    Args:
        hypotheses: List[str] (Model translations)
        references: List[str] (Ground truth)
    """
    # SacreBLEU expects references as a list of lists (for multiple refs support)
    # Here we have 1 reference per sentence.
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], force=True)
    return bleu.score

def main():
    parser = argparse.ArgumentParser(description="Evaluate RNN NMT Model")
    
    # 1. Data Paths
    parser.add_argument("--test_file", type=str, default="data/test.jsonl", help="Path to test set (JSONL)")
    parser.add_argument("--data_path", type=str, default="data_processed/processed_data.pt", help="Path to processed vocab data")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--output_file", type=str, default="rnn_evaluation_results.csv", help="CSV file to save results")
    
    # 2. Model Architecture (Must match training config!)
    parser.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--attention_type", type=str, default="dot", choices=["dot", "general", "additive"])
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # 3. Decoding Settings
    parser.add_argument("--max_len", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--limit", type=int, default=None, help="Debug: Limit number of samples")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on {device}...")

    # ==========================================
    # 1. Load Resources (Vocab & Tokenizer)
    # ==========================================
    print(f"Loading vocab from {args.data_path}...")
    # 'weights_only=False' is required to load custom Vocab objects
    data_dict = torch.load(args.data_path, weights_only=False)
    vocab_zh = data_dict['vocab_zh']
    vocab_en = data_dict['vocab_en']
    
    print("Initializing HanLP Tokenizer (this takes a few seconds)...")
    zh_tokenizer = ChineseTokenizerHanLP()

    # Load BPE Tokenizer for detokenization
    print("Loading BPE Tokenizer for detokenization...")
    json_path = os.path.join(os.path.dirname(args.data_path), "tokenizer_en.json")
    if os.path.exists(json_path):
        en_tokenizer = BPETokenizerEn(model_path=json_path)
    else:
        print(f"Error: Could not find {json_path}. Decoding might look fragmented.")
        en_tokenizer = BPETokenizerEn()

    # ==========================================
    # 2. Reconstruct Model & Load Weights
    # ==========================================
    print(f"Reconstructing {args.rnn_type.upper()} model with {args.attention_type} attention...")
    config = RNNConfig(
        vocab_size_src=len(vocab_zh),
        vocab_size_tgt=len(vocab_en),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        rnn_type=args.rnn_type,
        attention_type=args.attention_type,
        dropout=args.dropout,
        pad_idx_src=vocab_zh.pad_index,
        pad_idx_tgt=vocab_en.pad_index
    )
    
    model = RNNNMT(config).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    # Load state dict (weights)
    state_dict = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # ==========================================
    # 3. Load Test Data
    # ==========================================
    print(f"Loading test data from {args.test_file}...")
    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
            
    if args.limit:
        test_data = test_data[:args.limit]
        print(f"WARNING: Limiting evaluation to first {args.limit} samples.")

    # ==========================================
    # 4. Main Evaluation Loop
    # ==========================================
    results = []
    refs = []
    hyps_greedy = []
    hyps_beam = []
    
    bos_id = vocab_en.stoi['<bos>']
    eos_id = vocab_en.stoi['<eos>']
    
    print(f"Starting decoding (Greedy & Beam k={args.beam_size})...")
    
    for item in tqdm(test_data, desc="Evaluating"):
        src_text = item.get('zh_hy', item.get('zh'))
        if src_text is None:
            raise KeyError("Test file must contain 'zh_hy' (or fallback 'zh').")
        ref_text = item['en']
        
        # 1. Preprocess Source (On-the-fly)
        clean_src = clean_text(src_text)
        src_tokens = zh_tokenizer.encode(clean_src)
        src_ids = vocab_zh.encode(src_tokens) # [len]
        
        # Convert to tensor and add batch dim [1, len]
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_len_tensor = torch.tensor([len(src_ids)], dtype=torch.long).to(device)

        # 2. Greedy Decoding
        try:
            greedy_ids = greedy_decode(model, src_tensor, src_len_tensor, args.max_len, bos_id, eos_id)
            # Remove batch dim and decode to string
            greedy_sent = en_tokenizer.tokenizer.decode(greedy_ids[0].tolist(), skip_special_tokens=True)
        except Exception as e:
            print(f"Error in Greedy: {e}")
            greedy_sent = ""

        # 3. Beam Search Decoding
        try:
            # beam_search_decode returns best sequence tensor [1, len]
            beam_ids = beam_search_decode(model, src_tensor, src_len_tensor, args.max_len, bos_id, eos_id, beam_size=args.beam_size)
            beam_sent = en_tokenizer.tokenizer.decode(beam_ids[0].tolist(), skip_special_tokens=True)
        except Exception as e:
            print(f"Error in Beam: {e}")
            beam_sent = ""

        # 4. Collect Results
        refs.append(ref_text)
        hyps_greedy.append(greedy_sent)
        hyps_beam.append(beam_sent)
        
        # results.append({
        #     "Source (ZH)": src_text,
        #     "Reference (EN)": ref_text,
        #     "Greedy": greedy_sent,
        #     "Beam": beam_sent,
        #     "Fluency (1-5)": "", # To be filled manually
        #     "Adequacy (1-5)": "" # To be filled manually
        # })

    # ==========================================
    # 5. Compute Metrics & Save
    # ==========================================
    print("\n" + "="*30)
    print("       RNN EVALUATION REPORT       ")
    print("="*30)
    
    score_greedy = calculate_bleu(hyps_greedy, refs)
    score_beam = calculate_bleu(hyps_beam, refs)
    
    print(f"Model: {args.rnn_type.upper()} + {args.attention_type} Attention")
    print(f"BLEU-4 (Greedy): {score_greedy:.2f}")
    print(f"BLEU-4 (Beam k={args.beam_size}):  {score_beam:.2f}")
    
    # # Save CSV
    # df = pd.DataFrame(results)
    # df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    # print(f"\nResults saved to {args.output_file}")
    # print("Note: Please manually fill in 'Fluency' and 'Adequacy' columns for analysis.")

if __name__ == "__main__":
    main()