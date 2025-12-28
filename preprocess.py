import argparse
import json
import torch
import os
from tqdm import tqdm
from data_utils import (
    clean_text, filter_example, 
    ChineseTokenizerHanLP, BPETokenizerEn, 
    Vocab
)

def main():
    parser = argparse.ArgumentParser(description="NMT Preprocessing Pipeline")
    parser.add_argument("--train_file", type=str, required=True, help="Path to input JSONL (e.g. train_100k.jsonl)")
    parser.add_argument("--valid_file", type=str, required=True, help="Path to valid JSONL")
    parser.add_argument("--output_dir", type=str, default="./data_processed", help="Output directory")
    parser.add_argument("--vocab_size_en", type=int, default=10000, help="BPE Vocab size for English")
    parser.add_argument("--min_freq_zh", type=int, default=3, help="Min freq for Chinese tokens")
    parser.add_argument("--max_len", type=int, default=100, help="Max sequence length")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load and Clean Raw Data
    def load_data(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                src_text = obj.get('zh_hy', obj.get('zh'))
                if src_text is None:
                    raise KeyError("Expected 'zh_hy' (or fallback 'zh') in input JSON line: {}".format(line[:120]))
                pair = filter_example(src_text, obj['en'], max_len=args.max_len)
                if pair:
                    data.append(pair) # (zh, en)
        return data

    print("Loading data...")
    train_data = load_data(args.train_file)
    valid_data = load_data(args.valid_file)
    print(f"Train samples: {len(train_data)}, Valid samples: {len(valid_data)}")

    # 2. Initialize Tokenizers
    print("Initializing tokenizers...")
    
    # Chinese: HanLP (Pre-trained)
    zh_tokenizer = ChineseTokenizerHanLP()
    
    # English: BPE (Needs training)
    # We save English sentences to a temp file to train BPE
    temp_en_file = os.path.join(args.output_dir, "temp_en_corpus.txt")
    with open(temp_en_file, "w", encoding="utf-8") as f:
        for _, en in train_data:
            f.write(en + "\n")
            
    en_tokenizer = BPETokenizerEn(vocab_size=args.vocab_size_en)
    print("Training BPE for English...")
    en_tokenizer.train([temp_en_file])
    en_tokenizer.save(os.path.join(args.output_dir, "tokenizer_en.json"))
    os.remove(temp_en_file) # cleanup

    # 3. Tokenize Data
    print("Tokenizing data (this may take a while for HanLP)...")
    
    def tokenize_dataset(dataset):
        processed = []
        for zh, en in tqdm(dataset):
            zh_toks = zh_tokenizer.encode(zh)
            en_toks = en_tokenizer.encode(en)
            processed.append({"zh": zh_toks, "en": en_toks})
        return processed

    train_tokenized = tokenize_dataset(train_data)
    valid_tokenized = tokenize_dataset(valid_data)

    # 4. Build Vocabularies
    print("Building Vocabularies...")
    # For Chinese, we build from tokenized data
    zh_iterator = (item["zh"] for item in train_tokenized)
    vocab_zh = Vocab(zh_iterator, min_freq=args.min_freq_zh)
    
    # For English, BPE handles its own vocab, but we wrap it in our Vocab class for consistency
    # (Extracting vocab from tokenizer object usually requires accessing its internal dict)
    en_vocab_dict = en_tokenizer.tokenizer.get_vocab()
    # We create a simple wrapper that respects the BPE IDs
    # Note: BPE tokenizer handles IDs internally, but for consistent API we might map them.
    # To keep it simple: We will just rely on BPE's IDs for English numericalization 
    # but store a dummy Vocab object if needed for consistency. 
    # Actually, let's build a Vocab object that mirrors the BPE mapping.
    vocab_en = Vocab()
    vocab_en.stoi = en_vocab_dict
    vocab_en.itos = [k for k, v in sorted(en_vocab_dict.items(), key=lambda item: item[1])]
    
    print(f"ZH Vocab Size: {len(vocab_zh)}")
    print(f"EN Vocab Size: {len(vocab_en)}")

    # 5. Numericalize and Save
    print("Numericalizing...")
    
    def numericalize(dataset):
        indices = []
        for item in dataset:
            zh_ids = vocab_zh.encode(item["zh"])
            # For English BPE, we can use the tokenizer's encode directly or the vocab map
            # Using vocab_en.encode ensures we add BOS/EOS consistently
            en_ids = vocab_en.encode(item["en"])
            indices.append({"zh": zh_ids, "en": en_ids})
        return indices

    train_indices = numericalize(train_tokenized)
    valid_indices = numericalize(valid_tokenized)

    output_data = {
        "train": train_indices,
        "valid": valid_indices,
        "vocab_zh": vocab_zh,
        "vocab_en": vocab_en
    }
    
    save_path = os.path.join(args.output_dir, "processed_data.pt")
    torch.save(output_data, save_path)
    print(f"Done! Processed data saved to {save_path}")

if __name__ == "__main__":
    main()