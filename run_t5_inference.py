import argparse
import json
import torch
import pandas as pd
import sacrebleu
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run inference using fine-tuned T5 model.")
    parser.add_argument("--model_path", type=str, default="./t5_result", help="Path to the fine-tuned model directory")
    parser.add_argument("--test_file", type=str, default="data/test.jsonl", help="Path to the test dataset (JSONL)")
    parser.add_argument("--output_file", type=str, default="t5_inference_results.csv", help="Path to save the results CSV")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    print(f"Using device: {args.device}")

    # Load Tokenizer and Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        model.to(args.device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Test Data
    print(f"Loading test data from {args.test_file}...")
    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))

    print(f"Total test samples: {len(test_data)}")

    # Prepare for batch inference
    results = []
    references = []
    hypotheses = []

    # Process in batches
    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Running Inference"):
        batch = test_data[i : i + args.batch_size]
        
        # Prepare inputs with the same prefix used during training
        # Note: Check fine_tune_t5.py for the exact prefix. It was "translate Chinese to English: "
        src_texts = [item.get('zh_hy', item.get('zh')) for item in batch]
        input_texts = [f"translate Chinese to English: {text}" for text in src_texts]
        ref_texts = [item['en'] for item in batch]

        # Tokenize inputs
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        # Generate translations
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=args.max_length)

        # Decode outputs
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Collect results
        for src, ref, hyp in zip(src_texts, ref_texts, decoded_preds):
            results.append({
                "Source": src,
                "Reference": ref,
                "Hypothesis": hyp
            })
            references.append(ref)
            hypotheses.append(hyp)

    # Calculate BLEU score
    # Using force=True to ensure consistent tokenization handling
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], force=True)
    print(f"\nBLEU Score: {bleu.score:.2f}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f"Results saved to {args.output_file}")

    # Save a summary file with the score
    summary_file = args.output_file.replace(".csv", "_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test File: {args.test_file}\n")
        f.write(f"BLEU Score: {bleu.score:.2f}\n")
        f.write(f"Details: {str(bleu)}\n")
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
