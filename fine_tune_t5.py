import argparse
import json
import os
import torch
import time
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
# Ensure connection to HF mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
def load_jsonl_to_dataset(file_path, tokenizer, max_src_len=128, max_tgt_len=128):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
            
    # Convert to Hugging Face Dataset
    raw_dataset = Dataset.from_list(data_list)
    
    def preprocess_function(examples):
        inputs = ["translate Chinese to English: " + ex for ex in examples["zh"]]
        targets = examples["en"]
        
        model_inputs = tokenizer(inputs, max_length=max_src_len, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_tgt_len, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=["zh", "en", "index"])
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to raw train.jsonl")
    parser.add_argument("--valid_file", type=str, required=True, help="Path to raw valid.jsonl")
    parser.add_argument("--model_name", type=str, default="google/mt5-small") # Consider using 'google/mt5-small' for better Chinese support
    parser.add_argument("--output_dir", type=str, default="./t5_result")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    # 1. Load Tokenizer & Model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 2. Prepare Data
    print("Processing Data...")
    train_dataset = load_jsonl_to_dataset(args.train_file, tokenizer)
    valid_dataset = load_jsonl_to_dataset(args.valid_file, tokenizer)

    # 3. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=False, # T5 often has issues with fp16, causing NaN loss
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        bf16=True,     # Enable BF16 (H100 exclusive acceleration and numerical stability)
        tf32=True,     # (Optional) Enable TF32 acceleration, usually enabled by default but better to be explicit      
        dataloader_num_workers=4,  
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

# 4. Start Training & Track Metrics
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print("Starting Training...")
    train_result = trainer.train()
    
    # 5. Extract Efficiency Metrics
    metrics = train_result.metrics
    
    # 获取显存峰值
    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        metrics["peak_vram_mb"] = peak_mem_bytes / 1024 / 1024
    
    # 获取 Trainer 的完整日志历史 (包含每一步的 Loss，用于画收敛图)
    # trainer.state.log_history 是一个列表，包含 {'loss':..., 'step':...}
    history = trainer.state.log_history
    
    # Save Custom Logs
    final_stats = {
        "train_runtime": metrics["train_runtime"],
        "train_samples_per_second": metrics["train_samples_per_second"],
        "total_flos": metrics["total_flos"], # 浮点运算总量，衡量计算复杂度
        "peak_vram_mb": metrics.get("peak_vram_mb", 0),
        "history": history  # 包含 loss 曲线数据
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_logs.json"), "w") as f:
        json.dump(final_stats, f, indent=4)
        
    print("Saving Model...")
    trainer.save_model()
    print(f"Logs saved to {args.output_dir}/training_logs.json")
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()