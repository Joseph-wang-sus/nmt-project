#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT_DIR/data"
PROC_DIR="$ROOT_DIR/data_processed"
CHECKPOINT_DIR="$ROOT_DIR/checkpoints"
mkdir -p "$PROC_DIR" "$CHECKPOINT_DIR"

echo "[1/4] Running preprocessing pipeline..."
python preprocess.py \
  --train_file "$DATA_DIR/train_100k.jsonl" \
  --valid_file "$DATA_DIR/valid.jsonl" \
  --output_dir "$PROC_DIR" \
  --vocab_size_en 16000 \
  --min_freq_zh 2 \
  --max_len 160

PROCESSED_PT="$PROC_DIR/processed_data.pt"
if [[ ! -f "$PROCESSED_PT" ]]; then
  echo "Expected $PROCESSED_PT to exist after preprocessing." >&2
  exit 1
fi

TRANSFORMER_D_MODELS=(128 256)
TRANSFORMER_LRS=(5e-4 1e-3)
TRANSFORMER_LAYERS=(2 3)

echo "[2/4] Launching Transformer training grid..."
for d_model in "${TRANSFORMER_D_MODELS[@]}"; do
  for n_layers in "${TRANSFORMER_LAYERS[@]}"; do
    for lr in "${TRANSFORMER_LRS[@]}"; do
      n_heads=$((d_model / 32))
      (( n_heads < 2 )) && n_heads=2
      save_dir="$CHECKPOINT_DIR/transformer_d${d_model}_l${n_layers}_lr${lr}"
      mkdir -p "$save_dir"
      echo "--> Transformer d_model=$d_model n_layers=$n_layers lr=$lr n_heads=$n_heads"
      python train_transformer.py \
        --data_path "$PROCESSED_PT" \
        --save_path "$save_dir" \
        --d_model "$d_model" \
        --n_heads "$n_heads" \
        --n_layers "$n_layers" \
        --batch_size 128 \
        --lr "$lr" \
        --epochs 10 \
        --norm_type layernorm \
        --pos_type absolute
    done
  done
done

echo "[3/4] Launching RNN training grid..."
RNN_TYPES=(gru lstm)
ATTN_TYPES=(dot general)
TEACHER_FORCING=(0.5 0.7)
HIDDEN_SIZES=(256 384)

for rnn_type in "${RNN_TYPES[@]}"; do
  for attn_type in "${ATTN_TYPES[@]}"; do
    for teacher_ratio in "${TEACHER_FORCING[@]}"; do
      for hidden_size in "${HIDDEN_SIZES[@]}"; do
        save_dir="$CHECKPOINT_DIR/rnn_${rnn_type}_${attn_type}_h${hidden_size}_tf${teacher_ratio}"
        mkdir -p "$save_dir"
        echo "--> RNN rnn_type=$rnn_type attention=$attn_type hidden=$hidden_size tf=$teacher_ratio"
        python train_rnn.py \
          --data_path "$PROCESSED_PT" \
          --save_dir "$save_dir" \
          --rnn_type "$rnn_type" \
          --attention_type "$attn_type" \
          --hidden_size "$hidden_size" \
          --embed_size 256 \
          --batch_size 96 \
          --lr 0.001 \
          --epochs 10 \
          --teacher_forcing "$teacher_ratio"
      done
    done
  done
done

echo "[4/4] Pipeline complete. Checkpoints stored under $CHECKPOINT_DIR"
