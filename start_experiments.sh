#!/usr/bin/env bash
set -euo pipefail

# Root & data/checkpoint dirs
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT_DIR/data"
PROC_DIR="$ROOT_DIR/data_processed"
CHECKPOINT_DIR="$ROOT_DIR/checkpoints"
mkdir -p "$PROC_DIR" "$CHECKPOINT_DIR"

PROCESSED_PT="$PROC_DIR/processed_data.pt"

# If not preprocessed yet, you can follow your preprocessing script here; otherwise comment out
if [[ ! -f "$PROCESSED_PT" ]]; then
  echo "Expected $PROCESSED_PT to exist. Please run preprocessing first."
  exit 1
fi

echo "[1/2] RNN grid search starting..."

RNN_TYPES=("lstm")
ATTN_TYPES=("dot" "general" "additive")
HIDDEN_SIZES=(256)
LRS=("1e-3")
TEACHER_FORCING=(0.0 1.0)
BATCH_SIZES=(128)

for rnn_type in "${RNN_TYPES[@]}"; do
  for attn_type in "${ATTN_TYPES[@]}"; do
    for hidden_size in "${HIDDEN_SIZES[@]}"; do
      for lr in "${LRS[@]}"; do
        for tf in "${TEACHER_FORCING[@]}"; do
          for bs in "${BATCH_SIZES[@]}"; do
            save_dir="$CHECKPOINT_DIR/rnn_${rnn_type}_${attn_type}_h${hidden_size}_bs${bs}_lr${lr}_tf${tf}"
            mkdir -p "$save_dir"
            echo "--> RNN rnn_type=${rnn_type} attn=${attn_type} hidden=${hidden_size} bs=${bs} lr=${lr} tf=${tf}"
            python train_rnn.py \
              --data_path "$PROCESSED_PT" \
              --save_dir "$save_dir" \
              --rnn_type "$rnn_type" \
              --attention_type "$attn_type" \
              --hidden_size "$hidden_size" \
              --embed_size 256 \
              --batch_size "$bs" \
              --lr "$lr" \
              --epochs 15 \
              --teacher_forcing "$tf"
          done
        done
      done
    done
  done
done

echo "[2/2] RNN grid search complete. Checkpoints under $CHECKPOINT_DIR"
