#!/bin/sh
# table3_big.sh

VOCAB_DIR="artifacts_pretrained"
NAME="table3_big"
CHECKPOINT_DIR="checkpoints/${NAME}"

# Training (300,000 steps per paper)
python demo_wmt14_pretrained.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --d_model 1024 \
    --ffn_hidden 4096 \
    --n_head 16 \
    --drop_prob 0.3 \
    --max_steps 300000
    
# Inference & Evaluation (Average last 20 checkpoints for 'big' model)
python inference.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --vocab_dir ${VOCAB_DIR} \
    --avg_checkpoints 20 \
    --output_file "results/${NAME}_decode.txt"