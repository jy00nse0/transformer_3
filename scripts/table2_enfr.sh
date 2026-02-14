#!/bin/sh
# bash scripts/table2_enfr.sh | tee logs/table2_enfr$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"
NAME="table3_base"
CHECKPOINT_DIR="checkpoints/${NAME}/model_best.pt"

# Training (Base model configuration)
python demo_wmt14_pretrained.py \
    --dataset en2fr \
    --save_dir ${VOCAB_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --gradient_checkpointing \
    --n_head 8 \
    --max_steps 100000
    
# Inference & Evaluation (Average last 5 checkpoints for 'base' model)
python inference.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --vocab_dir ${VOCAB_DIR} \
    --split test \
    --output_file "results/${NAME}_decode.txt"

echo "✅ Finished table3_base"
