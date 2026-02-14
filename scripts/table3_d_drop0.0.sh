#!/bin/sh
## bash scripts/table3_d_drop0.0.sh | tee logs/table3_d_drop0.0_$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"

echo "================================================================================"
echo "🚀 Experiment: table3_d_drop0.0 (dropout=0.0)"
echo "================================================================================"

# Training
python demo_wmt14_pretrained.py \
    --load_dir ${VOCAB_DIR} \
    --save_dir ${VOCAB_DIR} \
    --checkpoint_dir "checkpoints/table3_d_drop0.0" \
    --gradient_checkpointing \
    --drop_prob 0.0 \
    --max_steps 100000

# Inference & Evaluation
python inference.py \
    --checkpoint_path "checkpoints/table3_d_drop0.0/model_best.pt" \
    --vocab_dir ${VOCAB_DIR} \
    --output_file "results/table3_d_drop0.0.txt"

echo "✅ Finished table3_d_drop0.0"
