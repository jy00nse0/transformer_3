#!/bin/sh
## bash scripts/table3_b_dk16.sh | tee logs/table3_b_dk16_$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"

echo "================================================================================"
echo "🚀 Experiment: table3_b_dk16 (d_k=16)"
echo "================================================================================"

# Training
python demo_wmt14_pretrained.py \
    --load_dir ${VOCAB_DIR} \
    --save_dir ${VOCAB_DIR} \
    --checkpoint_dir "checkpoints/table3_b_dk16" \
    --gradient_checkpointing \
    --custom \
    --d_k 16 \
    --max_steps 100000

# Inference & Evaluation
python inference_batched.py \
    --checkpoint_path "checkpoints/table3_b_dk16/model_best.pt" \
    --vocab_dir ${VOCAB_DIR} \
    --output_file "results/table3_b_dk16.txt"

echo "✅ Finished table3_b_dk16"
