#!/bin/sh
## bash scripts/table3_a_h16.sh | tee logs/table3_a_h16_$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"

echo "================================================================================"
echo "🚀 Experiment: table3_a_h16 (16 attention heads)"
echo "================================================================================"

# Training
python demo_wmt14_pretrained.py \
    --load_dir ${VOCAB_DIR} \
    --save_dir ${VOCAB_DIR} \
    --checkpoint_dir "checkpoints/table3_a_h16" \
    --gradient_checkpointing \
    --n_head 16 \
    --max_steps 100000

# Inference & Evaluation
python inference.py \
    --checkpoint_path "checkpoints/table3_a_h16/model_averaged_last5.pt" \
    --vocab_dir ${VOCAB_DIR} \
    --output_file "results/table3_a_h16_decode.txt"

echo "✅ Finished table3_a_h16"
