#!/bin/sh
## bash scripts/table3_d_ls0.2.sh | tee logs/table3_d_ls0.2_$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"

echo "================================================================================"
echo "🚀 Experiment: table3_d_ls0.2 (label_smoothing=0.2)"
echo "================================================================================"

# Training
python demo_wmt14_pretrained.py \
    --load_dir ${VOCAB_DIR} \
    --save_dir ${VOCAB_DIR} \
    --checkpoint_dir "checkpoints/table3_d_ls0.2" \
    --gradient_checkpointing \
    --label_smoothing 0.2 \
    --max_steps 100000

# Inference & Evaluation
python inference.py \
    --checkpoint_path "checkpoints/table3_d_ls0.2/model_best.pt" \
    --vocab_dir ${VOCAB_DIR} \
    --output_file "results/table3_d_ls0.2.txt"

echo "✅ Finished table3_d_ls0.2"
