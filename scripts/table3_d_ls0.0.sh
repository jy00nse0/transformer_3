#!/bin/sh
## bash scripts/table3_d_ls0.0.sh | tee logs/table3_d_ls0.0_$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"

echo "================================================================================"
echo "🚀 Experiment: table3_d_ls0.0 (label_smoothing=0.0)"
echo "================================================================================"

# Training
python demo_wmt14_pretrained.py \
    --load_dir ${VOCAB_DIR} \
    --save_dir ${VOCAB_DIR} \
    --checkpoint_dir "checkpoints/table3_d_ls0.0" \
    --gradient_checkpointing \
    --label_smoothing 0.0 \
    --max_steps 100000

# Inference & Evaluation
python inference.py \
    --checkpoint_path "checkpoints/table3_d_ls0.0/model_averaged_last5.pt" \
    --vocab_dir ${VOCAB_DIR} \
    --output_file "results/table3_d_ls0.0.txt"

echo "✅ Finished table3_d_ls0.0"
