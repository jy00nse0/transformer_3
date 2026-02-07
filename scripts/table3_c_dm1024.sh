#!/bin/sh
## bash scripts/table3_c_dm1024.sh | tee logs/table3_c_dm1024_$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"

echo "================================================================================"
echo "🚀 Experiment: table3_c_dm1024 (d_model=1024)"
echo "   Config -> N:6, d_model:1024, d_ff:4096, d_k:128"
echo "================================================================================"

# Training
python demo_wmt14_pretrained.py \
    --load_dir ${VOCAB_DIR} \
    --save_dir ${VOCAB_DIR} \
    --checkpoint_dir "checkpoints/table3_c_dm1024" \
    --gradient_checkpointing \
    --n_layers 6 \
    --d_model 1024 \
    --ffn_hidden 4096 \
    --custom \
    --d_k 128 \
    --max_steps 100000

# Inference & Evaluation
python inference.py \
    --checkpoint_path "checkpoints/table3_c_dm1024/model_averaged_last5.pt" \
    --vocab_dir ${VOCAB_DIR} \
    --output_file "results/table3_c_dm1024_translations.txt"

echo "✅ Finished table3_c_dm1024"
