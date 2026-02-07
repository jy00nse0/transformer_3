#!/bin/sh
## bash scripts/table3_c_n4.sh | tee logs/table3_c_n4_$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"

echo "================================================================================"
echo "🚀 Experiment: table3_c_n4 (N=4 layers)"
echo "   Config -> N:4, d_model:512, d_ff:2048, d_k:64"
echo "================================================================================"

# Training
python demo_wmt14_pretrained.py \
    --load_dir ${VOCAB_DIR} \
    --save_dir ${VOCAB_DIR} \
    --checkpoint_dir "checkpoints/table3_c_n4" \
    --gradient_checkpointing \
    --n_layers 4 \
    --d_model 512 \
    --ffn_hidden 2048 \
    --kdim 64 \
    --max_steps 100000

# Inference & Evaluation
python inference.py \
    --checkpoint_path "checkpoints/table3_c_n4/model_averaged_last5.pt" \
    --vocab_dir ${VOCAB_DIR} \
    --output_file "results/table3_c_n4_translations.txt"

echo "✅ Finished table3_c_n4"
