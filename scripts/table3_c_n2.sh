#!/bin/sh
## bash scripts/table3_c_n2.sh | tee logs/table3_c_n2_$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"

echo "================================================================================"
echo "🚀 Experiment: table3_c_n2 (N=2 layers)"
echo "   Config -> N:2, d_model:512, d_ff:2048, d_k:64"
echo "================================================================================"

# Training
python demo_wmt14_pretrained.py \
    --load_dir ${VOCAB_DIR} \
    --save_dir ${VOCAB_DIR} \
    --checkpoint_dir "checkpoints/table3_c_n2" \
    --gradient_checkpointing \
    --n_layers 2 \
    --d_model 512 \
    --ffn_hidden 2048 \
    --kdim 64 \
    --max_steps 100000

# Inference & Evaluation
python inference.py \
    --checkpoint_path "checkpoints/table3_c_n2/model_averaged_last5.pt" \
    --vocab_dir ${VOCAB_DIR} \
    --output_file "results/table3_c_n2_translations.txt"

echo "✅ Finished table3_c_n2"
