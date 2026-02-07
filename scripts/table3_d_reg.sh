#!/bin/sh

## bash scripts/table3_d_reg.sh | tee logs/table3_d_$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"

# Dropout 실험 (0.0, 0.2)
for p in 0.0 0.2; do
    python demo_wmt14_pretrained.py --checkpoint_dir "checkpoints/table3_d_drop${p}" --drop_prob ${p} --max_steps 100000
    python inference.py --checkpoint_dir "checkpoints/table3_d_drop${p}" --vocab_dir ${VOCAB_DIR} --output_file "results/table3_d_drop${p}.txt"
done

# Label Smoothing 실험 (0.0, 0.2)
for ls in 0.0 0.2; do
    python demo_wmt14_pretrained.py --checkpoint_dir "checkpoints/table3_d_ls${ls}" --label_smoothing ${ls} --max_steps 100000
    python inference.py --checkpoint_dir "checkpoints/table3_d_ls${ls}" --vocab_dir ${VOCAB_DIR} --output_file "results/table3_d_ls${ls}.txt"
done