#!/bin/sh
# table3_b_keysize.sh

VOCAB_DIR="artifacts_pretrained"

# d_k=16 실험
python demo_wmt14_pretrained.py --checkpoint_dir "checkpoints/table3_b_dk16" --kdim 16 --max_steps 100000
python inference.py --checkpoint_dir "checkpoints/table3_b_dk16" --vocab_dir ${VOCAB_DIR} --output_file "results/table3_b_dk16.txt"

# d_k=32 실험
python demo_wmt14_pretrained.py --checkpoint_dir "checkpoints/table3_b_dk32" --kdim 32 --max_steps 100000
python inference.py --checkpoint_dir "checkpoints/table3_b_dk32" --vocab_dir ${VOCAB_DIR} --output_file "results/table3_b_dk32.txt"