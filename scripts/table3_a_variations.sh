#!/bin/sh
# table3_a_variations.sh

VOCAB_DIR="artifacts_pretrained"
# 각 실험별 설정을 배열로 정의 (실험명:헤드수)
experiments=("table3_base:8" "table3_a_h1:1" "table3_a_h4:4" "table3_a_h16:16" "table3_a_h32:32")

for exp in "${experiments[@]}"; do
    if [ "${exp}" = "table3_base:8" ]; then
        continue
    fi
    NAME="${exp%%:*}"
    HEADS="${exp##*:}"
    CHECKPOINT_DIR="checkpoints/${NAME}"
    
    echo "Starting Experiment: ${NAME} (Heads: ${HEADS})"
    
    # Training (100,000 steps per paper)
    python demo_wmt14_pretrained.py \
        --load_dir ${VOCAB_DIR} \
        --save_dir ${VOCAB_DIR} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --gradient_checkpointing \
        --n_head ${HEADS} \
        --max_steps 100000
        
    # Inference & Evaluation (use averaged checkpoint from training)
    python inference.py \
        --checkpoint_path "${CHECKPOINT_DIR}/model_averaged_last5.pt" \
        --vocab_dir ${VOCAB_DIR} \
        --output_file "results/${NAME}_decode.txt"
done