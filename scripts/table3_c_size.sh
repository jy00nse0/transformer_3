#!/bin/bash
# table3_c_size.sh
# bash scripts/table3_c_size.sh | tee logs/table3_c_$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"
MAX_STEPS=100000

# Table 3 (C) 행의 7가지 실험 정의
# 형식: 실험명:N:d_model:d_ff:d_k
# d_v는 논문 명세에 따라 d_k와 동일하게 설정됨
experiments=(
    "table3_c_n2:2:512:2048:64"      # 1열: 레이어 수 감소 (N=2)
    "table3_c_n4:4:512:2048:64"      # 2열: 레이어 수 감소 (N=4)
    "table3_c_n8:8:512:2048:64"      # 3열: 레이어 수 증가 (N=8)
    "table3_c_dm256:6:256:1024:32"   # 4열: 모델 차원 축소 (d_model=256)
    "table3_c_dm1024:6:1024:4096:128" # 5열: 모델 차원 확대 (d_model=1024)
    "table3_c_ff1024:6:512:1024:64"  # 6열: FFN 차원 축소 (d_ff=1024)
    "table3_c_ff4096:6:512:4096:64"  # 7열: FFN 차원 확대 (d_ff=4096)
)

for exp in "${experiments[@]}"; do
    IFS=":" read -r NAME N DM FF DK <<< "$exp"
    CHECKPOINT_DIR="checkpoints/${NAME}"
    
    echo "================================================================================"
    echo "🚀 Running Experiment: ${NAME}"
    echo "   Config -> N:${N}, d_model:${DM}, d_ff:${FF}, d_k:${DK}"
    echo "================================================================================"

    # 1. Training (H100 최적화: max_tokens 상향 및 gradient_checkpointing 적용)
    python demo_wmt14_pretrained.py \
        --load_dir ${VOCAB_DIR} \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        --save_dir "${VOCAB_DIR}" \
        --n_layers "${N}" \
        --d_model "${DM}" \
        --ffn_hidden "${FF}" \
        --kdim "${DK}" \
        --max_steps "${MAX_STEPS}" \
        --max_tokens 60000 \
        --gradient_checkpointing \
        --num_workers 8

    # 2. Inference & Evaluation
    python inference.py \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        --vocab_dir "${VOCAB_DIR}" \
        --avg_checkpoints 5 \
        --output_file "results/${NAME}_translations.txt"
        
    echo -e "✅ Finished ${NAME}\n"
done