# Inference & Evaluation (Average last 5 checkpoints for 'base' model)
python3 inference_batched_enfr.py --checkpoint_path "checkpoints/table2_enfr/model_best_pt/model_best.pt" --vocab_dir "artifacts_enfr" --lang_pair en-fr --split test --output_file "results/table2_enfrdecode.txt"

echo "✅ Finished table2_enfr"
