#!/bin/bash
#SBATCH --job-name="inference-dp"
#SBATCH --time=24:00:00
#SBATCH --output=logs/inference%j.out
#SBATCH --error=logs/inference%j.err
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

inference_paths="/export/fs06/kramesh3/dp-fact/inference-results/$(date +%Y-%m-%d_%H-%M-%S)/"
mkdir -p "$inference_paths"

# Array of temperatures to test
temperatures=(0.3 0.5 0.7 1.0)

# For every file in the directory /export/fs06/kramesh3/dp-fact/vanilla-dp-models/ with epochs_25 or epochs_15 in it
#for lora_weights_path in /export/fs06/kramesh3/dp-fact/vanilla-dp-models-updated/*epochs_25* /export/fs06/kramesh3/dp-fact/vanilla-dp-models-updated/*epochs_15*; do
for lora_weights_path in /export/fs06/kramesh3/dp-fact/vanilla-dp-models/*epochs_25* /export/fs06/kramesh3/dp-fact/vanilla-dp-models-updated/*epochs_15*; do

    # if [[ "$lora_weights_path" == *common-pile* ]] and there is no target epsiloln in the path
    if [[ "$lora_weights_path" == *common-pile* ]] && [[ "$lora_weights_path" != *target_epsilon* ]]; then
        echo "Skipping model: $lora_weights_path as it is not from common-pile or does not have target epsilon"
        continue
    fi
    if [[ "$lora_weights_path" == *wiki-merged-split-200* ]]; then
        dataset_name="wiki-merged"
    elif [[ "$lora_weights_path" == *common-pile-news-filtered* ]]; then
        dataset_name="common-pile-news-filtered"
    elif [[ "$lora_weights_path" == *wikipedia-large* ]]; then
        dataset_name="wikipedia-large"
    fi

    if [[ "$lora_weights_path" == *wiki-merged* ]]; then
        sequence_len=256
        max_new_tokens=256
    else
        sequence_len=512
        max_new_tokens=512
    fi

    # Extract the model name from the path
    model_name=$(basename "$lora_weights_path")
    
    # Loop through each temperature
    for temperature in "${temperatures[@]}"; do
        echo "Running inference for model: $lora_weights_path with temperature: $temperature"
        
        # Run the inference script
        python inference.py \
            --model_name EleutherAI/gpt-j-6B \
            --lora_weights_path "$lora_weights_path" \
            --dataset_name $dataset_name \
            --path_to_dataset /export/fs06/kramesh3/datasets/$dataset_name \
            --output_file "${inference_paths}/${model_name}_inference_${dataset_name}_temperature_${temperature}.csv" \
            --sequence_len "$sequence_len" \
            --batch_size 2 \
            --num_beams 1 \
            --top_p 0.9 \
            --temperature "$temperature" \
            --repetition_penalty 1.0 \
            --num_return_seq 5 \
            --max_length "$max_new_tokens" \
            --min_new_tokens 1 \
            --eval_batch_size 2 \
            --device cuda
    done
done