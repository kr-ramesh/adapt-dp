#!/bin/bash
#SBATCH --job-name="inference-dp-v2"
#SBATCH --time=24:00:00
#SBATCH --output=logs/inference%j.out
#SBATCH --error=logs/inference%j.err
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

inference_paths="/export/fs06/kramesh3/dp-fact/inference-results/$(date +%Y-%m-%d_%H-%M-%S)/"
mkdir -p "$inference_paths"
# For every file in the directory /export/fs06/kramesh3/dp-fact/vanilla-dp-models/ with epochs_25 in it
for lora_weights_path in /export/fs06/kramesh3/dp-fact/adapt-dp-models/*epochs_25*; do

    if [[ "$lora_weights_path" == *wiki-merged-split-200* ]]; then
        dataset_name="wiki-merged"
    else
        dataset_name="common-pile-news-filtered"
    fi
    echo "Running inference for model: $lora_weights_path"
    if [[ "$lora_weights_path" == *common-pile* ]]; then
        echo "Skipping model: $lora_weights_path as it is not from common-pile"
        continue
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
    
    # Run the inference script
    #python inference.py \
    #    --model_name EleutherAI/gpt-j-6B \
    #    --lora_weights_path "$lora_weights_path" \
    #    --dataset_name $dataset_name \
    #    --path_to_dataset /export/fs06/kramesh3/datasets/$dataset_name \
    #    --output_file "${inference_paths}/${model_name}_inference_${dataset_name}.csv" \
    #    --sequence_len "$sequence_len" \
    #    --batch_size 2 \
    #    --num_beams 1 \
    #    --top_p 0.9 \
    #    --temperature 0.5 \
    #    --repetition_penalty 1.0 \
    #    --num_return_seq 5 \
    #    --max_length "$max_new_tokens" \
    #    --min_new_tokens 1 \
    #    --eval_batch_size 2 \
    #    --device cuda
done

# For every file in the directory /export/fs06/kramesh3/dp-fact/vanilla-dp-models/ with epochs_25 or epochs_15 in it
for lora_weights_path in /export/fs06/kramesh3/dp-fact/vanilla-dp-models/*epochs_25* /export/fs06/kramesh3/dp-fact/vanilla-dp-models/*epochs_15*; do
    echo "Running inference for model: $lora_weights_path"
    if [[ "$lora_weights_path" == *common-pile* ]]; then
        echo "Skipping model: $lora_weights_path as it is not from common-pile"
        continue
    fi
    if [[ "$lora_weights_path" == *wiki-merged-split-200* ]]; then
        dataset_name="wiki-merged"
    else
        dataset_name="common-pile-news-filtered"
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
    
    # Run the inference script
    python inference.py \
        --model_name EleutherAI/gpt-j-6B \
        --lora_weights_path "$lora_weights_path" \
        --dataset_name $dataset_name \
        --path_to_dataset /export/fs06/kramesh3/datasets/$dataset_name \
        --output_file "${inference_paths}/${model_name}_inference_${dataset_name}.csv" \
        --sequence_len "$sequence_len" \
        --batch_size 2 \
        --num_beams 1 \
        --top_p 0.9 \
        --temperature 0.5 \
        --repetition_penalty 1.0 \
        --num_return_seq 5 \
        --max_length "$max_new_tokens" \
        --min_new_tokens 1 \
        --eval_batch_size 2 \
        --device cuda
done