#!/bin/bash

lora_weights_path="/export/fs06/kramesh3/dp-fact/adapt-dp-models/wiki-merged-split-200_gpt-j-6B_model_clipbound_lr_0.2_target_quantile_0.3_batchsize_1024_epochs_5_gas_1024_noise_multiplier__sequence_len_256_learning_rate_1e-4"
lora_weights_path="/export/fs06/kramesh3/dp-fact/vanilla-dp-models/wiki-merged-split-200_gpt-j-6B_model_lora_r_4_lora_alpha_512_batchsize_128_epochs_5_gas_128_noise_multiplier__sequence_len_256_learning_rate_1e-4"
lora_weights_dir="/export/fs06/kramesh3/dp-fact/vanilla-dp-models/"
model_path="wiki-merged-split-200_gpt-j-6B_model_lora_r_4_lora_alpha_512_batchsize_1024_epochs_25_gas_1024_noise_multiplier_0.05_sequence_len_256_learning_rate_1e-4"
model_path="wiki-merged-split-200_gpt-j-6B_model_lora_r_4_lora_alpha_32_batchsize_32_epochs_15_gas_32_noise_multiplier_0.0_sequence_len_256_learning_rate_1e-4"
lora_weights_path="${lora_weights_dir}${model_path}"

model_path="wiki-merged-split-200_gpt-j-6B_model_lora_r_4_lora_alpha_512_batchsize_1024_epochs_25_gas_1024_noise_multiplier_0.1_sequence_len_256_learning_rate_1e-4"

python inference.py \
    --model_name EleutherAI/gpt-j-6B \
    --lora_weights_path "$lora_weights_path" \
    --dataset_name wiki-merged \
    --path_to_dataset /export/fs06/kramesh3/datasets/wiki-merged \
    --output_file ${model_path}_inference_wiki-merged.csv \
    --sequence_len 1024 \
    --batch_size 2 \
    --num_beams 1 \
    --top_p 0.9 \
    --temperature 0.5 \
    --repetition_penalty 1.0 \
    --num_return_seq 5 \
    --max_length 256 \
    --min_new_tokens 1 \
    --eval_batch_size 2 \
    --dry_test_run \
    --device cuda
