#!/bin/bash
#SBATCH --job-name="vanilla-no-dp"
#SBATCH --time=72:00:00
#SBATCH --output=logs/nodp_all_1024%j.out
#SBATCH --error=logs/nodp_all_1024%j.err
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

MODELNAME=${2:-EleutherAI/gpt-j-6B}
SAVEDIR=${3:-/export/fs06/kramesh3/dp-fact/models-dp/my_privacy_gpt-j-6B_model_v2}
BATCHSIZE=${4:-4096}
EPOCHS=${5:-1}
LR=${6:-1e-4}
SEQUENCE_LEN=${7:-1024}

EPSILON=None

LR=1e-4
gradient_accumulation_steps=${BATCHSIZE}
EPOCHS=15
lora_alpha=512
lora_r=4
#torchrun --nproc_per_node=1
max_grad_norm=0.05
master_port=$(shuf -i 10000-20000 -n 1)  # Random port for distributed training
echo "Using master port: $master_port"
for target_epsilon in 8 16 4; do #0.05 0.01; do
for DATASET in common-pile-news-filtered wikipedia-large; do
    echo "Training on dataset: $DATASET"
    SAVEDIR="/export/fs06/kramesh3/dp-fact/vanilla-dp-models/${DATASET}_gpt-j-6B_model_lora_r_${lora_r}_lora_alpha_${lora_alpha}_batchsize_${BATCHSIZE}_epochs_${EPOCHS}_gas_${gradient_accumulation_steps}_target_epsilon_${target_epsilon}_sequence_len_${SEQUENCE_LEN}_learning_rate_${LR}"
    echo "Saved model directory: $SAVEDIR"
    python -m torch.distributed.launch --nproc_per_node=1 --master_port "$master_port" train.py --dataset_name "$DATASET" --path_to_dataset "/export/fs06/kramesh3/datasets/${DATASET}" \
                                        --model_name "$MODELNAME" --path_to_save_model "$SAVEDIR" --per_device_train_batch_size "$BATCHSIZE" \
                                        --epochs "$EPOCHS" --gradient_accumulation_steps $gradient_accumulation_steps --project_name "vanilla" \
                                        --lr "$LR" --epochs "$EPOCHS" --noise_multiplier 0.0 --clipping 'flat' --max_grad_norm $max_grad_norm --lora_r $lora_r --lora_alpha $lora_alpha --sequence_len $SEQUENCE_LEN
done
echo "Training completed. Model saved to $SAVEDIR"

SEQUENCE_LEN=512
for DATASET in common-pile-news-filtered wikipedia-large; do
    echo "Training on dataset: $DATASET"
    SAVEDIR="/export/fs06/kramesh3/dp-fact/vanilla-dp-models/${DATASET}_gpt-j-6B_model_lora_r_${lora_r}_lora_alpha_${lora_alpha}_batchsize_${BATCHSIZE}_epochs_${EPOCHS}_gas_${gradient_accumulation_steps}_target_epsilon_${target_epsilon}_sequence_len_${SEQUENCE_LEN}_learning_rate_${LR}"
    echo "Saved model directory: $SAVEDIR"
    python -m torch.distributed.launch --nproc_per_node=1 --master_port "$master_port" train.py --dataset_name "$DATASET" --path_to_dataset "/export/fs06/kramesh3/datasets/${DATASET}" \
                                        --model_name "$MODELNAME" --path_to_save_model "$SAVEDIR" --per_device_train_batch_size "$BATCHSIZE" \
                                        --epochs "$EPOCHS" --gradient_accumulation_steps $gradient_accumulation_steps --project_name "vanilla" \
                                        --lr "$LR" --epochs "$EPOCHS" --noise_multiplier 0.0 --clipping 'flat' --max_grad_norm $max_grad_norm --lora_r $lora_r --lora_alpha $lora_alpha --sequence_len $SEQUENCE_LEN
done
echo "Training completed. Model saved to $SAVEDIR"
