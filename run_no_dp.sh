#!/bin/bash
#SBATCH --job-name="vanilla-dp"
#SBATCH --time=24:00:00
#SBATCH --output=logs/vanilla%j.out
#SBATCH --error=logs/vanilla%j.err
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

DATASET=${1:-wiki-merged-split-200}             # Default: 'med-summ'
MODELNAME=${2:-EleutherAI/gpt-j-6B}
SAVEDIR=${3:-/export/fs06/kramesh3/dp-fact/models-dp/my_privacy_gpt-j-6B_model_v2}
BATCHSIZE=${4:-32}
EPOCHS=${5:-1}
LR=${6:-1e-4}
SEQUENCE_LEN=${7:-256}

EPSILON=None

LR=1e-4
gradient_accumulation_steps=${BATCHSIZE}
EPOCHS=15
lora_alpha=32
lora_r=4
max_grad_norm=1.0
echo "Training $MODELNAME on $DATASET; saving to $SAVEDIR (epsilon=$EPSILON)"
#torchrun --nproc_per_node=1
master_port=$(shuf -i 10000-20000 -n 1)  # Random port for distributed training
echo "Using master port: $master_port"
for noise_multiplier in 0.0; do #0.05 0.01; do
    SAVEDIR="/export/fs06/kramesh3/dp-fact/vanilla-dp-models/${DATASET}_gpt-j-6B_model_lora_r_${lora_r}_lora_alpha_${lora_alpha}_batchsize_${BATCHSIZE}_epochs_${EPOCHS}_gas_${gradient_accumulation_steps}_noise_multiplier_${noise_multiplier}_sequence_len_${SEQUENCE_LEN}_learning_rate_${LR}"
    echo "Saved model directory: $SAVEDIR"
    echo "Training with noise_multiplier: $noise_multiplier"
    python -m torch.distributed.launch --nproc_per_node=1 --master_port "$master_port" train.py --dataset_name "$DATASET" --path_to_dataset "/export/fs06/kramesh3/datasets/${DATASET}" \
                                        --model_name "$MODELNAME" --path_to_save_model "$SAVEDIR" --per_device_train_batch_size "$BATCHSIZE" \
                                        --epochs "$EPOCHS" --gradient_accumulation_steps $gradient_accumulation_steps --project_name "vanilla" \
                                        --lr "$LR" --epochs "$EPOCHS" --noise_multiplier "$noise_multiplier" --clipping 'flat' --max_grad_norm $max_grad_norm --lora_r $lora_r --lora_alpha $lora_alpha
done
echo "Training completed. Model saved to $SAVEDIR"