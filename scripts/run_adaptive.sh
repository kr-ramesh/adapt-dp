#!/bin/bash
#SBATCH --job-name="adaptive-dp"
#SBATCH --time=24:00:00
#SBATCH --output=logs/adaptive%j.out
#SBATCH --error=logs/adaptive%j.err
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

DATASET=${1:-wiki-merged-split-200}             # Default: 'med-summ'
MODELNAME=${2:-EleutherAI/gpt-j-6B}
SAVEDIR=${3:-/export/fs06/kramesh3/dp-fact/models-dp/my_privacy_gpt-j-6B_model_v2}
BATCHSIZE=${4:-4096}
EPOCHS=${5:-1}
LR=${6:-1e-4}
SEQUENCE_LEN=${7:-512}

DATASET="common-pile-news-filtered"

EPSILON=None
lora_alpha=512
lora_r=4
LR=1e-4
gradient_accumulation_steps=$BATCHSIZE
EPOCHS=25
clipbound_learning_rate=0.2
target_unclipped_quantile=0.3
min_clipbound=0.05
max_clipbound=1.0
unclipped_num_std=0.1
max_grad_norm=0.1
echo "Training $MODELNAME on $DATASET; saving to $SAVEDIR (epsilon=$EPSILON)"
#torchrun --nproc_per_node=1 
master_port=$(shuf -i 10000-20000 -n 1)  # Random port for distributed training
echo "Using master port: $master_port"

for noise_multiplier in 0.01 0.1 0.2; do #0.05 0.01; do
    unclipped_num_std=0.1
    SAVEDIR="/export/fs06/kramesh3/dp-fact/adapt-dp-models/${DATASET}_gpt-j-6B_model_clipbound_lr_${clipbound_learning_rate}_target_quantile_${target_unclipped_quantile}_batchsize_${BATCHSIZE}_epochs_${EPOCHS}_gas_${gradient_accumulation_steps}_noise_multiplier_${noise_multiplier}_sequence_len_${SEQUENCE_LEN}_learning_rate_${LR}"
    echo "Saved model directory: $SAVEDIR"
    echo "Training with noise_multiplier: $noise_multiplier"
    python -m torch.distributed.launch --nproc_per_node=1 --master_port "$master_port" train.py --dataset_name "$DATASET" --path_to_dataset "/export/fs06/kramesh3/datasets/${DATASET}" \
                                        --model_name "$MODELNAME" --path_to_save_model "$SAVEDIR" --per_device_train_batch_size "$BATCHSIZE" \
                                        --epochs "$EPOCHS" --gradient_accumulation_steps $gradient_accumulation_steps --project_name "adaclip" \
                                        --lr "$LR" --epochs "$EPOCHS" --noise_multiplier "$noise_multiplier" --clipping adaptive --lora_r $lora_r --lora_alpha $lora_alpha \
                                        --clipbound_learning_rate $clipbound_learning_rate --target_unclipped_quantile $target_unclipped_quantile --min_clipbound $min_clipbound --max_clipbound $max_clipbound --unclipped_num_std $unclipped_num_std --max_grad_norm $max_grad_norm
done
echo "Training completed. Model saved to $SAVEDIR"