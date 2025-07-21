DATASET=${1:-wiki-merged}             # Default: 'med-summ'
MODELNAME=${2:-EleutherAI/gpt-j-6B}
MODELNAME="EleutherAI/gpt-neo-125M"
SAVEDIR=${3:-/export/fs06/kramesh3/dp-fact/models-dp/my_privacy_gpt-j-6B_model_v2}
BATCHSIZE=${4:-128}
EPOCHS=${5:-1}
LR=${6:-1e-4}
SEQUENCE_LEN=${7:-512}

EPSILON=None

LR=5e-4
gradient_accumulation_steps=128
EPOCHS=5
SAVEDIR="/export/fs06/kramesh3/dp-fact/vanilla-dp-models/my_privacy_gpt-j-6B_model_clipbound_lr_${clipbound_learning_rate}_target_quantile_${target_unclipped_quantile}_batchsize_${BATCHSIZE}_epochs_${EPOCHS}_gas_${gradient_accumulation_steps}_noise_multiplier_${noise_multiplier}_sequence_len_${SEQUENCE_LEN}_learning_rate_${LR}"
echo "Training $MODELNAME on $DATASET; saving to $SAVEDIR (epsilon=$EPSILON)"
#torchrun --nproc_per_node=1 
master_port=$(shuf -i 10000-20000 -n 1)  # Random port for distributed training
echo "Using master port: $master_port"
for noise_multiplier in 0.3; do #0.05 0.01; do
    echo "Training with noise_multiplier: $noise_multiplier"
    python -m torch.distributed.launch --nproc_per_node=1 --master_port "$master_port" train.py --dataset_name "$DATASET" --path_to_dataset "/export/fs06/kramesh3/datasets/wiki-merged" \
                                        --model_name "$MODELNAME" --path_to_save_model "$SAVEDIR" --per_device_train_batch_size "$BATCHSIZE" \
                                        --epochs "$EPOCHS" --gradient_accumulation_steps $gradient_accumulation_steps --project_name "vanilla" \
                                        --lr "$LR" --epochs "$EPOCHS" --noise_multiplier "$noise_multiplier" --clipping 'flat' --max_grad_norm 0.01 --lora_r 8 --lora_alpha 512
done
echo "Training completed. Model saved to $SAVEDIR"
