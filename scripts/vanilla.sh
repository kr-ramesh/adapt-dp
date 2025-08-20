DATASET=${1:-wikipedia-large}             # Default: 'med-summ'
#MODELNAME=${2:-EleutherAI/gpt-j-6B}
MODELNAME="EleutherAI/gpt-neo-125M"
SAVEDIR=${3:-/export/fs06/kramesh3/dp-fact/models-dp/my_privacy_gpt-j-6B_model_v2}
BATCHSIZE=${4:-128}
EPOCHS=${5:-1}
LR=${6:-1e-4}
SEQUENCE_LEN=${7:-512}

EPSILON=None

LR=1e-4
gradient_accumulation_steps=128
EPOCHS=5
lora_alpha=512
lora_r=4
# Replace model_name / with _
MODELNAMEFILE=$(echo "$MODELNAME" | tr '/' '_')
SAVE_DIR="/export/fs06/kramesh3/dp-fact/vanilla-pld-dp-models"
mkdir -p "$SAVE_DIR"
SAVEDIR="$SAVE_DIR/test_${MODELNAMEFILE}_model_lora_r_${lora_r}_lora_alpha_${lora_alpha}_batchsize_${BATCHSIZE}_epochs_${EPOCHS}_gas_${gradient_accumulation_steps}_noise_multiplier_${noise_multiplier}_sequence_len_${SEQUENCE_LEN}_learning_rate_${LR}"
echo "Training $MODELNAME on $DATASET; saving to $SAVEDIR (epsilon=$EPSILON)"
#torchrun --nproc_per_node=1 
master_port=$(shuf -i 10000-20000 -n 1)  # Random port for distributed training
echo "Using master port: $master_port"
for noise_multiplier in 8.0; do #0.05 0.01; do
    echo "Training with noise_multiplier: $noise_multiplier"
    python -m torch.distributed.launch --nproc_per_node=1 --master_port "$master_port" train.py --dataset_name "$DATASET" --path_to_dataset "/export/fs06/kramesh3/datasets/${DATASET}" \
                                        --model_name "$MODELNAME" --path_to_save_model "$SAVEDIR" --per_device_train_batch_size "$BATCHSIZE" \
                                        --epochs "$EPOCHS" --gradient_accumulation_steps $gradient_accumulation_steps --project_name "vanilla-pld" \
                                        --lr "$LR" --epochs "$EPOCHS" --target_epsilon "$noise_multiplier" --clipping 'flat' --max_grad_norm 1.0 --lora_r $lora_r --lora_alpha $lora_alpha --pld
done
echo "Training completed. Model saved to $SAVEDIR"
