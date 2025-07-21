# train.py

import os
import torch
from transformers import HfArgumentParser
from args import ModelArguments, DataArguments, TrainArguments, PrivacyArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import ALL_DATASETS
from train_module import train

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments, PrivacyArguments))
    #model_args, data_args, training_args, privacy_args 
    #model_args, data_args, training_args, privacy_args, _ = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args, privacy_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # DDP params from env
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo") if world_size > 1 else None

    # 1. Tokenizer/model/dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name, torch_dtype=torch.float16)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for GPT-2 compatibility

    data_args.sequence_len = model_args.sequence_len
    dataset = ALL_DATASETS[data_args.dataset_name](data_args, tokenizer)

    # Tokenize data
    training_args.batch_size = training_args.per_device_train_batch_size
    
    dataset.dataset = dataset.dataset.map(
            dataset.preprocess_function, batched=True, num_proc=8, desc="tokenizing dataset",
            remove_columns=dataset.dataset.column_names['train'],
            load_from_cache_file=False)
    
    print(dataset.dataset)

    model, privacy_engine = train(
        model, model_args.path_to_save_model, dataset, training_args, privacy_args, model_args, data_args,
        cuda=True, local_rank=local_rank, world_size=world_size
    )

    # Only on rank 0 save
    if local_rank == 0 and privacy_engine is not None and model is not None:
        priv_ckpt_path = model_args.path_to_save_model + "_pvt"
        privacy_engine.save_checkpoint(path=priv_ckpt_path, module=model, optimizer=None)
        try:
            model.save_pretrained(model_args.path_to_save_model)
        except Exception:
            try:
                model._module.save_pretrained(model_args.path_to_save_model)
            except Exception:
                model._module.module.save_pretrained(model_args.path_to_save_model)

if __name__ == "__main__":
    main()