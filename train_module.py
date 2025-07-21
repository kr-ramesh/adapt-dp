import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine
from opacus.utils.adaptive_clipping.adaptive_clipping_utils import PrivacyEngineAdaptiveClipping
from opacus.validators import ModuleValidator
from tqdm import tqdm
import gc
# Integrate weights and biases 
import wandb


def to_dict(obj):
    if hasattr(obj, '__dict__'):
        return vars(obj)
    elif hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return asdict(obj)
    elif isinstance(obj, dict):
        return obj
    else:
        raise TypeError(f"Don't know how to convert {type(obj)} to dict")


def default_collate(batch):
    # Ensure each element is a tensor, else convert
    keys = batch[0].keys()
    return {k: torch.stack([torch.as_tensor(sample[k]) for sample in batch]) for k in keys}

def get_dataloader(dataset, batch_size, num_replicas=None, rank=None, shuffle=True, collate_fn=default_collate):
    """
    Returns a DataLoader with optional DistributedSampler, and a robust collate_fn.
    Collate function must produce tensors with shape [batch, ...] for each key.
    """
    sampler = None
    if num_replicas is not None and rank is not None:
        sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
    # shuffle only if not using sampler
    do_shuffle = shuffle if sampler is None else False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=do_shuffle,
        collate_fn=collate_fn,  # use default_collate unless you specify
        drop_last=True,
        pin_memory=True,  # speeds up GPU transfers
        num_workers=4     # or your desired degree of parallelism
    )

def train(
    model, path_to_save_model,
    dataset,
    train_args,
    opacus_config,
    model_args,
    data_args,
    cuda=True,
    local_rank=0,
    world_size=1,
):
    #TODO: Change this
    # DDP device and rank
    if cuda:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")
    is_main = (local_rank == 0)

    # Model validation for DP
    # LoRA fine-tuning 
    lora_config = LoraConfig(
                r=train_args.lora_r,
                lora_alpha=train_args.lora_alpha,
                target_modules= ['q_proj', 'v_proj'],#["c_attn"],  # 'c_attn' is a GPT2 example; adjust for your model
                lora_dropout=0,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
    print(f"Using LoRA config: {lora_config}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # (optional debugging info)
    model = ModuleValidator.fix(model)

    # Wrap in DDP asap AFTER moving to CUDA
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    #ds = dataset_class(train_args, tokenizer, train_args.get("ds_config", {}))
    train_set = dataset.dataset["train"]
    train_args.data_size = len(train_set)
    train_set.set_format(type='torch')
    # Assign 50 samples from the train_set for validation
    val_set = train_set.select(range(50))
    #val_set = dataset.dataset["validation"] if "validation" in dataset.dataset else None

    # 2. DataLoader with DDPSampler
    train_loader = get_dataloader(train_set, batch_size=train_args.batch_size,
                                 num_replicas=world_size if world_size > 1 else None,
                                 rank=local_rank if world_size > 1 else None)
    val_loader = get_dataloader(val_set, batch_size=len(val_set),
                               num_replicas=world_size if world_size > 1 else None,
                               rank=local_rank if world_size > 1 else None) if val_set else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.lr)
    
    #setup_wandb(train_args, opacus_config, project_name=train_args.project_name, run_name=f"{model_args.model_name}_run")
    config = {}
    config.update({f"train_{k}": v for k, v in to_dict(train_args).items()})
    config.update({f"model_{k}": v for k, v in to_dict(model_args).items()})
    config.update({f"opacus_{k}": v for k, v in to_dict(opacus_config).items()})
    config.update({f"data_{k}": v for k, v in to_dict(data_args).items()})

    run_name = f"{model_args.model_name}_dataset_{data_args.dataset_name}_seq_len_{model_args.sequence_len}_lr_{train_args.lr}_batch_size_{train_args.batch_size}_gas_{train_args.gradient_accumulation_steps}_noise_multiplier_{opacus_config.noise_multiplier}_lora_{train_args.lora_r}_alpha_{train_args.lora_alpha}"
    #wandb.init(project=train_args.project_name, name=run_name, config=config)

    model.train()
    privacy_engine = PrivacyEngine()
    if opacus_config.target_epsilon is not None:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=train_args['epochs'],
            target_epsilon=opacus_config.target_epsilon,
            target_delta=opacus_config.target_delta,
            max_grad_norm=opacus_config.max_grad_norm,
            clipping=opacus_config.clipping
        )
        if is_main:
            print(f"OPACUS: using target_epsilon={opacus_config.target_epsilon} (noise_multiplier found automatically).")
    elif opacus_config.noise_multiplier is not None:
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=opacus_config.noise_multiplier,
            max_grad_norm=opacus_config.max_grad_norm,
            target_delta=opacus_config.target_delta,
            clipping=opacus_config.clipping,
            clipbound_learning_rate=opacus_config.clipbound_learning_rate,
            target_unclipped_quantile=opacus_config.target_unclipped_quantile,
            min_clipbound=opacus_config.min_clipbound,
            max_clipbound=opacus_config.max_clipbound,
            unclipped_num_std=0.1
        )
        if is_main:
            print(f"OPACUS: using noise_multiplier={opacus_config.noise_multiplier} (fixed; epsilon tracked).")
    else:
        raise ValueError("You must specify either target_epsilon or noise_multiplier for DP training.")

    optimizer.attach_step_hook(
            privacy_engine.accountant.get_optimizer_hook_fn(sample_rate=1/len(train_loader)* train_args.gradient_accumulation_steps)
        )
    
    print(train_args)
    print(f"Learning rate: {train_args.lr}, Noise multiplier: {opacus_config.noise_multiplier}, Clipping norm: {opacus_config.max_grad_norm}")

    # -- OPACUS Privacy Setup (see previous answer for epsilon/noise decision & clipping) --
    # Validate the model
    if not ModuleValidator.validate(model, strict=True):
        print("Model validated.")
    else:
        print("Model validation failed. Please check your model configuration.")
    
    if is_main:
        print("Starting training ...")
    
    model.train()

    # Printing privacy parameters
    print(f"Privacy parameters: noise_multiplier={opacus_config.noise_multiplier}, target_delta={opacus_config.target_delta}, clipping={opacus_config.clipping}")
    for epoch in range(train_args.epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        #losses = 0.0
        losses = []
        optimizer.zero_grad()
        with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=4, optimizer=optimizer) as new_data_loader:
            for step, batch in enumerate(tqdm(new_data_loader, desc=f"Epoch {epoch+1}/{train_args.epochs}", disable=not is_main)):
                for k in batch:
                    batch[k] = batch[k].to(device)
                # Check if batch is empty
                if len(batch['input_ids']) == 0:
                    #print("Skipping empty batch at step", step)
                    continue
            
                outputs = model(**batch)
                loss = outputs.loss
                #loss = loss / gradient_accumulation_steps
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
                
                #if (step+1) % 10 == 0:  # Print every 10 steps
                #    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                #    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

                #losses = losses + loss.item() #* gradient_accumulation_steps)  # Save the original loss (not normalized)
                losses.append(loss.item())  # Save the original loss (not normalized)
        
        train_loss = sum(losses) / len(losses)
        # Adaptive clipping: safest method to get current clip bound
        if hasattr(optimizer, "max_grad_norm") and hasattr(optimizer, "noise_multiplier"):
            mgn, nm = optimizer.max_grad_norm, optimizer.noise_multiplier
        
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'max_grad_norm': mgn,
            'noise_multiplier': nm
        }

        if is_main:
            #wandb.log(metrics)
            print(f"Epoch {epoch+1} Train loss: {sum(losses) / len(losses):.4f}")
        
        
        if val_loader and is_main:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    for k in batch:
                        batch[k] = batch[k].to(device)
                    outputs = model(**batch)
                    loss = outputs.loss
                    val_losses.append(loss.item())
            val_loss = sum(val_losses)/len(val_losses)
            print(f"Epoch {epoch+1} Val loss: {val_loss:.4f}")
            #wandb.log({"epoch": epoch+1, "val_loss": val_loss})
            model.train()

        # Save the checkpoint every 5 epochs
        if(epoch + 1) % 5 == 0 and is_main:
            checkpoint_path = f"{path_to_save_model}/ckpt_epoch_{epoch+1}"
            try:
                model.save_pretrained(checkpoint_path)
            except Exception:
                try:
                    model._module.save_pretrained(checkpoint_path)
                except Exception:
                    model._module.module.save_pretrained(checkpoint_path)
            #model_to_save = model.module if hasattr(model, 'module') else model
            #model_to_save.save_pretrained(checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    # Return the privacy spent
    #print("Epsilon spent:", privacy_engine.get_epsilon(opacus_config.target_delta))
    # Note: Only MAIN rank returns privacy engine (for saving model)
    return model, privacy_engine if is_main else (None, None)