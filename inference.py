import argparse
import torch
#from args import ModelArguments, DataArguments
from data_utils import ALL_DATASETS
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from peft import PeftModel, PeftConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, choices=list(ALL_DATASETS.keys()), help="Name of the dataset to use")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="Path to LoRA weights directory or checkpoint")
    parser.add_argument("--path_to_dataset", type=str, required=True, help="Path to the test file containing prompts")
    parser.add_argument("--output_file", type=str, default="results.csv")
    parser.add_argument("--sequence_len", type=int, default=1025, help="Maximum sequence length for generation")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dry_test_run", action="store_true", help="Run a dry test run with few samples only.")
    parser.add_argument("--eval_batch_size", type=int, default=2)

    parser.add_argument("--max_length", type=int, default=128, help="Max total generated sequence length")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width (1 = disable) ")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling ; default beam search")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus (top-p) sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--num_return_seq", type=int, default=1, help="Number of generated completions per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--min_new_tokens", type=int, default=1, help="Minimum number of new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    
    return parser.parse_args()



def load_model_and_tokenizer(model_name, lora_weights_path, device="cuda"):
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Applying LoRA from: {lora_weights_path}")
    model.load_adapter(lora_weights_path)
    model.eval()
    model.to(device)
    return model, tokenizer

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.lora_weights_path, device=device)
    print(f"Loaded model: {args.model_name} with LoRA weights from {args.lora_weights_path}")

    print(f"Reading test file: {args.path_to_dataset}")
    dataset = ALL_DATASETS[args.dataset_name](args, tokenizer)

    output_df = dataset.compute_test_metrics(model, tokenizer, args)
    print(f"Saving results to: {args.output_file}")
    output_df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
    