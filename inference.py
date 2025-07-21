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
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test file containing prompts")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="Path to LoRA weights directory or checkpoint")
    parser.add_argument("--output_file", type=str, default="results.csv")
    parser.add_argument("--sequence_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--prompt_begin", type=str, default="")
    parser.add_argument("--prompt_end", type=str, default="")

    parser.add_argument("--max_length", type=int, default=128, help="Max total generated sequence length")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width (1 = disable) ")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling ; default beam search")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus (top-p) sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of generated completions per prompt")
    
    return parser.parse_args()



def load_model_and_tokenizer(model_name, lora_weights_path, device="cuda"):
    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Applying LoRA from: {lora_weights_path}")
    peft_model = PeftModel.from_pretrained(base_model, lora_weights_path)
    peft_model = peft_model.merge_and_unload()
    peft_model.eval()
    peft_model.to(device)
    return peft_model, tokenizer

def batchify(inputs, batch_size):
    for i in range(0, len(inputs), batch_size):
        yield inputs[i:i+batch_size]

@torch.no_grad()
def run_inference(
    model,
    tokenizer,
    texts,
    prompt_begin="",
    prompt_end="",
    sequence_len=128,
    batch_size=2,
    device="cuda",
    generation_kwargs=None
):
    results = []
    for batch in batchify(texts, batch_size):
        prompts = [prompt_begin + text + prompt_end for text in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=sequence_len)
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=generation_kwargs.get("max_length", sequence_len),
            num_beams=generation_kwargs.get("num_beams", 1),
            do_sample=generation_kwargs.get("do_sample", False),
            top_k=generation_kwargs.get("top_k", 50),
            top_p=generation_kwargs.get("top_p", 1.0),
            temperature=generation_kwargs.get("temperature", 1.0),
            repetition_penalty=generation_kwargs.get("repetition_penalty", 1.0),
            num_return_sequences=generation_kwargs.get("num_return_sequences", 1),
            pad_token_id=tokenizer.eos_token_id
        )
        # output_ids shape: [batch_size * num_return_sequences, seq_len]
        generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        results.extend(generated_texts)
    return results

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Reading test file: {args.test_file}")

    dataset = ALL_DATASETS[data_args.dataset_name](data_args, tokenizer)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.lora_weights_path, device=device)

    generation_kwargs = dict(
        max_length=args.max_length,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_return_sequences,
    )

    # Run inference
    predictions = run_inference(
        model,
        tokenizer,
        texts,
        prompt_begin=args.prompt_begin,
        prompt_end=args.prompt_end,
        sequence_len=args.sequence_len,
        batch_size=args.batch_size,
        device=device,
        generation_kwargs=generation_kwargs
    )

    # If num_return_sequences > 1, reshape results for CSV output
    if args.num_return_sequences > 1:
        # Expand each row to multiple rows
        expanded = []
        for idx, row in df.iterrows():
            for i in range(args.num_return_sequences):
                base = row.copy()
                base["generated"] = predictions[idx * args.num_return_sequences + i]
                base["sample_id"] = i
                expanded.append(base)
        df_out = pd.DataFrame(expanded)
    else:
        df["generated"] = predictions
        df_out = df

    print(f"Saving results to {args.output_file}")
    df_out.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()