import datasets
import evaluate
import torch
import numpy as np
import pandas as pd
import random
import os, re, ast, json
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
#TODO: Format the control code part of this

# Modified from https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
def main_preprocess_function(examples, tokenizer, text_field, prompt_begin, prompt_end, label_field, sequence_len, single_token=True):
    """
    Preprocess function for the main task of prompt tuning.
    This function will prepare the input for the model.
    """
    batch_size = len(examples[text_field])

    # Prepare the context with the text in between of prompts, e.g. "Sentence : <text> Label :"
    inputs = [prompt_begin + str(x) + prompt_end for x in examples[text_field]]

    # Prepare the prediction part
    targets = [str(x) for x in examples[label_field]]

    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)

    # Concatenate the context and prediction parts as one input and set -100 to the labels of the context part
    # This is because only the label part will be used to calculate the loss
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]

        # Check if model adds a <s> token (often id = tokenizer.bos_token_id)
        first_token_is_special = (
            hasattr(tokenizer, "bos_token_id") and label_input_ids[0] == tokenizer.bos_token_id
        )
        if single_token:
            # Tokenizer adds <s> to input_ids so just take the last id
            # NOTE THAT THIS ASSUMES THE LABEL IS SINGLE TOKEN
            label_input_ids = [label_input_ids[-1]]
        else:
            # Tokenizer adds <s> to input_ids so just take the rest
            #label_input_ids = labels["input_ids"][i][1:]
            label_input_ids = label_input_ids[1:] if first_token_is_special else label_input_ids
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    
    #lst = np.array(model_inputs["input_ids"][0].tolist())
    #print(labels["input_ids"][0].tolist())
    #lst = np.array(labels["input_ids"][0])
    #print(lst[0])
    #print(tokenizer.decode(np.where(lst != -100, lst, tokenizer.pad_token_id)))
    
    # Pad the samples with sequence_len and trim if longer than sequence_len
    # NOTE THAT IF CONTEXT IS LONGER THAN SEQUENCE_LEN, THERE WILL BE NOTHING TO PREDICT, LABEL IS ALL -100
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            sequence_len - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (sequence_len - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (sequence_len - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:sequence_len])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:sequence_len])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:sequence_len])

    model_inputs["labels"] = labels["input_ids"]
    #lst = np.array(model_inputs["input_ids"][0].tolist())
    #print(tokenizer.decode(np.where(lst != -100, lst, tokenizer.pad_token_id), skip_special_tokens=True))
    #lst = np.array(labels["input_ids"][0].tolist())
    #print(tokenizer.decode(np.where(lst != -100, lst, tokenizer.pad_token_id), skip_special_tokens=True))
    
    return model_inputs


class CustomDataset:
    dataset = None
    classes = None # List of class labels
    text_field = None # Name of the field in the dataset that contains the text
    prompt_begin = None # Prompt to add to the beginning of the text, e.g. "Sentence : "
    prompt_end = None # Prompt to add to the end of the text, e.g. " Label :"
    label_field = None # Name of the field in the dataset that contains the label
    evaluate = None # Evaluation metric

    def __init__(self, tokenizer, sequence_len):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len

    def target_max_len(self):
        target_lens = [len(self.tokenizer(class_label)["input_ids"]) for class_label in self.classes]
        target_max_len = max(target_lens)
        return target_max_len

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        Original Trainer may lead to a memory issue.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids
    
    def preprocess_function(self, example):
        return main_preprocess_function(example, self.tokenizer, self.text_field, self.prompt_begin,
                                         self.prompt_end, self.label_field, self.sequence_len, single_token=False)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # Only keep predictions up to last token
        predictions = predictions[..., :-1]
        # Only keep labels from the first token
        labels = labels[..., 1:]
        # Replace -100 of the labels as we don't want the content
        predictions = np.where(labels != -100, predictions, self.tokenizer.pad_token_id)
        # Decode generated summaries into text
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Compute ROUGE scores
        result = self.evaluate.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        return {k: round(v, 4) for k, v in result.items()}

    def compute_test_metrics(self, model, tokenizer, args):
        
        print(f"Testing for the entire dataset. Number of generations  per prompt: {args.num_return_seq}")
        if (self.path_to_test_dataset is None):
            test_dataset = self.dataset['test']
        else:
            print("Loading custom test dataset...")
            df = pd.read_csv(self.path_to_test_dataset)
            df = df[df[self.text_field].notna()]
            test_dataset = Dataset.from_pandas(df)
        if(args.dry_test_run):
            print("Test run...")
            test_dataset = test_dataset.select(range(5))
            #num_return_seq = 2
            

        # Need to use this only if your text_field is 'label'
        if(self.text_field == "label"):
            test_dataset = test_dataset.map(lambda x: {self.text_field: test_dataset.features[self.text_field].int2str(x[self.text_field])})  
            new_features = test_dataset.features.copy()
            new_features[self.text_field] = datasets.Value("string")
            test_dataset = test_dataset.cast(new_features)
        
        # Ensuring that we retain attributes from other columns. Can add functionality later to accept this as an argument from the user.
        retain_columns = [col for col in test_dataset.column_names if col not in [self.label_field]]
        output_dataframe = {}
        for col in retain_columns:
            output_dataframe[col] = [element for element in test_dataset[col] for i in range(args.num_return_seq)]

        test_dataset = test_dataset.map(
            lambda x: {self.text_field: [self.prompt_begin + str(article) + self.prompt_end for article in x[self.text_field]]},
            batched=True,
            num_proc=None,
        )
        
        # Tokenize data
        def test_preprocess_function(examples):
            model_inputs = tokenizer(examples[self.text_field], padding=False)

            # 2. reserve the original article and summary for saving
            #model_inputs[self.label_field] = examples[self.label_field]
            return model_inputs

        with torch.no_grad():
            test_dataset = test_dataset.map(
                test_preprocess_function,
                batched=True, num_proc=None, desc="tokenizing dataset",
                remove_columns=test_dataset.column_names)

        # Filter out samples too long, e.g. more than 750 tokens
        #test_dataset = test_dataset.filter(lambda x: len(x['input_ids']) < 750)
        
        print("Length of test data", len(test_dataset))

        test_dataset.set_format(type="torch")

        def generate_batched(
            model,
            tokenizer,
            device,
            query_tensors,
            batch_size: int = 4,
            return_prompt: bool = True,
            pad_to_multiple_of: int = None,
        ):
            outputs = []

            tokenizer.padding_side = "left"

            # in case we have fewer examples than bs
            batch_size = min(len(query_tensors), batch_size)

            for i in range(0, len(query_tensors), batch_size):
                # prevent overflow if query tensors are not even multiple of bs
                end_index = min(len(query_tensors), i + batch_size)

                batch = query_tensors[i:end_index]
                batch_mask = [torch.ones_like(element) for element in batch]
                inputs = {"input_ids": batch, "attention_mask": batch_mask}

                padded_inputs = tokenizer.pad(
                    inputs,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    #generations = model.generate(**padded_inputs, **generation_kwargs)
                    if(args.temperature==0.0):
                        # Greedy decoding
                        print("Greedy decoding...")
                        generations = model.generate(**padded_inputs, do_sample=False, min_new_tokens = args.min_new_tokens, max_new_tokens = args.max_new_tokens, num_return_sequences=args.num_return_seq, num_beams = 5, eos_token_id=tokenizer.eos_token_id, bad_words_ids = [[1, 4768, 5275]], repetition_penalty = args.repetition_penalty)
                    else:
                        generations = model.generate(**padded_inputs, do_sample=True, min_new_tokens = args.min_new_tokens, max_new_tokens = args.max_new_tokens, top_k = args.top_k, top_p = args.top_p, temperature = args.temperature, num_return_sequences=args.num_return_seq, eos_token_id=tokenizer.eos_token_id, bad_words_ids = [[1, 4768, 5275]], repetition_penalty = args.repetition_penalty)
                ind = 0
                for mask in padded_inputs["attention_mask"]:
                    for ind_item in range(ind, ind+args.num_return_seq):
                        output = generations[ind_item][(1 - mask).sum() :]  # remove padding

                        if not return_prompt:
                            output = output[(mask).sum() :]  # remove prompt
                        outputs.append(output)
                    ind+=args.num_return_seq
            return outputs

        if hasattr(model, "generate"):
            model = model
        # The following is for GradSampleModule wrapping
        elif hasattr(model._module, "generate"):
            model = model._module
        # The following is for GradSampleModule and DPDDP wrapping
        elif hasattr(model._module.module, "generate"):
            model = model._module.module
        else:
            raise ValueError("Cannot find generate function in the model.")

        model.eval()

        response_tensors = generate_batched(
            model, tokenizer, args.device,
            test_dataset["input_ids"],
            batch_size=args.eval_batch_size, return_prompt=False,
        )
        responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                     for r in response_tensors]
        input_data = [tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                      for r in test_dataset["input_ids"] for rep in range(args.num_return_seq)] #TODO: Return num_sequences in place of 3 in the range
        output_dataframe['input_prompt'], output_dataframe['output_text'] = input_data, responses

        df = pd.DataFrame(output_dataframe)
        return df
    
class GenericCustomDataset(CustomDataset):
    """
    Generic dataset class configurable for different dataset sources/fields.
    """
    def __init__(self, args, tokenizer, config):

        self.name = config.get("name")
        self.path_to_dataset = getattr(args, "path_to_dataset", None) #args.path_to_dataset
        
        # === Load dataset ===
        load_type = config.get("load_type", "from_disk")
        dataset_name = config.get("dataset_name")
        
        if load_type == "from_disk":
            self.dataset = load_from_disk(self.path_to_dataset)
        elif load_type == "from_hf":
            self.dataset = load_dataset(dataset_name)
        else:
            raise ValueError(f"Unknown load_type: {load_type}")

        # === Optional: test dataset loading ===
        test_path = getattr(args, "path_to_test_dataset", None)
        if test_path:
            try:
                self.dataset['test'] = load_from_disk(test_path)
            except Exception as e:
                if hasattr(self, "create_test_dataset"):
                    self.dataset['test'] = self.create_test_dataset(test_path)
                else:
                    print(f"Could not load or create test set: {e}")

        # === Field/Prompt configs ===
        self.control_field = config.get("control_field", None)
        self.text_field = config.get("text_field", self.control_field)
        self.label_field = config.get("label_field", None)
        self.prompt_begin = config.get("prompt_begin", "")
        self.prompt_end = config.get("prompt_end", "")
        self.evaluate = evaluate.load("rouge")
        
        # === Optional: any per-dataset post-processing ===
        if config.get("max_label_words"):
            max_words = config["max_label_words"]
            f = lambda x: len(x[self.label_field].split()) <= max_words and len(x[self.label_field]) != 0
            self.dataset = self.dataset.filter(f)
        
        # Filter out empty label_field
        self.dataset = self.dataset.filter(lambda x: x[self.label_field] is not None and len(x[self.label_field]) > 0)

        # === Any custom data sample limiting ===
        if config.get('n_train'):
            self.dataset['train'] = self.dataset['train'].select(range(config['n_train']))
        if config.get('n_test'):
            self.dataset['test'] = self.dataset['test'].select(range(config['n_test']))

        # === Any custom mapping ===
        map_fn = config.get("map_fn")
        if map_fn:
            self.dataset = self.dataset.map(map_fn)
        
        # === Save other args ===
        self.path_to_test_dataset = test_path
        # Convert to datasetdict if not already
        if not isinstance(self.dataset, DatasetDict):
            self.dataset = DatasetDict({"train": self.dataset})
        
        super().__init__(tokenizer, args.sequence_len)

    # Example (can override in subclass if needed)
    def create_test_dataset(self, path_to_dataset):
        output_dir = path_to_dataset.split(".csv")[0]
        os.makedirs(output_dir, exist_ok=True)
        df_test = pd.read_csv(path_to_dataset)
        dataset = DatasetDict()
        dataset['test'] = Dataset.from_pandas(df_test)
        dataset.save_to_disk(output_dir)
        return dataset['test']

ALL_DATASET_CONFIGS = {
    "hfhub": {
        "name": "hfhub",
        "load_type": "from_hf_path",
        "control_field": None,  # Set dynamically
        "label_field": None,    # Set dynamically
        "prompt_begin": None,   # Set dynamically
        "prompt_end": None,     # Set dynamically
    },
    "tab": {
        "name": "tab",
        "load_type": "from_disk",
        "control_field": "control",
        "text_field": "control",
        "label_field": "text",
        "prompt_begin": "",
        "prompt_end": "\n"
    },
    "mimic": {
        "name": "mimic",
        "load_type": "from_disk",
        "control_field": "ICD9_CODE",
        "text_field": "LONG_TITLE",
        "label_field": "TEXT",
        "prompt_begin": "Diagnosis: ",
        "prompt_end": " Summary :",
        # If needed, add a 'create_dataset_fn' for dataset creation as in old MIMIC class
    },
    "wiki": {
        "name": "wiki",
        "load_type": "from_disk",
        "control_field": "Name",
        "text_field": "Name",
        "label_field": "Text",
        "prompt_begin": "Name : ",
        "prompt_end": "\nBiography:",
        # To imitate "self.dataset['test'] = self.dataset['train']", set in loader if needed
    },
    "wiki-instruct": {
        "name": "wiki-instruct",
        "load_type": "from_disk",
        "control_field": "Name",
        "text_field": "Name",
        "label_field": "Text",
        "prompt_begin": "You are a helpful assistant. Write a response that appropriately completes the request.\n\n### Input:\nGenerate a biography about ",
        "prompt_end": "\n\n### Response:",
        # If train==test, set in loader
    },
    "med-wiki": {
        "name": "med-wiki",
        "load_type": "from_hf",
        "dataset_name": "gamino/wiki_medical_terms",
        "control_field": "page_title",
        "text_field": "page_title",
        "label_field": "page_text",
        "prompt_begin": "Medical Term: ",
        "prompt_end": "\nDescription: ",
        "max_label_words": 1024
    },
    "med-wiki-instruct": {
        "name": "med-wiki-instruct",
        "load_type": "from_hf",
        "dataset_name": "gamino/wiki_medical_terms",
        "control_field": "page_title",
        "text_field": "page_title",
        "label_field": "page_text",
        "prompt_begin": "You are a helpful assistant. Write a response that appropriately completes the request.\n\n### Input:\nGenerate a description of this medical term: ",
        "prompt_end": "\n\n### Response:",
        "max_label_words": 1024,
        "n_test": 500  # Optional: Select first 500 in train if no custom test is provided
    },
    "med-wiki-instruct-baseline": {
        "name": "med-wiki-instruct-baseline",
        "load_type": "from_hf",
        "dataset_name": "gamino/wiki_medical_terms",
        "control_field": "page_title",
        "text_field": "page_title",
        "label_field": "page_text",
        "prompt_begin": "You are a helpful assistant. Write a response that appropriately completes the request.\n\n### Input:\nGenerate a description of this medical term: ",
        "prompt_end": "\n\n### Response:",
        "max_label_words": 1024,
        "n_train": 100,  # Random subset for training
        "n_test": 500    # Optional: Select first 500 of train if no test
    },
    "med-abst-instruct": {
        "name": "med-abst-instruct",
        "load_type": "from_hf",
        "dataset_name": "TimSchopf/medical_abstracts",
        "control_field": "condition_label",
        "text_field": "condition_label",
        "label_field": "medical_abstract",
        "prompt_begin": "You are a helpful assistant. Write a response that appropriately completes the request.\n\n### Input:\nGenerate a text for the following medical condition: ",
        "prompt_end": "\n\n### Response:",
        # You'll need actual mapping code for the label, here's a placeholder:
        #"map_fn": lambda x: {"condition_label": [dict_mapping[x["condition_label"]]]},
        "dict_mapping": {
            1: "Neoplasms", 2: "Digestive system diseases", 3: "Nervous system diseases", 
            4: "Cardiovascular diseases", 5: "General pathological conditions"
        }
    },
    "med-summ": {
        "name": "med-summ",
        "load_type": "from_disk",
        "control_field": "prompt",
        "text_field": "prompt",
        "label_field": "answer",
        "prompt_begin": "You are a helpful assistant. Write a response that appropriately completes the request.\n\n### Input:\n",
        "prompt_end": "\n\n### Response:",
    },
    "clinical-notes": {
        "name": "clinical-notes",
        "load_type": "from_disk",
        "control_field": "Condition",
        "text_field": "Condition",
        "label_field": "full_note",
        "prompt_begin": "You are a helpful assistant. Write a response that appropriately completes the request.\n\n### Input:\nGenerate a SOAP note for the following medical condition: ",
        "prompt_end": "\n\n### Response:",
        # Usually uses all train for test unless test set provided
    },
    "wiki-science": {
        "name": "wiki-science",
        "load_type": "from_disk",
        "control_field": "title",
        "text_field": "title",
        "label_field": "content",
        "prompt_begin": "",
        "prompt_end": "",
    },
    "common-pile-news": {
        "name": "common-pile-news",
        "load_type": "from_disk",
        "control_field": "headline",
        "text_field": "headline",
        "label_field": "text",
        "prompt_begin": "",
        "prompt_end": "",
        # 'filter' code to drop empty text needed if using this class
    },
    "wiki-ai": {
        "name": "wiki-ai",
        "load_type": "from_disk",
        "control_field": "title",
        "text_field": "title",
        "label_field": "content",
        "prompt_begin": "",
        "prompt_end": "",
    },
    "wikipedia-large": {
        "name": "wikipedia-large",
        "load_type": "from_disk",
        "control_field": "title",
        "text_field": "title",
        "label_field": "content",
        "prompt_begin": "",
        "prompt_end": "",
    },
    "wiki-merged": {
        "name": "wiki-merged",
        "load_type": "from_disk",
        "control_field": "title",
        "text_field": "title",
        "label_field": "content",
        "prompt_begin": "",
        "prompt_end": "",
    },
    "wiki-merged-split-150": {
        "name": "wiki-merged-split-150",
        "load_type": "from_disk",
        "control_field": "title",
        "text_field": "title",
        "label_field": "content",
        "prompt_begin": "",
        "prompt_end": "",
    },
    "wiki-merged-split-200": {
        "name": "wiki-merged-split-200",
        "load_type": "from_disk",
        "control_field": "title",
        "text_field": "title",
        "label_field": "content",
        "prompt_begin": "",
        "prompt_end": "",
    },
}

ALL_DATASETS = {
    name: (lambda args, tokenizer, config=config: GenericCustomDataset(args, tokenizer, config))
    for name, config in ALL_DATASET_CONFIGS.items()
}