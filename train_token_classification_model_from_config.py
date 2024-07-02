"""
Rough outline of v1 of a standard sequence tagging/token classification 
model finetuning script
"""
import argparse
import evaluate
import os
import sys
import yaml
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments)

from finetuning_utils.core_functions import parse_config

# ================================================================================
# === TODO: split this into separate .py file ===
# === need to handle e.g. imports and also the tokenizer being passed to align_labels ?

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# ================================================================================

# I think you should put all this in functions in main file so then call them from main()
# that way they dont run first when loading script

""" wnut = load_dataset("wnut_17")
label_list = wnut["train"].features[f"ner_tags"].feature.names
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval") """

# ================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TODO check where this appears, maybe in help?")

    parser.add_argument('-c', '--config', type=str, help='path to config yaml file to load')
    parser.add_argument('--debug_mode', action=argparse.BooleanOptionalAction)
    #parser.add_argument('--names', nargs='+', type=str, help='space separated names of ur friends lol')
    #parser.add_argument('--datasets', nargs='+', type=str, help='space separated names of datasets to use in experiment')

    args = parser.parse_args()

    if args.debug_mode:
        print("RUNNING IN DEBUG MODE")
    else:
        print("RUNNING IN NOT DEBUG MODE xD")

    if not args.config:
        print("ERROR: need a config YAML file to proceed with training or inference!")
        print("- exit -")
        sys.exit(1) # check how you are supposed to exit with error O_o

    # load the yaml file
    with open(f"{args.config}.yaml", 'r') as fo:
        dd = yaml.safe_load(fo)
        model_cfg = parse_config(dd)
    #print(model_cfg)
    
    # TODO: validate schema of model_cfg here
    
    # build the id2label and label2id stuff
    if model_cfg.labeling_scheme.tagging_format == "IOB2":
        labels_with_scheme = ['O'] 
        for label_name in model_cfg.labeling_scheme.label_names:
            for prefix in ['B', 'I']:
                labels_with_scheme.append(f'{prefix}-{label_name}')
        id2label = dict(enumerate(labels_with_scheme))
        label2id = {v:k for k,v in id2label.items()}
        print(id2label)
        print("====")
    else:
        print("bad tag format xd")
        sys.exit(1)
    
    # load model
    # TODO: change from_pretrained if finetune local model? it's load from disk i think
    model = AutoModelForTokenClassification.from_pretrained(
        model_cfg.model_info.model_checkpoint,
        num_labels=len(labels_with_scheme),
        id2label=id2label,
        label2id=label2id
        )
    
    # TODO: GET TypeError: transformers.training_args.TrainingArguments() argument after ** must be a mapping, not types.SimpleNamespace
    # SO TRY CONVERT BACK TO DICT - might need to use a class instead of SimpleNamespace
    print(model_cfg.training_args)
    tmp_training_args_d = vars(model_cfg.training_args)
    training_args = TrainingArguments(**tmp_training_args_d)

    # =====================
    # TODO improve this part - refactor
    wnut = load_dataset(model_cfg.dataset_info.name)
    label_list = wnut["train"].features[f"ner_tags"].feature.names
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_info.model_checkpoint)
    tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")
    # =====================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_wnut["train"],
        eval_dataset=tokenized_wnut["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if not args.debug_mode:
        trainer.train()