import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datasets import Dataset, DatasetDict, load_metric
import argparse
from datasets import load_metric
import evaluate
import numpy as np
import copy


import json

def compute_metrics(eval_pred):
    logits, labels = eval_pred


    # logits_tensor = torch.tensor(logits)

    # token_ids = torch.argmax(logits_tensor, dim=-1)
    token_ids = logits[0]
    label_ids = logits[1]

    # Ensure token_ids and label_ids are torch tensors
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids)
    if not isinstance(label_ids, torch.Tensor):
        label_ids = torch.tensor(label_ids)

    score = []
    exact_match_score = []
    mismatches = []
    for i in range(token_ids.shape[0]):
        relation_predict = []
        relation_answer = []
        for j in range(token_ids.shape[1]):
            if 128260 <= token_ids[i, j] <= 128734:
                relation_predict.append(token_ids[i, j].item() - 128260)
            if 128260 <= label_ids[i, j] <= 128734:
                relation_answer.append(label_ids[i, j].item() - 128260)

        # 한 개라도 겹치는 경우
        if set(relation_answer) & set(relation_predict):
            score.append(1)
        else:
            score.append(0)
            mismatches.append({"Answer": relation_answer, "Predict": relation_predict})

        # 정확히 포함하는 경우
        if set(relation_answer) == set(relation_predict):
            exact_match_score.append(1)
        else:
            exact_match_score.append(0)


    with open("wrong_ans.json", 'w') as f:
        json.dump(mismatches, f, indent=4)

    acc = np.mean(score)
    exact_match_acc = np.mean(exact_match_score)
        

    return {"In":acc, "Exact_Match": exact_match_acc}

def preprocess_logits_for_metrics(logits, labels):


    logits_tensor = torch.tensor(logits)
    pred_ids = torch.argmax(logits_tensor, dim=-1)
    return pred_ids, labels



def preprocess_dataset(data):
    dataset = {'query': [], 'paths': [], 'q_entity': []}
    for idx, val in enumerate(data):
        q = val[0] 
        paths = val[1]
        entities = val[2]
        for path in paths:
            if isinstance(path, str):
                path = [path]
            dataset['query'].append(q)
            dataset['paths'].append(path)
            dataset['q_entity'].append(', '.join(entities))
    return dataset

# def preprocess_logits_for_metrics(logits, labels):

#     # pred_ids = torch.argmax(logits, dim=-1)
#     # print(pred_ids)
#     # print(labels)
#     # exit(1)

#     pred_ids = torch.argmax(logits, dim=-1)
#     return pred_ids, labels

def preprocess_function(examples, tokenizer, args):


    if args.path_only:
        input_text = [f"Please generate a valid relation path that can be helpful for answering the following query: {query}" for query in examples['query']] # source
        formatted_path = [f"{input_text[idx]} "+f"Answer: <PATH>{'<SEP>'.join(f'<{p}>' for p in path)}</PATH>{tokenizer.eos_token}" for idx, path in enumerate(examples['paths'])] # source + target
    else:
        # for idx, patn in enumerate(examples['paths']):
        #     f"<PATH>{'<SEP>'.join(f'<{p}>' for p in path)}</PATH>"

        input_text = [f"Please generate a valid entity and relation path that can be helpful for answering the following query: {query}" for query in examples['query']] # source
        formatted_path = [f"{input_text[idx]} "+f"Answer: <ENT>{examples['q_entity'][idx]}</ENT>" + f"<PATH>{'<SEP>'.join(f'<{p}>' for p in path)}</PATH>" for idx, path in enumerate(examples['paths'])]
        # formatted_path = [f"Answer: <PATH>{'<SEP>'.join(f'<{p}>' for p in path)}</PATH>" for idx, path in enumerate(examples['paths'])]


    model_inputs = tokenizer(formatted_path, truncation=True, padding='max_length', max_length=128, pad_to_max_length = True)

    labels = tokenizer(formatted_path, truncation=True, padding='max_length', max_length=128, pad_to_max_length = True)
    # print(model_inputs)
    # print(labels)

    model_inputs["labels"] = labels["input_ids"]

    # labels['labels'] = labels["input_ids"]
    
    return model_inputs

def main(args):
    if args.path_only == True:
        print("path only running...")
    else:
        print("path and entity running...")
    # Prepare the training dataset
    query_path_set_processed = pd.read_parquet("data/rog_original_0007.parquet")
    # query_path_set_processed = query_path_set_processed.iloc[:20, :]


    train_data, test_data = train_test_split(list(zip(query_path_set_processed['query'], query_path_set_processed['paths'], query_path_set_processed['q_entity'])), test_size=0.2, random_state=123)

    train_data = preprocess_dataset(train_data)
    test_data = preprocess_dataset(test_data)

    train_data = Dataset.from_dict(train_data)
    test_data = Dataset.from_dict(test_data)

    # Make relations as special tokens
    relations = pd.read_pickle("data/id2rel.pkl")

    rel_set = set()
    for paths in relations.values():
            rel_set.add(f"<{paths}>")


    dataset = DatasetDict({
        'train': train_data,
        'test': test_data
    })

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    # Load the pre-trained LLaMA 3 model and tokenizer
    if args.model == "llama":
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        special_tokens = {'additional_special_tokens': ['<PATH>', '</PATH>', '<SEP>', '<ENT>', '</ENT>'] + list(rel_set)}
        tokenizer.add_special_tokens(special_tokens)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # 모델의 임베딩 레이어 크기 업데이트
        model.resize_token_embeddings(len(tokenizer))

        # print("Special Tokens Map: ", tokenizer.special_tokens_map)
        # print("All Special Tokens: ", list(tokenizer.all_special_tokens))
        # print("Special Token IDs: ", {token: tokenizer.convert_tokens_to_ids(token) for token in tokenizer.all_special_tokens})

    special_tokens_list = {tokenizer.convert_tokens_to_ids(token) for token in tokenizer.all_special_tokens}

    tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer, args), batched=True, num_proc=16)
    # tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=16)

    # columns_to_remove = ['query', 'paths', 'q_entity']
    columns_to_remove = ['paths', 'q_entity']

    # Remove the columns from each split in the dataset dictionary
    dataset_dict = tokenized_dataset.map(lambda examples: examples, remove_columns=columns_to_remove)


    # Print the updated dataset to verify
    print(dataset_dict)
    # exit(1)

    # Define the LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    if args.path_only: 
        output_directory = f"./lora_llama3/{args.save_path}"
    else:
        output_directory = f"./lora_llama3/{args.save_path}"
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_directory,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        save_only_model=True,
        eval_strategy="epoch",
        logging_dir=f"{output_directory}/logs", 
        num_train_epochs=20,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        # metric_for_best_model="accuracy",
        # eval_accumulation_steps=10
    )
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm = False)
    # Assuming dataset_dict["test"] is a Hugging Face Dataset with your original data
    queries = dataset_dict["test"]["query"]

    # Combine original data with queries
    data_with_queries = {**dataset_dict["test"].to_dict(), "query": queries}

    # Create a new Dataset including queries
    eval_dataset_with_queries = Dataset.from_dict(data_with_queries)


    # Create the Trainer and start fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        # eval_dataset=dataset_dict["test"],
        eval_dataset=eval_dataset_with_queries,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # compute_metrics=custom_compute_metrics,
        # compute_metrics=compute_metric_with_extra(tokenizer),
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        # preprocess_logits_for_metrics=lambda logits, labels: preprocess_logits_for_metrics(logits, labels, dataset_dict["test"]["query"]),
    )

    if args.checkpoint:

        checkpoint_path = "./lora_llama3/path_only/checkpoint-23000" 
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()
    trainer.save_model(f"./{args.save_model}")

    # Evaluate the fine-tuned model
    results = trainer.evaluate()

    # Print evaluation results
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainininining')
    parser.add_argument('--model', type=str, default="llama")
    parser.add_argument('--path_only', action='store_true', help='If set, only use the path')
    parser.add_argument('--checkpoint', action='store_true', help='If set, use the checkpoint')
    parser.add_argument('--save_model', type=str, default="save_model")
    parser.add_argument('--save_path', type=str, default="save path")
    args = parser.parse_args()


    main(args)