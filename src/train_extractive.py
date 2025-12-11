import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import Dataset
from collections import defaultdict
import numpy as np
from torch.utils.data import DataLoader
from evaluate import load
from transformers import default_data_collator
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import get_scheduler
import os
import yaml



default_config = dict(
    # modeling / tokenization
    model_name="deepset/roberta-base-squad2",
    max_length=512,
    stride=128,
    n_best=25,
    max_answer_length=30,

    # training hyperparams
    batch_size=8,
    eval_batch_size=8,
    epochs=10,
    learning_rate=1e-6,

    # dataset / filtering
    spoiler_type="phrase",

    # misc
    save_dir="./models_task2",
    mixed_precision="fp16",
)

config_path = "./configs/extractive_config.yaml"

if os.path.exists(config_path):
    try:
        with open(config_path, "r") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        # merge: user keys override defaults
        config = {**default_config, **user_cfg}
        config["learning_rate"] = float(config["learning_rate"])
        config["max_length"] = int(config["max_length"])
        config["stride"] = int(config["stride"])
        config["n_best"] = int(config["n_best"])
        config["max_answer_length"] = int(config["max_answer_length"])
        config["batch_size"] = int(config["batch_size"])
        config["eval_batch_size"] = int(config["eval_batch_size"])
        config["epochs"] = int(config["epochs"])
        print(f"[config] Loaded config from {config_path}")
    except Exception as e:
        print(f"[config] Failed to load {config_path}: {e}. Using defaults.")
        config = default_config
else:
    print(f"[config] No {config_path} found. Using default config.")
    config = default_config

if "models" in config:
    model_list = config["models"]
else:
    model_list = [config["model_name"]]

def convert2squadFormat(df):
    df_fin = df[['uuid','targetTitle','postText',"mergedParas","tokPos","spoiler"]]
    # Create answer field, skipping rows where tokPos is empty
    def get_answer(x):
        if len(x['tokPos']) > 0:
            return {'text': x['spoiler'], "answer_start": [x['tokPos'][0][0]]}
        else:
            return {'text': x['spoiler'], "answer_start": [0]}
    
    df_fin["asnwers"] = df_fin.apply(get_answer, axis=1)
    df_fin = df_fin.drop(columns=["tokPos","spoiler"])
    df_fin.columns = ["id","title","question","context","answers"]
    
    return df_fin

def preprocess_training_examples(examples):
    questions = examples['question']
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples):
    questions = examples["question"]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

def compute_metrics(start_logits, end_logits, features, examples, predictOnly=False):
    example_to_features = defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_spoilers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        spoilers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    spoilers.append(answer)

        # Select the answer with the best score
        if len(spoilers) > 0:
            best_answer = max(spoilers, key=lambda x: x["logit_score"])
            predicted_spoilers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_spoilers.append({"id": example_id, "prediction_text": ""})
            
    predicted_texts = [i['prediction_text'] for i in predicted_spoilers]
    
    if predictOnly:
        return predicted_texts
    
    actual_spoilers_squad = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    actual_spoilers = [i['answers']['text'][0] for i in actual_spoilers_squad]    
    
    squad_metrics_eval = squad_metric.compute(predictions=predicted_spoilers, references=actual_spoilers_squad)
    bleu_eval = bleu.compute(predictions=predicted_texts, references=actual_spoilers)
    
    return [squad_metrics_eval,bleu_eval],actual_spoilers,predicted_texts

bleu = load("bleu")
squad_metric = load("squad")

errorUuid = {"ad9271b7-9983-42f5-9bd9-fdfcb171ddaa":[[[4, 37],[4, 222]]]}
def parse_spoiler(x):
    spoiler = []
    if x['uuid'] in errorUuid:
        x['spoilerPositions'] = errorUuid[x['uuid']]

    for s in x['spoilerPositions']:
        st,en = s[0],s[1]
        spoiler.append(x['targetParagraphs'][st[0]][st[1]:en[1]])
        
    return spoiler

def findPosTags(x):    
    tokPos = []
    for pos in x['spoilerPositions']:
 
        # Auto-fix flat or malformed formats
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            # try to convert [4,37] → [[4,37],[4,37]]
            if isinstance(pos, (list, tuple)) and len(pos) == 2 and all(isinstance(i,int) for i in pos):
                st = [pos[0], pos[0]]
                en = [pos[1], pos[1]]
            else:
                continue
        else:
            st, en = pos
 
        idx = 0
        for i,p in enumerate([x['targetTitle']] + x['targetParagraphs']):
            if i == st[0] + 1:
                start_ind = idx + st[1]
                end_ind = idx + en[1]
                tokPos.append([start_ind, end_ind])
                break
 
            if i == 0:
                idx += len(p) + 3
            else:
                idx += len(p) + 1
       
    return tokPos

def read_prep(path,train=True):
    with open(path, 'rb') as json_file:
        json_list = list(json_file)

    results = []
    for json_str in json_list:
        result = json.loads(json_str)
        results.append(result)
    df = pd.DataFrame(results)
    df['tags'] = df.tags.apply(lambda x:x[0],1)
    df['postText'] = df.postText.apply(lambda x:x[0],1)    
    
    # Parsing for faulty spoiler ids
    df['spoilerParsed'] = df.apply(parse_spoiler,1)
    df['mergedParas'] = df['targetParagraphs'].apply(lambda x:" ".join(x),1)
    df.mergedParas = df.targetTitle + " - " + df.mergedParas
    df['tokPos'] = df.apply(findPosTags,1)
    df['label'] = df['tags'].map({"phrase":0,"passage":1,"multi":2})
    
    return df

df_train = read_prep("./data/train.jsonl")
df_valid = read_prep("./data/validation.jsonl")

spoiler_type = config.get("spoiler_type", "phrase")
train_df = df_train[df_train.tags==spoiler_type]
val_df = df_valid[df_valid.tags==spoiler_type]

len(train_df),len(val_df)

train_df = convert2squadFormat(train_df)
val_df = convert2squadFormat(val_df)

train_data = Dataset.from_pandas(train_df.reset_index(drop=True), split="train")
val_data = Dataset.from_pandas(val_df.reset_index(drop=True), split="test")




# Para Generation
# config = dict(
# max_length = 512,
# stride = 128,
# n_best = 15,
# max_answer_length = 100,
# batch_size = 8,
# epochs = 20,
# learning_rate = 1e-6,
# model_name = model_name,
# spoiler_type = "passage"
# )

# Phrase Generation



for model_name in model_list:
    print(f"Training model: {model_name}")
    max_length = config["max_length"]
    stride = config["stride"]
    n_best = config["n_best"]
    max_answer_length = config["max_answer_length"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    train_dataset = train_data.map(
    preprocess_training_examples,
    batched=True,
    remove_columns=train_data.column_names,
)  
    validation_dataset = val_data.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=val_data.column_names,
)
    len(train_dataset), len(validation_dataset)
    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")
    train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=config["batch_size"],
)
    eval_dataloader = DataLoader(
    validation_set, collate_fn=default_data_collator, batch_size=8 
)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    accelerator = Accelerator(mixed_precision="fp16")
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
    num_train_epochs = config["epochs"]
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
    progress_bar = tqdm(range(num_training_steps))
    all_metrics = []
    max_bleu = 0.2
    save_dir = config.get("save_dir", "./models_task2")
    for epoch in range(num_train_epochs):
    # Training
     model.train()
     train_loss = 0
     for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        train_loss+=loss.item()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    

    # Evaluation
     model.eval()
     start_logits = []
     end_logits = []
     accelerator.print("Evaluation!")
    
     for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

     start_logits = np.concatenate(start_logits)
     end_logits = np.concatenate(end_logits)
     start_logits = start_logits[: len(validation_dataset)]
     end_logits = end_logits[: len(validation_dataset)]

     metrics,theoretical_texts,predicted_texts = compute_metrics(
        start_logits, end_logits, validation_dataset, val_data
    )
     all_metrics.append(metrics)
     bleu_score = metrics[1].get("bleu", 0.0)
     if bleu_score > max_bleu:
        model_save_path = os.path.join(save_dir, model_name.replace("/", "_"), f"{spoiler_type}")
        os.makedirs(model_save_path, exist_ok=True)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        max_bleu = bleu_score
        print(f"[save] new best bleu {max_bleu:.4f} -> saved to {model_save_path}")
     print(f"epoch {epoch}:", metrics)

