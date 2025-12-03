import os
from pathlib import Path
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import get_scheduler
from torch.optim import AdamW
from datasets import Dataset
from evaluate import load
from tqdm.auto import tqdm
from accelerate import Accelerator
import numpy as np
from extractive_predictor import ExtractivePredictor

project_root = Path(__file__).parent.parent
model_path = project_root / "models_task2" / "deepset_roberta-base-squad2" / "phrase"
# Configuration
config = dict(
    max_input_length=512,
    max_target_length=128,
    batch_size=4,
    epochs=10,
    learning_rate=1e-5,
    model_name="google/flan-t5-base",
    spoiler_type="phrase",
    extractive_model_path=model_path
)

# Load evaluation metrics
bleu = load("bleu")
bertscore = load("bertscore")

# Helper functions from original script
errorUuid = {"ad9271b7-9983-42f5-9bd9-fdfcb171ddaa": [[[4, 37], [4, 222]]]}

def parse_spoiler(x):
    spoiler = []
    if x['uuid'] in errorUuid:
        x['spoilerPositions'] = errorUuid[x['uuid']]

    for s in x['spoilerPositions']:
        st, en = s[0], s[1]
        spoiler.append(x['targetParagraphs'][st[0]][st[1]:en[1]])
    
    return " ".join(spoiler)  # Join multiple spoilers

def read_prep(path):
    with open(path, 'rb') as json_file:
        json_list = list(json_file)

    results = []
    for json_str in json_list:
        result = json.loads(json_str)
        results.append(result)
    
    df = pd.DataFrame(results)
    df['tags'] = df.tags.apply(lambda x: x[0], 1)
    df['postText'] = df.postText.apply(lambda x: x[0], 1)
    df['spoiler'] = df.apply(parse_spoiler, 1)
    df['mergedParas'] = df['targetParagraphs'].apply(lambda x: " ".join(x), 1)
    df.mergedParas = df.targetTitle + " - " + df.mergedParas
    
    return df

def generate_extractive_predictions(df, predictor):
    """Generate extractive predictions for each sample."""
    predictions = []
    
    print(f"Generating extractive predictions for {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        question = row['postText']
        context = row['mergedParas']
        
        try:
            result = predictor.predict(question=question, context=context, top_k=1)
            pred = result[0]["answer"] if result else ""
        except Exception as e:
            print(f"Error predicting: {e}")
            pred = ""
        
        predictions.append(pred)
    #df = df.copy()
    df['extractive_pred'] = predictions
    return df

# Prepare dataset for seq2seq
def prepare_refinement_data(df):
    data = []
    skipped = 0
    for _, row in df.iterrows():
        # Input: instruction + spoiler + context
        extractive_pred = row.get('extractive_pred', '')
        ground_truth = row['spoiler']
        if not extractive_pred or not ground_truth:
            skipped += 1
            continue
        input_text = f"Summarize: {extractive_pred}"
        target_text = ground_truth
        
        data.append({
            'input': input_text,
            'target': target_text,
            'extractive_pred': extractive_pred,
            'id': row['uuid']
        })
    print(f"Skipped {skipped} samples with empty input/target")
    return pd.DataFrame(data)



# Preprocessing function
def preprocess_function(examples):
    inputs = examples['input']
    targets = examples['target']
    
    model_inputs = tokenizer(
        inputs,
        max_length=config['max_input_length'],
        truncation=True,
        padding='max_length'
    )
    
    labels = tokenizer(
        targets,
        max_length=config['max_target_length'],
        truncation=True,
        padding='max_length'
    )
    
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

if os.path.exists(config['extractive_model_path']):
    predictor = ExtractivePredictor(config['extractive_model_path'])
    use_extractive = True
else:
    print(f"Warning: Extractive model not found at {config['extractive_model_path']}")
    print("Will use ground truth as input (oracle mode)")
    use_extractive = False

# Load and prepare data
print("Loading data...")
df_train = read_prep("./data/train.jsonl")
df_valid = read_prep("./data/validation.jsonl")

spoiler_type = config['spoiler_type']
train_df = df_train[df_train.tags == spoiler_type]
val_df = df_valid[df_valid.tags == spoiler_type]


print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

print("\n" + "=" * 60)
print("Generating extractive predictions...")
print("=" * 60)
    
if use_extractive:
    train_df = generate_extractive_predictions(train_df, predictor)
    val_df = generate_extractive_predictions(val_df, predictor)
else:
        # Oracle mode
    train_df['extractive_pred'] = train_df['spoiler']
    val_df['extractive_pred'] = val_df['spoiler']

# Prepare datasets
train_data = prepare_refinement_data(train_df)
val_data = prepare_refinement_data(val_df)

train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_data.reset_index(drop=True))

# Initialize model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
model = AutoModelForSeq2SeqLM.from_pretrained(config['model_name'])

# Preprocess datasets
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

val_dataset_processed = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

# Set format
train_dataset.set_format('torch')
val_dataset_processed.set_format('torch')

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=config['batch_size']
)

eval_dataloader = DataLoader(
    val_dataset_processed,
    batch_size=config['batch_size']
)

# Setup training
optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
accelerator = Accelerator(mixed_precision='no')
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_training_steps = config['epochs'] * len(train_dataloader)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

# Evaluation function
def evaluate_model():
    model.eval()
    predictions = []
    references = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch['input_ids'],
                #attention_mask=batch['attention_mask'],
                max_length=config['max_target_length'],
                num_beams=4,
                early_stopping=True
            )
        
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        
        # Clean up labels (remove padding tokens)
        decoded_labels = [label.replace(tokenizer.pad_token, '').strip() for label in decoded_labels]
        
        predictions.extend(decoded_preds)
        references.extend(decoded_labels)
    
    # Compute metrics
    bleu_score = bleu.compute(predictions=predictions, references=references)
    bert_score = bertscore.compute(
        predictions=predictions,
        references=references,
        lang='en',
        model_type='bert-base-uncased'
    )
    
    # Average BERTScore F1
    avg_bertscore = np.mean(bert_score['f1'])
    
    return {
        'bleu': bleu_score['bleu'],
        'bertscore_f1': avg_bertscore
    }, predictions, references

# Training loop
print("Starting training...")
best_score = 0.0

for epoch in range(config['epochs']):
    # Training
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        
        accelerator.backward(loss)
        total_loss += loss.item()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    num_batches = len(train_dataloader)
    avg_train_loss = total_loss / num_batches if num_batches > 0 and total_loss > 0 else float('nan')
    
    # Evaluation
    print(f"\nEpoch {epoch + 1} - Evaluating...")
    metrics, predictions, references = evaluate_model()
    
    # Combined score (weighted average of BLEU and BERTScore)
    combined_score = 0.5 * metrics['bleu'] + 0.5 * metrics['bertscore_f1']
    
    print(f"Epoch {epoch + 1}:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  BLEU: {metrics['bleu']:.4f}")
    print(f"  BERTScore F1: {metrics['bertscore_f1']:.4f}")
    print(f"  Combined Score: {combined_score:.4f}")
    
    # Save best model
    if combined_score > best_score:
        best_score = combined_score
        output_dir = f"./models_refinement_{spoiler_type}/"
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(output_dir)
        print(f"  Saved new best model! (score: {combined_score:.4f})")
    
    # Show sample predictions
    print("\nSample predictions:")
    for i in range(min(3, len(predictions))):
        print(f"  Pred: {predictions[i]}")
        print(f"  Ref:  {references[i]}\n")

print("Training complete!")
print(f"Best combined score: {best_score:.4f}")
print(f"Model saved to: ./models_refinement_{spoiler_type}/")
print("=" * 60)