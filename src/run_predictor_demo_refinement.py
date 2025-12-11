"""
Evaluate the trained refinement model on validation data.
"""

import json
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
from tqdm import tqdm
import numpy as np

# Configuration
config = dict(
    max_input_length=512,
    max_target_length=128,
    batch_size=8,
    spoiler_type="phrase",  # Options: "passage", "title", "post"
    model_path="./models_refinement_phrase/",  # Path to your trained model
)

# Load metrics
bleu_metric = load("bleu")
bertscore_metric = load("bertscore")

# Helper functions
errorUuid = {"ad9271b7-9983-42f5-9bd9-fdfcb171ddaa": [[[4, 37], [4, 222]]]}

def parse_spoiler(x):
    spoiler = []
    if x['uuid'] in errorUuid:
        x['spoilerPositions'] = errorUuid[x['uuid']]
    for s in x['spoilerPositions']:
        st, en = s[0], s[1]
        spoiler.append(x['targetParagraphs'][st[0]][st[1]:en[1]])
    return " ".join(spoiler)

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

def evaluate_refinement_model():
    project_root = Path(__file__).parent
    
    # Load trained model and tokenizer
    print(f"Loading model from {config['model_path']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model_path'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load extractive predictor (same as in training)
    from extractive_predictor import ExtractivePredictor
    extractive_model_path = project_root / "models_task2" / "deepset_roberta-base-squad2" / "phrase"
    
    if extractive_model_path.exists():
        predictor = ExtractivePredictor(str(extractive_model_path))
        use_extractive = True
        print(f"Loaded extractive model from {extractive_model_path}")
    else:
        print("Extractive model not found, using oracle mode")
        use_extractive = False
    
    # Load validation data
    print("Loading validation data...")
    df_valid = read_prep("./data/validation.jsonl")
    val_df = df_valid[df_valid.tags == config['spoiler_type']]
    print(f"Validation samples: {len(val_df)}")
    
    # Generate predictions
    all_predictions = []
    all_references = []
    
    print("Generating predictions...")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        question = row['postText']
        context = row['mergedParas']
        ground_truth = row['spoiler']
        
        # Get extractive prediction
        if use_extractive:
            try:
                result = predictor.predict(question=question, context=context, top_k=1)
                extractive_pred = result[0]["answer"] if result else ""
            except Exception as e:
                print(f"Error in extractive prediction: {e}")
                extractive_pred = ""
        else:
            extractive_pred = ground_truth  # Oracle mode
        
        if not extractive_pred:
            all_predictions.append("")
            all_references.append(ground_truth)
            continue
        
        # Prepare input for refinement model
        input_text = f"Summarize: {extractive_pred}"
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            max_length=config['max_input_length'],
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=config['max_target_length'],
                num_beams=4,
                early_stopping=True
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        all_predictions.append(prediction)
        all_references.append(ground_truth)
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # BLEU-4
    '''bleu_result = bleu_metric.compute(predictions=all_predictions, references=all_references)
    bleu4_score = bleu_result['precisions'][3]'''
    def compute_sentence_bleu(reference: str, prediction: str) -> float:
     if not reference or not prediction:
        return 0.0
     try:
        result = bleu_metric.compute(
            predictions=[prediction],
            references=[[reference]],
            max_order=4,
            smooth=True
        )
        return float(result["bleu"])
     except Exception:
        return 0.0

    bleu_scores = []
    for pred, ref in zip(all_predictions, all_references):
      score = compute_sentence_bleu(ref, pred)
      bleu_scores.append(score)

    bleu4_score = float(np.mean(bleu_scores))
    bertscore_result = bertscore_metric.compute(
        predictions=all_predictions,
        references=all_references,
        lang='en',
        model_type='bert-base-uncased'
    )
    avg_bertscore_f1 = sum(bertscore_result['f1']) / len(bertscore_result['f1'])
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {config['model_path']}")
    print(f"Spoiler type: {config['spoiler_type']}")
    print(f"Num samples: {len(all_predictions)}")
    print(f"BLEU-4: {bleu4_score:.4f}")
    print(f"BERTScore F1: {avg_bertscore_f1:.4f}")
    print("=" * 60)
    
    # Save results
    results = {
        "model_path": config['model_path'],
        "spoiler_type": config['spoiler_type'],
        "num_samples": len(all_predictions),
        "bleu4": float(bleu4_score),
        "bertscore_f1": float(avg_bertscore_f1),
    }
    
    # Save sample predictions
    samples = []
    for i in range(len(all_predictions)):
        samples.append({
            "prediction": all_predictions[i],
            "reference": all_references[i]
        })
    results["samples"] = samples
    
    output_path = Path(f"./results/refinement_eval_results_{config['spoiler_type']}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    evaluate_refinement_model()