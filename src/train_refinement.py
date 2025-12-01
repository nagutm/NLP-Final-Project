"""
Refinement Model Training Script
Trains T5/FLAN models to refine extractive spoiler predictions into better outputs.
Accepts RoBERTa extractive predictions and trains to generate improved spoilers.
"""

import json
import logging
import os
import sys
from typing import Dict, List, Optional

import random
import numpy as np
import torch
import yaml
from datasets import Dataset
from evaluate import load
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Type conversions
    numeric_keys = [
        "max_input_length",
        "max_target_length",
        "batch_size",
        "eval_batch_size",
        "num_epochs",
        "learning_rate",
        "warmup_steps",
        "weight_decay",
        "gradient_accumulation_steps",
        "dataset_fraction",
    ]
    
    for key in numeric_keys:
        if key in config:
            if key in ["learning_rate", "weight_decay"]:
                config[key] = float(config[key])
            elif key == "dataset_fraction":
                config[key] = float(config[key])
            else:
                config[key] = int(config[key])
    
    return config


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_refinement_pairs(
    extractive_predictions: List[Dict],
    original_data: List[Dict],
) -> List[Dict]:
    """
    Create training pairs for refinement model.
    
    Args:
        extractive_predictions: List of dicts with 'uuid' and 'spoiler' (RoBERTa outputs)
        original_data: Original JSONL data with ground truth spoilers
    
    Returns:
        List of dicts with 'input' (extractive spoiler), 'output' (ground truth)
    """
    # Map uuid -> original sample
    uuid_to_sample = {sample.get("uuid"): sample for sample in original_data}
    
    pairs = []
    for pred in extractive_predictions:
        uuid = pred.get("uuid")
        if uuid not in uuid_to_sample:
            continue
        
        original = uuid_to_sample[uuid]
        extracted = pred.get("spoiler", "").strip()
        
        # Get ground truth spoiler
        ground_truth = original.get("spoiler", [])
        if isinstance(ground_truth, list):
            ground_truth = " ".join(ground_truth)
        ground_truth = ground_truth.strip()
        
        # Validate that both extracted and ground truth are non-empty
        if not extracted or not ground_truth:
            continue
        
        # Additional validation: ensure strings are not just whitespace
        extracted = extracted.strip()
        ground_truth = ground_truth.strip()
        if not extracted or not ground_truth:
            continue
        
        # Get spoiler type for context
        spoiler_type = original.get("tags", ["unknown"])[0] if isinstance(original.get("tags"), list) else "unknown"
        
        # Create refinement pair
        pair = {
            "input": f"Refine ({spoiler_type}): {extracted}",
            "output": ground_truth,
            "spoiler_type": spoiler_type,
        }
        pairs.append(pair)
    
    logger.info(f"Created {len(pairs)} refinement training pairs")
    return pairs


def preprocess_function(examples: Dict, tokenizer, config: Dict) -> Dict:
    """Preprocess function for T5 refinement training."""
    max_input_length = config.get("max_input_length", 256)
    max_target_length = config.get("max_target_length", 128)
    
    # Ensure pad_token is set for the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )
    
    # Replace padding token ids with -100 (ignored by loss)
    # -100 is the default ignore_index for CrossEntropyLoss
    pad_token_id = tokenizer.pad_token_id
    
    # Ensure pad_token_id is valid
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    
    labels_input_ids = labels["input_ids"]
    
    # Replace padding tokens with -100 to ignore them in loss calculation
    labels_with_ignore = []
    for label_seq in labels_input_ids:
        processed_seq = [
            (token_id if token_id != pad_token_id else -100) for token_id in label_seq
        ]
        labels_with_ignore.append(processed_seq)
    
    model_inputs["labels"] = labels_with_ignore
    return model_inputs


def train_refinement_model(config_path: str = "configs/refinement_config.yaml"):
    """Main training function."""
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    
    logger.info("=" * 60)
    logger.info("CUDA/Device Information")
    logger.info("=" * 60)
    logger.info(f"CUDA Available: {cuda_available}")
    logger.info(f"GPU Device Count: {device_count}")
    if cuda_available:
        for i in range(device_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Configuration loaded from {config_path}")
    
    # Create directories
    os.makedirs(config.get("output_dir", "models/refinement_model"), exist_ok=True)
    os.makedirs(config.get("checkpoint_dir", "models/refinement_model/checkpoints"), exist_ok=True)
    os.makedirs(config.get("log_dir", "logs/refinement"), exist_ok=True)
    
    # Load model and tokenizer
    model_name = config.get("model_name", "google/flan-t5-base")
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Ensure pad_token is set (required for T5 models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Load original data
    data_dir = config.get("data_dir", "data")
    train_file = os.path.join(data_dir, "train.jsonl")
    val_file = os.path.join(data_dir, "validation.jsonl")
    
    logger.info(f"Loading training data from {train_file}")
    train_data = load_jsonl(train_file)
    
    logger.info(f"Loading validation data from {val_file}")
    val_data = load_jsonl(val_file)
    
    # Apply dataset subsampling for faster iteration
    dataset_fraction = config.get("dataset_fraction", 0.05)
    if dataset_fraction < 1.0:
        train_size = max(1, int(len(train_data) * dataset_fraction))
        val_size = max(1, int(len(val_data) * dataset_fraction))
        train_data = random.sample(train_data, train_size)
        val_data = random.sample(val_data, val_size)
        logger.info(f"Subsampled to {dataset_fraction*100}%: {len(train_data)} train, {len(val_data)} val")
    
    # Load RoBERTa predictions
    predictions_file = config.get("predictions_file", "results/predictions_phrase.json")
    
    if not os.path.exists(predictions_file):
        logger.error(f"Predictions file not found: {predictions_file}")
        logger.info("Using oracle mode (ground truth as extracted) for testing...")
        
        def create_oracle_predictions(data):
            """Create oracle predictions from ground truth (for testing)."""
            preds = []
            for sample in data:
                spoilers = sample.get("spoiler", [])
                if spoilers:
                    preds.append({
                        "spoiler": spoilers[0] if isinstance(spoilers, list) else spoilers,
                    })
            return preds
        
        train_preds = create_oracle_predictions(train_data)
        val_preds = create_oracle_predictions(val_data)
    else:
        # Load actual RoBERTa predictions
        logger.info(f"Loading RoBERTa predictions from {predictions_file}")
        with open(predictions_file, "r", encoding="utf-8") as f:
            all_predictions = json.load(f)
        
        logger.info(f"Loaded {len(all_predictions)} predictions")
        
        # Split predictions proportionally to train/val split
        train_fraction = len(train_data) / (len(train_data) + len(val_data))
        split_idx = int(len(all_predictions) * train_fraction)
        
        train_preds = []
        val_preds = []
        
        # Match predictions to data using index (assuming same order)
        for idx, pred in enumerate(all_predictions):
            if pred.get("top_prediction"):  # Only use non-empty predictions
                if idx < split_idx:
                    # Map to train data
                    if idx < len(train_data):
                        train_preds.append({
                            "uuid": train_data[idx].get("uuid"),
                            "spoiler": pred.get("top_prediction", ""),
                        })
                else:
                    # Map to validation data
                    val_idx = idx - split_idx
                    if val_idx < len(val_data):
                        val_preds.append({
                            "uuid": val_data[val_idx].get("uuid"),
                            "spoiler": pred.get("top_prediction", ""),
                        })
        
        logger.info(f"Extracted {len(train_preds)} train predictions and {len(val_preds)} val predictions")
    
    train_pairs = create_refinement_pairs(train_preds, train_data)
    val_pairs = create_refinement_pairs(val_preds, val_data)
    
    logger.info(f"Training pairs: {len(train_pairs)}")
    logger.info(f"Validation pairs: {len(val_pairs)}")
    
    if not train_pairs or not val_pairs:
        logger.error("No training or validation pairs created!")
        return
    
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_dict({
        "input": [p["input"] for p in train_pairs],
        "output": [p["output"] for p in train_pairs],
        "spoiler_type": [p["spoiler_type"] for p in train_pairs],
    })
    
    val_dataset = Dataset.from_dict({
        "input": [p["input"] for p in val_pairs],
        "output": [p["output"] for p in val_pairs],
        "spoiler_type": [p["spoiler_type"] for p in val_pairs],
    })
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    
    def preprocess_with_config(batch):
        return preprocess_function(batch, tokenizer, config)
    
    train_dataset = train_dataset.map(
        preprocess_with_config,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        preprocess_with_config,
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    # Validate datasets and log statistics
    logger.info("Validating preprocessed datasets...")
    
    def validate_labels(dataset, dataset_name):
        """Validate that labels are properly formatted."""
        invalid_count = 0
        all_padding_count = 0
        
        for idx in range(min(100, len(dataset))):  # Check first 100 samples
            sample = dataset[idx]
            labels = sample.get("labels", [])
            
            if not labels:
                invalid_count += 1
                continue
            
            # Check if all labels are -100 (padding)
            if all(label == -100 for label in labels):
                all_padding_count += 1
        
        logger.info(f"{dataset_name} validation (first 100 samples):")
        logger.info(f"  - Invalid samples: {invalid_count}")
        logger.info(f"  - All-padding samples: {all_padding_count}")
        
        # Log a sample for debugging
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"  - Sample input_ids length: {len(sample.get('input_ids', []))}")
            logger.info(f"  - Sample labels length: {len(sample.get('labels', []))}")
            labels_sample = sample.get('labels', [])
            non_padding_labels = [l for l in labels_sample[:20] if l != -100]
            logger.info(f"  - Sample non-padding labels (first 20): {non_padding_labels[:10]}")
        
        return invalid_count == 0 and all_padding_count == 0
    
    validate_labels(train_dataset, "Train dataset")
    validate_labels(val_dataset, "Validation dataset")
    
    # Filter out samples with all-padding labels (which cause NaN loss)
    logger.info("Filtering out invalid samples (all-padding labels)...")
    
    def filter_valid_samples(example):
        """Filter out examples where all labels are -100 (padding)."""
        labels = example.get("labels", [])
        if not labels:
            return False
        # Keep only samples with at least one non-padding label
        return any(label != -100 for label in labels)
    
    initial_train_size = len(train_dataset)
    initial_val_size = len(val_dataset)
    
    train_dataset = train_dataset.filter(filter_valid_samples)
    val_dataset = val_dataset.filter(filter_valid_samples)
    
    filtered_train = initial_train_size - len(train_dataset)
    filtered_val = initial_val_size - len(val_dataset)
    
    if filtered_train > 0 or filtered_val > 0:
        logger.warning(f"Filtered out {filtered_train} train samples and {filtered_val} val samples with all-padding labels")
    
    logger.info(f"Final dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("No valid training or validation samples after filtering!")
        logger.error("Check your input data - all labels may be empty or invalid.")
        return None, None
    
    # Load BLEU metric (same as new_training.py)
    logger.info("Loading BLEU metric...")
    bleu_metric = load("bleu")
    
    # Create compute_metrics function for BLEU evaluation
    def compute_metrics(eval_pred):
        """
        Compute BLEU score for validation predictions.
        Based on the metric computation from new_training.py.
        
        Args:
            eval_pred: EvalPrediction object containing predictions and label_ids
        
        Returns:
            Dictionary with BLEU score
        """
        predictions, labels = eval_pred
        
        # With predict_with_generate=True, predictions are already generated token IDs
        # Decode predictions (skip special tokens)
        if isinstance(predictions, tuple):
            predictions = predictions[0]  # In case it's a tuple
        
        # Ensure predictions are in the right format (list/tensor of token IDs)
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Convert to numpy array with signed integer type (int64)
        predictions = np.asarray(predictions, dtype=np.int64)
        
        # Handle 2D arrays (batch_size x seq_length)
        if predictions.ndim > 1:
            # Flatten for batch_decode if needed, or keep as is if tokenizer handles it
            pass
        
        # Ensure all prediction values are non-negative (valid token IDs)
        # Clip any negative values to 0 (though there shouldn't be any)
        predictions = np.clip(predictions, 0, None).astype(np.int64)
        
        decoded_preds = tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=True)
        
        # Decode labels (replace -100 with pad_token_id, then decode)
        # Labels contain -100 for padding tokens which should be ignored
        # Convert to numpy array if needed
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Convert to numpy array with signed integer type
        labels = np.asarray(labels, dtype=np.int64)
        
        # Handle pad_token_id - use eos_token_id if pad_token_id is None
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        
        # Ensure pad_token_id is non-negative
        if pad_token_id < 0:
            pad_token_id = 0
        
        # Replace -100 with pad_token_id (must be non-negative)
        labels = np.where(labels != -100, labels, pad_token_id)
        
        # Ensure all label values are non-negative (valid token IDs)
        labels = np.clip(labels, 0, None).astype(np.int64)
        
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
        
        # Prepare for BLEU computation (similar to new_training.py)
        # BLEU expects references as list of lists
        references = [[ref] if isinstance(ref, str) else ref for ref in decoded_labels]
        predictions_list = decoded_preds
        
        # Compute BLEU (same as new_training.py)
        try:
            bleu_result = bleu_metric.compute(
                predictions=predictions_list,
                references=references
            )
            
            # Extract BLEU score
            bleu_score = bleu_result.get('bleu', 0.0)
        except Exception as e:
            logger.warning(f"Error computing BLEU score: {e}")
            bleu_score = 0.0
        
        logger.info(f"BLEU score: {bleu_score:.4f}")
        
        return {
            "bleu": bleu_score,
        }
    
    # Training arguments
    use_cuda = cuda_available and device_count > 0
    use_fp16 = use_cuda
    
    logger.info(f"Training will use: {'CUDA' if use_cuda else 'CPU'}")
    logger.info(f"Mixed Precision (FP16): {use_fp16}")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.get("output_dir", "models/refinement_model"),
        eval_strategy=config.get("eval_strategy", "epoch"),
        learning_rate=config.get("learning_rate", 1e-4),
        per_device_train_batch_size=config.get("batch_size", 8),
        per_device_eval_batch_size=config.get("eval_batch_size", config.get("batch_size", 8)),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        num_train_epochs=config.get("num_epochs", 3),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 300),
        logging_dir=config.get("log_dir", "logs/refinement"),
        logging_steps=100,
        save_strategy=config.get("save_strategy", "epoch"),
        save_total_limit=config.get("save_total_limit", 2),
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),  # Default to loss, can be changed to eval_bleu
        greater_is_better=config.get("metric_for_best_model", "eval_loss") != "eval_loss",  # True for BLEU, False for loss
        predict_with_generate=True,
        fp16=use_fp16,
        no_cuda=not use_cuda,
        seed=config.get("seed", 42),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        max_grad_norm=config.get("max_grad_norm", 1.0),  # Gradient clipping to prevent NaN
        report_to="none",  # Disable wandb/tensorboard if not needed
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(config.get("output_dir", "models/refinement_model"), "final_model")
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info("Training complete!")
    
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train refinement model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/refinement_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    train_refinement_model(args.config)
