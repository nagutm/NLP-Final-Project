"""
Extractive Model Training Script
Trains DeBERTa-v3-base for clickbait spoiler span extraction using question-answering approach.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import random
import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
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
    
    # Ensure numeric values are properly typed
    numeric_keys = [
        "max_seq_length",
        "batch_size",
        "eval_batch_size",
        "num_epochs",
        "learning_rate",
        "warmup_steps",
        "weight_decay",
        "gradient_accumulation_steps",
        "dataset_fraction",
        "max_span_length",
        "confidence_threshold",
    ]
    
    for key in numeric_keys:
        if key in config:
            if key in ["learning_rate", "weight_decay", "confidence_threshold"]:
                config[key] = float(config[key])
            else:
                # dataset_fraction should be treated as float not int
                if key == "dataset_fraction":
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


def create_qa_examples(data: List[Dict], dataset_type: str = "train") -> Tuple[List[Dict], List[Dict]]:
    """
    Convert clickbait spoiler dataset to QA format.
    
    Args:
        data: List of sample dictionaries from JSONL
        dataset_type: 'train' or 'validation'
    
    Returns:
        Tuple of (examples, features) where examples are QA-style and features track metadata
    """
    examples = []
    features = []
    
    for sample in data:
        # Extract necessary information
        post_text = " ".join(sample.get("postText", []))
        target_paragraphs = sample.get("targetParagraphs", [])
        spoiler_text = sample.get("spoiler", [])
        spoiler_positions = sample.get("spoilerPositions", [])
        tags = sample.get("tags", [])
        
        if not target_paragraphs or not spoiler_text:
            continue
        
        # Concatenate all target paragraphs to form the context
        context = " ".join(target_paragraphs)
        
        # Use post text as the question
        question = post_text
        
        # Process each spoiler position
        for spoiler_idx, spoiler_pos_list in enumerate(spoiler_positions):
            for para_idx, char_pos in enumerate(spoiler_pos_list):
                start_char, end_char = char_pos
                
                # Get the actual spoiler text at this position
                if para_idx < len(target_paragraphs):
                    paragraph = target_paragraphs[para_idx]
                    actual_spoiler = paragraph[start_char:end_char]
                    
                    # Calculate offset in concatenated context
                    context_offset = sum(
                        len(target_paragraphs[i]) + 1 for i in range(para_idx)  # +1 for space
                    )
                    
                    example = {
                        "question": question,
                        "context": context,
                        "answer_text": actual_spoiler,
                        "answer_start": context_offset + start_char,
                        "id": f"{sample.get('uuid', '')}__{spoiler_idx}_{para_idx}",
                    }
                    
                    examples.append(example)
                    features.append({
                        "id": sample.get("uuid", ""),
                        "tags": tags,
                        "platform": sample.get("postPlatform", ""),
                    })
    
    logger.info(f"Created {len(examples)} examples from {len(data)} samples")
    return examples, features


def prepare_train_features(examples: Dict, tokenizer, config: dict) -> Dict:
    """Prepare features for training."""
    max_seq_length = config.get("max_seq_length", 512)
    doc_stride = 128
    
    # Tokenize contexts and questions
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Get offset mapping
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    # Initialize start and end labels
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answer_start"][sample_idx]
        answer_text = examples["answer_text"][sample_idx]
        
        # If no answer, set positions to (0, 0)
        if answers == -1:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue
        
        start_char = answers
        end_char = start_char + len(answer_text)
        
        # Find token positions
        token_start_index = 0
        token_end_index = len(offsets) - 1
        
        for j, (offset_start, offset_end) in enumerate(offsets):
            if offset_start <= start_char < offset_end:
                token_start_index = j
            if offset_start < end_char <= offset_end:
                token_end_index = j
                break
        
        tokenized_examples["start_positions"].append(token_start_index)
        tokenized_examples["end_positions"].append(token_end_index)
    
    return tokenized_examples


def prepare_validation_features(examples: Dict, tokenizer, config: dict) -> Dict:
    """Prepare features for validation."""
    max_seq_length = config.get("max_seq_length", 512)
    doc_stride = 128
    
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Store mapping info for post-processing
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []
    
    for i in range(len(tokenized_examples["input_ids"])):
        sample_idx = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_idx])
    
    return tokenized_examples


def train_extractive_model(config_path: str = "configs/extractive_config.yaml"):
    """Main training function."""
    
    # Check CUDA availability and configuration
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    
    logger.info("=" * 60)
    logger.info("CUDA/Device Information")
    logger.info("=" * 60)
    logger.info(f"CUDA Available: {cuda_available}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"GPU Device Count: {device_count}")
    
    if cuda_available:
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  Memory: {props.total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA is not available. Training will use CPU.")
        logger.info("To enable CUDA, reinstall PyTorch with CUDA support:")
        logger.info("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Configuration loaded from {config_path}")
    
    # Create directories
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config['model_name']}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        model = AutoModelForQuestionAnswering.from_pretrained(config["model_name"])
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Attempting with trust_remote_code=True...")
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
        model = AutoModelForQuestionAnswering.from_pretrained(config["model_name"], trust_remote_code=True)
    
    # Load data
    data_dir = config["data_dir"]
    train_file = os.path.join(data_dir, config["train_file"])
    val_file = os.path.join(data_dir, config["validation_file"])
    
    logger.info(f"Loading training data from {train_file}")
    train_data = load_jsonl(train_file)
    train_examples, train_features = create_qa_examples(train_data, "train")
    
    logger.info(f"Loading validation data from {val_file}")
    val_data = load_jsonl(val_file)
    val_examples, val_features = create_qa_examples(val_data, "validation")

    # Subsample datasets if configured (e.g., 0.05 for 5%)
    dataset_fraction = config.get("dataset_fraction", 1.0)
    if 0.0 < dataset_fraction < 1.0:
        seed = config.get("seed", 42)
        random.seed(seed)

        n_train = max(1, int(len(train_examples) * dataset_fraction))
        n_val = max(1, int(len(val_examples) * dataset_fraction))

        logger.info(
            f"Subsampling datasets to {dataset_fraction*100:.2f}% -> {n_train} train, {n_val} val"
        )

        # If dataset is small enough that sample would be the same size, skip
        if n_train < len(train_examples):
            train_examples = random.sample(train_examples, n_train)
        if n_val < len(val_examples):
            val_examples = random.sample(val_examples, n_val)
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_dict({
        "question": [ex["question"] for ex in train_examples],
        "context": [ex["context"] for ex in train_examples],
        "answer_text": [ex["answer_text"] for ex in train_examples],
        "answer_start": [ex["answer_start"] for ex in train_examples],
        "id": [ex["id"] for ex in train_examples],
    })
    
    val_dataset = Dataset.from_dict({
        "question": [ex["question"] for ex in val_examples],
        "context": [ex["context"] for ex in val_examples],
        "answer_text": [ex["answer_text"] for ex in val_examples],
        "answer_start": [ex["answer_start"] for ex in val_examples],
        "id": [ex["id"] for ex in val_examples],
    })
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    # Prepare features
    logger.info("Preparing training features...")
    train_dataset = train_dataset.map(
        lambda x: prepare_train_features(x, tokenizer, config),
        batched=True,
        remove_columns=train_dataset.column_names,
        batch_size=1000,
    )
    
    logger.info("Preparing validation features...")
    val_dataset = val_dataset.map(
        lambda x: prepare_validation_features(x, tokenizer, config),
        batched=True,
        remove_columns=val_dataset.column_names,
        batch_size=1000,
    )
    
    # Determine device and mixed precision settings
    use_cuda = cuda_available and device_count > 0
    use_fp16 = use_cuda  # Only use FP16 if CUDA is available
    
    logger.info(f"Training will use: {'CUDA' if use_cuda else 'CPU'}")
    logger.info(f"Mixed Precision (FP16): {use_fp16}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["num_epochs"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        logging_dir=config["log_dir"],
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=False,
        seed=42,
        fp16=use_fp16,
        no_cuda=not use_cuda,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    model_save_path = os.path.join(config["output_dir"], "final_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    return model, tokenizer


if __name__ == "__main__":
    config_path = "configs/extractive_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    train_extractive_model(config_path)
