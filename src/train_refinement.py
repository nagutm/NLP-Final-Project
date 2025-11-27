"""
Refinement Model Training Script
Trains FLAN-T5 to refine extractive spoiler predictions.
"""

import json
import logging
import os
import sys
from typing import Dict, List

import torch
import yaml
from datasets import Dataset
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
        return yaml.safe_load(f)


def load_refinement_pairs(file_path: str) -> List[Dict]:
    """Load refinement training pairs."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_function(examples: Dict, tokenizer, config: dict):
    """
    Preprocess refinement pairs for training.
    
    Args:
        examples: Batch of examples with 'input' and 'output' fields
        tokenizer: T5 tokenizer
        config: Configuration dictionary
    
    Returns:
        Tokenized inputs and labels
    """
    max_input_length = config.get("max_input_length", 512)
    max_output_length = config.get("max_output_length", 128)
    
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    
    # Tokenize targets
    labels = tokenizer(
        examples["output"],
        max_length=max_output_length,
        truncation=True,
        padding="max_length",
    )
    
    model_inputs["labels"] = labels["input_ids"]
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
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)
    
    # Load model and tokenizer
    model_name = config["model_name"]
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Load refinement pairs
    train_file = os.path.join(config["data_dir"], "processed/refinement_train.json")
    val_file = os.path.join(config["data_dir"], "processed/refinement_val.json")
    
    logger.info(f"Loading training pairs from {train_file}")
    train_pairs = load_refinement_pairs(train_file)
    
    logger.info(f"Loading validation pairs from {val_file}")
    val_pairs = load_refinement_pairs(val_file)
    
    logger.info(f"Training examples: {len(train_pairs)}")
    logger.info(f"Validation examples: {len(val_pairs)}")
    
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
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config),
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    # Training arguments
    use_cuda = cuda_available and device_count > 0
    use_fp16 = use_cuda
    
    logger.info(f"Training will use: {'CUDA' if use_cuda else 'CPU'}")
    logger.info(f"Mixed Precision (FP16): {use_fp16}")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        evaluation_strategy=config.get("evaluation_strategy", "epoch"),
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config.get("eval_batch_size", config["batch_size"]),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        num_train_epochs=config["num_epochs"],
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 300),
        logging_dir=config["log_dir"],
        logging_steps=100,
        save_strategy=config.get("save_strategy", "epoch"),
        save_total_limit=config.get("save_total_limit", 2),
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        predict_with_generate=True,
        fp16=use_fp16,
        no_cuda=not use_cuda,
        seed=config.get("seed", 42),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
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
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(config["output_dir"], "final_model")
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