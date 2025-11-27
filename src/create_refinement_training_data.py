"""
Create Refinement Training Data
Generates (extractive_output, ground_truth) pairs for training refinement model.
"""

import json
import logging
import os
from typing import Dict, List
from pathlib import Path

import yaml
from tqdm import tqdm
from extractive_predictor import ExtractivePredictor

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


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_prompt(extractive_output: str, spoiler_type: str = None, style: str = "simple") -> str:
    """
    Create input prompt for refinement model.
    
    Args:
        extractive_output: The extractive model's prediction
        spoiler_type: Optional spoiler type (phrase/passage/multi)
        style: Prompt style ("simple", "with_type", "explicit")
    
    Returns:
        Formatted prompt string
    """
    if style == "with_type" and spoiler_type:
        return f"Refine this {spoiler_type} spoiler: {extractive_output}"
    elif style == "explicit":
        return f"Convert this verbose spoiler into a concise answer: {extractive_output}"
    else:  # simple
        return f"Refine this spoiler: {extractive_output}"


def generate_refinement_pairs(
    config: dict,
    split: str = "train"
) -> List[Dict]:
    """
    Generate refinement training pairs.
    
    Args:
        config: Configuration dictionary
        split: "train" or "validation"
    
    Returns:
        List of refinement pairs with input/output
    """
    # Load extractive model
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "extractive_model" / "final_model"
    #extractive_model_path = config["extractive_model_path"]
    predictor = ExtractivePredictor(model_path)
    
    # Load data
    data_file = config[f"{split}_file"]
    data_path = os.path.join(config["data_dir"], data_file)
    logger.info(f"Loading {split} data from {data_path}")
    data = load_jsonl(data_path)
    
    # Subsample if configured
    dataset_fraction = config.get("dataset_fraction", 1.0)
    if 0.0 < dataset_fraction < 1.0:
        import random
        random.seed(config.get("seed", 42))
        n_samples = max(1, int(len(data) * dataset_fraction))
        data = random.sample(data, n_samples)
        logger.info(f"Subsampled to {n_samples} examples ({dataset_fraction*100:.1f}%)")
    
    logger.info(f"Generating refinement pairs for {len(data)} examples...")
    
    refinement_pairs = []
    prompt_style = config.get("prompt_style", "simple")
    
    for sample in tqdm(data, desc=f"Processing {split} set"):
        # Extract information
        post_text = " ".join(sample.get("postText", []))
        target_paragraphs = sample.get("targetParagraphs", [])
        context = " ".join(target_paragraphs) if isinstance(target_paragraphs, list) else target_paragraphs
        
        # Handle spoiler as list or string
        spoiler = sample.get("spoiler", [])
        if isinstance(spoiler, list):
            ground_truth = " ".join(spoiler)
        else:
            ground_truth = spoiler
        
        spoiler_type = sample.get("tags", ["unknown"])[0]
        
        # if not ground_truth.strip():
        #     continue
        
        # Get extractive prediction
        extractive_output = predictor.predict(question=post_text, context=context, top_k=1)
        extractive_output = extractive_output[0]["answer"] if extractive_output else ""
        
        # if not extractive_output.strip():
        #     # If extraction failed, use ground truth as input
        #     # This teaches model to pass through good spoilers
        #     extractive_output = ground_truth
        
        # Create prompt
        input_text = create_prompt(extractive_output, spoiler_type, prompt_style)
        
        # Create pair
        pair = {
            "uuid": sample.get("uuid", ""),
            "input": input_text,
            "output": ground_truth,
            "spoiler_type": spoiler_type,
            "extractive_output": extractive_output,
            "ground_truth": ground_truth,
        }
        
        refinement_pairs.append(pair)
    
    logger.info(f"Generated {len(refinement_pairs)} refinement pairs")
    return refinement_pairs


def main():
    """Main function to generate refinement training data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate refinement training data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/refinement_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for refinement pairs"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate training pairs
    logger.info("="*80)
    logger.info("GENERATING TRAINING SET")
    logger.info("="*80)
    train_pairs = generate_refinement_pairs(config, split="train")
    
    train_output = os.path.join(args.output_dir, "refinement_train.json")
    with open(train_output, "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved training pairs to {train_output}")
    
    # Generate validation pairs
    logger.info("\n" + "="*80)
    logger.info("GENERATING VALIDATION SET")
    logger.info("="*80)
    val_pairs = generate_refinement_pairs(config, split="validation")
    
    val_output = os.path.join(args.output_dir, "refinement_val.json")
    with open(val_output, "w", encoding="utf-8") as f:
        json.dump(val_pairs, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved validation pairs to {val_output}")
    
    # Print statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Training pairs: {len(train_pairs)}")
    logger.info(f"Validation pairs: {len(val_pairs)}")
    
    # Show examples
    logger.info("\n" + "="*80)
    logger.info("SAMPLE REFINEMENT PAIRS")
    logger.info("="*80)
    
    for i, pair in enumerate(train_pairs[:3], 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"  Type: {pair['spoiler_type']}")
        logger.info(f"  Input: {pair['input'][:150]}...")
        logger.info(f"  Output: {pair['output'][:150]}...")
        logger.info(f"  Extractive was: {pair['extractive_output'][:100]}...")


if __name__ == "__main__":
    main()