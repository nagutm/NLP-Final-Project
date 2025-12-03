"""
Demo script to run the extractive predictor on sample data.
"""

import json
import sys
import logging
from pathlib import Path
from evaluate import load

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractive_predictor import ExtractivePredictor
from utils import load_jsonl
import statistics
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


bleu_metric = load("bleu")


def run_predictor_demo():
    project_root = Path(__file__).parent.parent
    # Try to load model list from config if available (optional)
    cfg_path = project_root / "extractive_config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text())
            model_list = cfg.get("models") or ([cfg.get("model_name")] if cfg.get("model_name") else None)
        except Exception as e:
            logger.warning(f"Failed to load config {cfg_path}: {e}")
            model_list = None
    else:
        model_list = None

    if not model_list:
        model_list = [
            "deepset/deberta-v3-base-squad2", 
            "deepset/roberta-base-squad2",
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        ]

    # Load validation data
    logger.info("Loading validation data...")
    val_data_path = project_root / "data" / "validation.jsonl"
    if not val_data_path.exists():
        logger.error(f"Validation file not found: {val_data_path}")
        return
    val_data = load_jsonl(str(val_data_path))
    num_samples = len(val_data)

    # We'll save per-model results and a combined mapping
    combined_results = {}

    for model_name in model_list:
        # create a filesystem-safe model name
        model_name_safe = model_name.replace("/", "_")
        model_path = project_root / "models_task2" / model_name_safe / "phrase"

        logger.info("\n" + "=" * 80)
        logger.info(f"MODEL: {model_name} -> looking for checkpoint at: {model_path}")
        if not model_path.exists():
            logger.warning(f"Skipping {model_name}: model path not found: {model_path}")
            continue

        try:
            logger.info("Loading predictor...")
            predictor = ExtractivePredictor(
                model_path=str(model_path),
                max_seq_length=512,
                max_answer_length=100,
                confidence_threshold=0.0,
            )
        except Exception as e:
            logger.error(f"Failed to load predictor for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        logger.info(f"Running predictions for {model_name} on up to {num_samples} phrase samples...")
        results = []
        processed = 0
        all_predictions = []
        all_references = []

        for i, example in enumerate(val_data):
            # Only process up to num_samples phrase examples
            if processed >= num_samples:
                break

            spoiler_type = example.get("tags", ["unknown"])
            if isinstance(spoiler_type, list):
                spoiler_type = spoiler_type[0] if spoiler_type else "unknown"
            if spoiler_type != "phrase":
                continue

            # Prepare input
            post = example.get("postText", [])
            if isinstance(post, list):
                post = " ".join(post) if post else ""
            target_paragraphs = example.get("targetParagraphs", [])
            article = " ".join(target_paragraphs) if isinstance(target_paragraphs, list) else target_paragraphs
            spoiler = example.get("spoiler", [])
            if isinstance(spoiler, list):
                spoiler = " ".join(spoiler) if spoiler else ""

            # Run prediction
            try:
                preds = predictor.predict(question=post, context=article, top_k=3)
            except Exception as e:
                logger.error(f"Error during prediction on sample {i}: {e}")
                preds = []

            # Build a compact result entry
            top_pred = preds[0]['answer'] if preds else ""
            all_predictions.append(top_pred)
            all_references.append(spoiler)
            entry = {
                "question": post,
                "context_snippet": (article[:300] + "...") if len(article) > 300 else article,
                "ground_truth": spoiler,
                "predictions": [
                    {
                        "answer": p.get("answer", ""),
                        "confidence": float(p.get("confidence", 0.0)),
                    }
                    for p in preds
                ],
                "top_prediction": top_pred,
                "spoiler_type": spoiler_type
            }
            results.append(entry)
            processed += 1
            logger.info(f"[{model_name_safe}] sample {processed}/{num_samples} — top: '{top_pred}'")
        if all_predictions and all_references:
            bleu_result = bleu_metric.compute(predictions=all_predictions, references=all_references)
            bleu4_score = bleu_result['precisions'][3]  # Index 3 = 4-gram precision
        else:
            bleu4_score = 0.0
        
        summary = {
            "model": model_name,
            "model_safe": model_name_safe,
            "num_examples": len(all_predictions),
            "bleu4": float(bleu4_score),
        }
        # Save per-model JSON
        out_dir = project_root / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        per_model_path = out_dir / f"predictions_{model_name_safe}.json"
        with per_model_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(results)} results for {model_name} -> {per_model_path}")

        summary_path = out_dir / f"summary_{model_name_safe}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved summary for {model_name} -> {summary_path}")

        logger.info(f"[{model_name_safe}] BLEU mean: {summary['bleu4']:.4f} | examples: {summary['num_examples']}")

    logger.info("Demo complete.")



if __name__ == "__main__":
    run_predictor_demo()
