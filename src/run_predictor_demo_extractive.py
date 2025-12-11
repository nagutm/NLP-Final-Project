"""
Demo script to run the extractive predictor on sample data.
"""

import json
import sys
import logging
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractive_predictor import ExtractivePredictor
from utils import load_jsonl
import evaluate
import statistics
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

bleu_metric = evaluate.load("bleu")

def _compute_bleu4(reference: str, candidate: str) -> float:
    if not reference or not candidate:
        return 0.0
    try:
        result = bleu_metric.compute(
            predictions=[candidate],
            references=[[reference]],
            max_order=4,
            smooth=True
        )
        return float(result["precisions"][3])
    except Exception:
        return 0.0

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
            "deepset/roberta-base-squad2"
        ]

    # Load validation data
    logger.info("Loading validation data...")
    val_data_path = project_root / "data" / "validation.jsonl"
    if not val_data_path.exists():
        logger.error(f"Validation file not found: {val_data_path}")
        return
    val_data = load_jsonl(str(val_data_path))
    num_samples = len(val_data)
    TARGET_SPOILER_TYPE = "phrase"


    for model_name in model_list:
        # create a filesystem-safe model name
        model_name_safe = model_name.replace("/", "_")
        model_path = project_root / "models_task2" / model_name_safe / TARGET_SPOILER_TYPE

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
        processed = 0
        bleu_vals = []
        samples = []
        for i, example in enumerate(val_data):
            # Only process up to num_samples phrase examples
            if processed >= num_samples:
                break

            spoiler_type = example.get("tags", ["unknown"])
            if isinstance(spoiler_type, list):
                spoiler_type = spoiler_type[0] if spoiler_type else "unknown"
            if spoiler_type != TARGET_SPOILER_TYPE:
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
            bleu4 = _compute_bleu4(spoiler, top_pred)
            bleu_vals.append(bleu4)
            processed += 1
            samples.append({
                "question": post,
                "prediction": top_pred,
                "reference": spoiler,
                "bleu": bleu4
            })

            logger.info(f"Sample {processed}/{num_samples} — BLEU: {bleu4:.4f}")
        if bleu_vals:
            mean_bleu = float(np.mean(bleu_vals))
            logger.info(f"BLEU-4: {mean_bleu:.4f} over {len(bleu_vals)} examples")
        else:
           logger.info("No examples processed.")
        results = {
            "model": model_name,
            "spoiler_type": TARGET_SPOILER_TYPE,
            "num_samples": len(bleu_vals),
            "bleu_mean": mean_bleu,
            "bleu_std": float(np.std(bleu_vals)) if bleu_vals else 0.0,
            "bleu_min": float(min(bleu_vals)) if bleu_vals else 0.0,
            "bleu_max": float(max(bleu_vals)) if bleu_vals else 0.0,
            "samples": samples
        }

        # Save to JSON
        out_dir = project_root / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"predictions_{model_name_safe}_{TARGET_SPOILER_TYPE}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")

        

    logger.info("Demo complete.")



if __name__ == "__main__":
    run_predictor_demo()
