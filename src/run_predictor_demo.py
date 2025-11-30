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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_predictor_demo(model_names=None):
    """Run predictor on validation samples."""
    
    # Get absolute path to model
    project_root = Path(__file__).parent.parent

    if model_names is None:
        model_names = ["extractive_model"]

    logger.info("Loading validation data...")
    val_data_path = project_root / "data" / "validation.jsonl"
    val_data = load_jsonl(str(val_data_path))
    
    # Run on first 100 examples
    num_samples = len(val_data)
    logger.info(f"Running predictions on {num_samples} samples...")
    
    for model_name in model_names:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Evaluating model: {model_name}")
        logger.info("=" * 80)

        # Build model path: models/<model_name>/final_model
        model_path = project_root / "models" / model_name / "final_model"

        if not model_path.exists():
            logger.error(f"Model not found at {model_path}, skipping.")
            continue

        logger.info(f"Loading predictor from {model_path}...")
        predictor = ExtractivePredictor(
            model_path=model_path,
            max_seq_length=512,
            max_answer_length=100,
            confidence_threshold=0.0,
        )

        results = []
    
        for i, example in enumerate(val_data):
         logger.info(f"\n{'='*80}")
         logger.info(f"Sample {i+1}/{num_samples}")
         logger.info(f"{'='*80}")
        
         # Extract fields
         post = example.get("postText", [])
         if isinstance(post, list):
            post = " ".join(post) if post else ""
        
         target_paragraphs = example.get("targetParagraphs", [])
         article = " ".join(target_paragraphs) if isinstance(target_paragraphs, list) else target_paragraphs
        
         spoiler = example.get("spoiler", [])
         if isinstance(spoiler, list):
            spoiler = " ".join(spoiler) if spoiler else ""
        
         spoiler_type = example.get("tags", ["unknown"])
         if isinstance(spoiler_type, list):
            spoiler_type = spoiler_type[0] if spoiler_type else "unknown"
        
         # Truncate for display
         post_display = (post[:100] + "...") if len(post) > 100 else post
         article_display = (article[:150] + "...") if len(article) > 150 else article
         spoiler_display = (spoiler[:100] + "...") if len(spoiler) > 100 else spoiler
        
         logger.info(f"Clickbait Post: {post_display}")
         logger.info(f"Spoiler Type: {spoiler_type}")
         logger.info(f"Article (first 150 chars): {article_display}")
         logger.info(f"Ground Truth Spoiler: {spoiler_display}")
         # Run prediction
         try:
            predictions = predictor.predict(
                question=post,
                context=article,
                top_k=3
            )
            
            if predictions:
              top_prediction = predictions[0]["answer"]
              top_confidence = float(predictions[0]["confidence"])
            else:
              top_prediction = ""
              top_confidence = 0.0
            if predictions:
              top_pred_lower = predictions[0]["answer"].lower()
              ground_truth_lower = spoiler.lower()
              match = top_pred_lower in ground_truth_lower or ground_truth_lower in top_pred_lower
            else:
              match = False
            
            # Add to results
            results.append({
                "ground_truth": spoiler,
                "top_prediction": top_prediction,
                "top_confidence": top_confidence,
                "spoiler_type": spoiler_type,
                "model_name": model_name,
                "match": match
            })
            
            logger.info(f"\nTop Predictions:")
            for j, pred in enumerate(predictions, 1):
                logger.info(f"  {j}. Answer: '{pred['answer']}' (Confidence: {pred['confidence']:.4f})")
                logger.info(f"     Char positions: {pred['char_start']}-{pred['char_end']}")
            logger.info(f"Match with ground truth: {'✓ YES' if match else '✗ NO'}")
        
         except Exception as e:
            logger.error(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            # Add error result
            results.append({
                "ground_truth": spoiler,
                "top_prediction": "",
                "top_confidence": 0.0,
                "spoiler_type": spoiler_type,
                "model_name": model_name,
                "match": False
            })
    
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"predictions_{model_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info("")
        logger.info(f"Demo complete for model: {model_name}")
        logger.info(f"Model location: {model_path}")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Total predictions: {len(results)}")

    logger.info("")
    logger.info("All models evaluated.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        models = ["extractive_model"]
    run_predictor_demo(models)
