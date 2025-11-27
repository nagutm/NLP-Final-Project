"""
BLEU-4 Evaluation Script
Computes BLEU-4 scores for clickbait spoiler predictions.
"""

import json
import logging
from typing import Dict, List

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_bleu4(reference: str, candidate: str) -> float:
    """
    Compute BLEU-4 score for a single prediction.
    
    Args:
        reference: Ground truth spoiler
        candidate: Predicted spoiler
        
    Returns:
        BLEU-4 score (0-1)
    """
    if not candidate.strip() or not reference.strip():
        return 0.0
    
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    # Use smoothing for short sentences
    smoothing = SmoothingFunction().method1
    
    try:
        score = sentence_bleu(
            [ref_tokens],
            cand_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing
        )
    except Exception as e:
        logger.warning(f"Error computing BLEU: {e}")
        score = 0.0
    
    return score


def evaluate_predictions(predictions_file: str) -> Dict:
    """
    Evaluate predictions and return BLEU-4 metrics.
    
    Args:
        predictions_file: Path to predictions JSON file
        
    Returns:
        Dictionary with overall and per-type BLEU-4 scores
    """
    # Load predictions
    logger.info(f"Loading predictions from {predictions_file}")
    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    logger.info(f"Evaluating {len(predictions)} predictions...")
    
    # Prepare data
    references = []
    candidates = []
    types = []
    
    for pred in predictions:
        references.append(pred['ground_truth'])
        candidates.append(pred['top_prediction'])
        types.append(pred['spoiler_type'])
    
    # Compute BLEU-4 scores
    logger.info("Computing BLEU-4 scores...")
    bleu_scores = []
    for i, (ref, cand) in enumerate(zip(references, candidates)):
        score = compute_bleu4(ref, cand)
        bleu_scores.append(score)
        
        # Log progress every 100 examples
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(references)} examples")
    
    # Aggregate overall results
    results = {
        'overall': {
            'bleu4_mean': float(np.mean(bleu_scores)),
            'bleu4_std': float(np.std(bleu_scores)),
            'bleu4_min': float(np.min(bleu_scores)),
            'bleu4_max': float(np.max(bleu_scores)),
            'bleu4_median': float(np.median(bleu_scores)),
            'num_examples': len(predictions),
            'num_zero_scores': int(sum(1 for s in bleu_scores if s == 0.0)),
            'num_perfect_scores': int(sum(1 for s in bleu_scores if s == 1.0)),
        },
        'by_type': {},
        'individual_scores': bleu_scores  # Store individual scores for analysis
    }
    
    # Per-type metrics
    unique_types = sorted(set(types))
    for spoiler_type in unique_types:
        type_indices = [i for i, t in enumerate(types) if t == spoiler_type]
        type_scores = [bleu_scores[i] for i in type_indices]
        
        if type_indices:
            results['by_type'][spoiler_type] = {
                'bleu4_mean': float(np.mean(type_scores)),
                'bleu4_std': float(np.std(type_scores)),
                'bleu4_min': float(np.min(type_scores)),
                'bleu4_max': float(np.max(type_scores)),
                'bleu4_median': float(np.median(type_scores)),
                'num_examples': len(type_indices),
                'num_zero_scores': int(sum(1 for s in type_scores if s == 0.0)),
                'num_perfect_scores': int(sum(1 for s in type_scores if s == 1.0)),
            }
    
    return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print("\n" + "="*80)
    print("BLEU-4 EVALUATION RESULTS")
    print("="*80)
    
    # Overall performance
    print("\nOVERALL PERFORMANCE:")
    overall = results['overall']
    print(f"  BLEU-4 (Mean):       {overall['bleu4_mean']:.4f} ± {overall['bleu4_std']:.4f}")
    print(f"  BLEU-4 (Median):     {overall['bleu4_median']:.4f}")
    print(f"  BLEU-4 (Min):        {overall['bleu4_min']:.4f}")
    print(f"  BLEU-4 (Max):        {overall['bleu4_max']:.4f}")
    print(f"  Total Examples:      {overall['num_examples']}")
    print(f"  Zero Scores:         {overall['num_zero_scores']} ({overall['num_zero_scores']/overall['num_examples']*100:.1f}%)")
    print(f"  Perfect Scores:      {overall['num_perfect_scores']} ({overall['num_perfect_scores']/overall['num_examples']*100:.1f}%)")
    
    # Per-type performance
    print("\nPERFORMANCE BY SPOILER TYPE:")
    print(f"  {'Type':<12} {'Mean':<10} {'Std':<10} {'Median':<10} {'Examples':<10} {'Zero%':<10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for spoiler_type in ['phrase', 'passage', 'multi']:
        if spoiler_type in results['by_type']:
            metrics = results['by_type'][spoiler_type]
            zero_pct = metrics['num_zero_scores'] / metrics['num_examples'] * 100
            print(f"  {spoiler_type:<12} "
                  f"{metrics['bleu4_mean']:<10.4f} "
                  f"{metrics['bleu4_std']:<10.4f} "
                  f"{metrics['bleu4_median']:<10.4f} "
                  f"{metrics['num_examples']:<10} "
                  f"{zero_pct:<10.1f}")
    
    # Comparison with SemEval baseline
    print("\n" + "="*80)
    print("COMPARISON WITH SEMEVAL-2023 BASELINE")
    print("="*80)
    print(f"\n  {'Metric':<25} {'Baseline':<12} {'Ours':<12} {'Difference':<12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    
    baseline_scores = {
        'overall': 0.40,
        'phrase': 0.65,
        'passage': 0.24,
        'multi': 0.12,
    }
    
    our_overall = results['overall']['bleu4_mean']
    diff_overall = our_overall - baseline_scores['overall']
    status = "✓" if diff_overall >= 0 else "✗"
    print(f"  {'Overall':<25} {baseline_scores['overall']:<12.4f} {our_overall:<12.4f} {diff_overall:+.4f} {status}")
    
    for spoiler_type in ['phrase', 'passage', 'multi']:
        if spoiler_type in results['by_type']:
            our_score = results['by_type'][spoiler_type]['bleu4_mean']
            baseline_score = baseline_scores[spoiler_type]
            diff = our_score - baseline_score
            status = "✓" if diff >= 0 else "✗"
            print(f"  {spoiler_type.capitalize():<25} {baseline_score:<12.4f} {our_score:<12.4f} {diff:+.4f} {status}")
    
    print("\n" + "="*80)
    
    # Summary assessment
    print("\nSUMMARY:")
    our_overall = results['overall']['bleu4_mean']
    if our_overall >= 0.40:
        print("  ✓ Performance matches or exceeds SemEval-2023 baseline!")
    elif our_overall >= 0.35:
        print("  ⚠ Performance is close to baseline. Consider:")
        print("    - Using SQuAD pre-trained model (deepset/deberta-v3-base-squad2)")
        print("    - Training on full dataset (not subsampled)")
        print("    - Increasing number of training epochs")
    else:
        print("  ✗ Performance is below baseline. Action needed:")
        print("    - CRITICAL: Use SQuAD pre-trained checkpoint")
        print("    - Ensure training on full dataset")
        print("    - Check data preprocessing pipeline")
        print("    - Verify model is actually learning (check training loss)")
    
    # Identify weaknesses
    if 'multi' in results['by_type']:
        multi_score = results['by_type']['multi']['bleu4_mean']
        if multi_score < 0.15:
            print(f"\n  ⚠ Multi-part spoilers performing poorly (BLEU: {multi_score:.4f})")
            print("    → This motivates your hybrid refinement approach!")
    
    print("\n" + "="*80)


def print_worst_predictions(predictions_file: str, results: Dict, n: int = 5):
    """Print the worst predictions for error analysis."""
    # Load predictions
    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    # Get individual scores
    scores = results['individual_scores']
    
    # Find worst predictions
    scored_predictions = list(zip(predictions, scores))
    scored_predictions.sort(key=lambda x: x[1])  # Sort by score (ascending)
    
    print("\n" + "="*80)
    print(f"WORST {n} PREDICTIONS (Error Analysis)")
    print("="*80)
    
    for i, (pred, score) in enumerate(scored_predictions[:n], 1):
        print(f"\n{i}. BLEU-4: {score:.4f} | Type: {pred['spoiler_type']}")
        print(f"   Post: {pred['post_text'][:100]}...")
        print(f"   Predicted: {pred['top_prediction'][:150]}...")
        print(f"   Ground Truth: {pred['ground_truth'][:150]}...")
        print(f"   " + "-"*76)


def print_best_predictions(predictions_file: str, results: Dict, n: int = 5):
    """Print the best predictions."""
    # Load predictions
    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    # Get individual scores
    scores = results['individual_scores']
    
    # Find best predictions
    scored_predictions = list(zip(predictions, scores))
    scored_predictions.sort(key=lambda x: x[1], reverse=True)  # Sort by score (descending)
    
    print("\n" + "="*80)
    print(f"BEST {n} PREDICTIONS")
    print("="*80)
    
    for i, (pred, score) in enumerate(scored_predictions[:n], 1):
        print(f"\n{i}. BLEU-4: {score:.4f} | Type: {pred['spoiler_type']}")
        print(f"   Post: {pred['post_text'][:100]}...")
        print(f"   Predicted: {pred['top_prediction'][:150]}...")
        print(f"   Ground Truth: {pred['ground_truth'][:150]}...")
        print(f"   " + "-"*76)


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate clickbait spoiler predictions (BLEU-4 only)")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON (optional)"
    )
    parser.add_argument(
        "--show_examples",
        action="store_true",
        help="Show best and worst prediction examples"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=5,
        help="Number of examples to show (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Evaluate
    results = evaluate_predictions(args.predictions)
    
    # Print main results
    print_results(results)
    
    # Show examples if requested
    if args.show_examples:
        print_worst_predictions(args.predictions, results, n=args.n_examples)
        print_best_predictions(args.predictions, results, n=args.n_examples)
    
    # Save if requested
    if args.output:
        # Remove individual_scores before saving (can be large)
        save_results = {k: v for k, v in results.items() if k != 'individual_scores'}
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()