"""
Inference module for extractive model.
Handles prediction on new clickbait posts.
"""

import logging
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger(__name__)


class ExtractivePredictor:
    """Predictor for extractive spoiler generation."""
    
    def __init__(
        self,
        model_path: str,
        max_seq_length: int = 512,
        max_answer_length: int = 100,
        confidence_threshold: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            max_seq_length: Maximum sequence length for tokenization
            max_answer_length: Maximum length of extracted span
            confidence_threshold: Minimum confidence to return prediction
            device: Device to load model on
        """
        self.device = device
        self.max_seq_length = max_seq_length
        self.max_answer_length = max_answer_length
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"Loading model from {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Attempting with trust_remote_code=True...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        logger.info(f"Model loaded on {device}")
    
    def predict(
        self,
        question: str,
        context: str,
        return_all_predictions: bool = False,
        top_k: int = 1,
    ) -> List[Dict[str, any]]:
        """
        Predict spoiler spans for a given question and context.
        
        Args:
            question: The clickbait post text (treated as question)
            context: The article text (treated as context)
            return_all_predictions: Whether to return all predictions or just top-1
            top_k: Number of top predictions to return
        
        Returns:
            List of predictions with format:
            {
                "answer": str,
                "confidence": float,
                "start_logit": float,
                "end_logit": float,
                "start_index": int,
                "end_index": int,
            }
        """
        # Tokenize with proper handling of single examples (not batch)
        # Ensure question and context are strings, not lists
        if isinstance(question, (list, tuple)):
            question = question[0] if question else ""
        if isinstance(context, (list, tuple)):
            context = context[0] if context else ""
        
        inputs = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        
        offset_mapping = inputs.pop("offset_mapping")
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        # Get predictions
        predictions = []
        
        # For each token position, calculate confidence
        for sample_idx in range(len(start_logits)):
            sample_start_logits = start_logits[sample_idx]
            sample_end_logits = end_logits[sample_idx]
            
            # Get top-k predictions
            start_indices = torch.argsort(sample_start_logits, descending=True)[:top_k]
            end_indices = torch.argsort(sample_end_logits, descending=True)[:top_k]
            
            for start_idx in start_indices:
                for end_idx in end_indices:
                    # Convert tensor to int if needed
                    start_idx_int = start_idx.item() if hasattr(start_idx, 'item') else int(start_idx)
                    end_idx_int = end_idx.item() if hasattr(end_idx, 'item') else int(end_idx)
                    
                    # Ensure valid span
                    if end_idx_int < start_idx_int or end_idx_int - start_idx_int + 1 > self.max_answer_length:
                        continue
                    
                    # Get confidence
                    confidence = (
                        sample_start_logits[start_idx_int].item() + 
                        sample_end_logits[end_idx_int].item()
                    ) / 2
                    
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Get character positions
                    try:
                        char_start = offset_mapping[sample_idx][start_idx_int][0].item()
                        char_end = offset_mapping[sample_idx][end_idx_int][1].item()
                        answer = context[char_start:char_end]
                        
                        predictions.append({
                            "answer": answer,
                            "confidence": confidence,
                            "start_logit": sample_start_logits[start_idx_int].item(),
                            "end_logit": sample_end_logits[end_idx_int].item(),
                            "start_index": start_idx_int,
                            "end_index": end_idx_int,
                            "char_start": char_start,
                            "char_end": char_end,
                        })
                    except (IndexError, ValueError):
                        continue
        
        # Sort by confidence
        predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
        
        # Return top-k or all
        if return_all_predictions:
            return predictions
        else:
            return predictions[:top_k] if predictions else []
    
    def batch_predict(
        self,
        samples: List[Dict[str, str]],
        top_k: int = 1,
    ) -> List[List[Dict[str, any]]]:
        """
        Predict on a batch of samples.
        
        Args:
            samples: List of dicts with 'question' and 'context' keys
            top_k: Number of top predictions per sample
        
        Returns:
            List of prediction lists
        """
        all_predictions = []
        for sample in samples:
            predictions = self.predict(
                sample["question"],
                sample["context"],
                top_k=top_k,
            )
            all_predictions.append(predictions)
        return all_predictions
