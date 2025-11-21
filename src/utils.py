"""
Utility functions for data loading, processing, and model inference.
"""

import json
import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def extract_answer_span(
    context: str,
    start_logits: List[float],
    end_logits: List[float],
    tokenizer,
    tokens: List[str],
    max_answer_length: int = 100,
    confidence_threshold: float = 0.0,
) -> Tuple[str, float, Tuple[int, int]]:
    """
    Extract answer span from context using start and end logits.
    
    Args:
        context: The context text
        start_logits: Start position logits
        end_logits: End position logits
        tokenizer: The tokenizer
        tokens: List of tokens
        max_answer_length: Maximum length of answer span
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        Tuple of (answer_text, confidence, (start_idx, end_idx))
    """
    # Get top predictions
    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits)
    
    # Ensure start < end and respects max length
    if end_idx < start_idx:
        end_idx = start_idx
    
    if end_idx - start_idx + 1 > max_answer_length:
        end_idx = start_idx + max_answer_length - 1
    
    # Get confidence as average of max logits
    confidence = (start_logits[start_idx] + end_logits[end_idx]) / 2
    
    # Convert token indices to character positions
    answer_tokens = tokens[start_idx:end_idx + 1]
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    return answer_text, confidence, (start_idx, end_idx)


def get_spoiler_type(tags: List[str]) -> str:
    """
    Determine spoiler type from tags.
    
    Returns:
        One of: 'phrase', 'passage', 'multipart'
    """
    if not tags:
        return "phrase"  # default
    
    for tag in tags:
        if tag in ["phrase", "passage", "multipart"]:
            return tag
    
    return "phrase"  # default


def calculate_char_positions(
    paragraph_idx: int,
    char_start: int,
    char_end: int,
    paragraphs: List[str],
) -> Tuple[int, int]:
    """
    Convert paragraph-relative character positions to full context positions.
    
    Args:
        paragraph_idx: Index of the paragraph
        char_start: Character start position within paragraph
        char_end: Character end position within paragraph
        paragraphs: List of all paragraphs
    
    Returns:
        Tuple of (context_char_start, context_char_end)
    """
    # Calculate offset from concatenating previous paragraphs
    context_offset = sum(len(paragraphs[i]) + 1 for i in range(paragraph_idx))  # +1 for space
    
    return context_offset + char_start, context_offset + char_end
