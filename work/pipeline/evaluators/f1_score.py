"""
F1 Score Evaluator

Token-level F1 score evaluation.
"""

from typing import List, Dict, Any

from ..core.interfaces import Evaluator
from ..core.models import Question, Answer, EvaluationResult


class F1ScoreEvaluator(Evaluator):
    """
    Evaluator using token-level F1 score.
    
    Computes precision, recall, and F1 based on token overlap.
    """
    
    def __init__(self):
        # TODO: Initialize evaluator
        self.config = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize F1 evaluator.
        
        Expected config:
            - tokenizer: Tokenization method ('whitespace', 'nltk', etc.)
            - lowercase: Whether to lowercase tokens
            - remove_stopwords: Whether to remove stopwords
        """
        # TODO: Implement initialization
        pass
    
    def evaluate_single(
        self,
        question: Question,
        predicted_answer: Answer,
    ) -> Dict[str, Any]:
        """
        Evaluate using F1 score.
        
        Args:
            question: Question with ground truth
            predicted_answer: Predicted answer
            
        Returns:
            Dict with precision, recall, and F1 scores
        """
        # TODO: Implement evaluation
        # 1. Tokenize ground truth and prediction
        # 2. Calculate overlap
        # 3. Compute precision, recall, F1
        # 4. Return results
        pass
    
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        questions: List[Question],
    ) -> EvaluationResult:
        """Aggregate F1 scores"""
        # TODO: Implement aggregation
        # Calculate average precision, recall, F1
        pass
    
    def get_name(self) -> str:
        """Get evaluator name"""
        return "F1Score"
