"""
F1 Score Evaluator

Token-level F1 score evaluation comparing predicted answers with ground truth.
Uses token overlap to compute precision, recall, and F1 score.
"""

from typing import List, Dict, Any, Set
import re
from collections import Counter

from ..core.interfaces import Evaluator
from ..core.models import Question, Answer, EvaluationResult


class F1ScoreEvaluator(Evaluator):
    """
    Evaluator using token-level F1 score.
    
    Computes precision, recall, and F1 based on token overlap between
    predicted answer and ground truth answer.
    
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    Precision = (# common tokens) / (# predicted tokens)
    Recall = (# common tokens) / (# ground truth tokens)
    """
    
    def __init__(self):
        """Initialize F1 evaluator with default config"""
        self.config: Dict[str, Any] = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': False,
        }
        self._stopwords: Set[str] = set()
    
    def _get_stopwords(self) -> Set[str]:
        """Get English stopwords"""
        # Common English stopwords
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'you', 'we', 'they', 'this',
            'what', 'which', 'who', 'where', 'when', 'why', 'how'
        }
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize F1 evaluator.
        
        Expected config:
            - tokenizer: Tokenization method ('whitespace', 'simple') - default: 'whitespace'
            - lowercase: Whether to lowercase tokens - default: True
            - remove_punctuation: Whether to remove punctuation - default: True
            - remove_stopwords: Whether to remove stopwords - default: False
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        
        # Load stopwords if needed, default to False
        if self.config.get('remove_stopwords'):
            self._stopwords = self._get_stopwords()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text before tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Lowercase
        if self.config.get('lowercase', True):
            text = text.lower()
        
        # Remove punctuation
        if self.config.get('remove_punctuation', True):
            text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into list of tokens.
        
        Args:
            text: Input text (should be normalized first)
            
        Returns:
            List of tokens
        """
        # Default to whitespace
        tokens = text.split()
        
        # Remove stopwords if configured, default to False
        if self.config.get('remove_stopwords'):
            tokens = [t for t in tokens if t not in self._stopwords]
        
        # Remove empty tokens
        tokens = [t for t in tokens if t]
        
        return tokens
    
    def _compute_f1(self, prediction_tokens: List[str], ground_truth_tokens: List[str]) -> Dict[str, float]:
        """
        Compute F1 score between prediction and ground truth tokens.
        
        Args:
            prediction_tokens: List of predicted tokens
            ground_truth_tokens: List of ground truth tokens
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        if not prediction_tokens or not ground_truth_tokens:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'common_tokens': 0,
                'predicted_tokens': len(prediction_tokens),
                'ground_truth_tokens': len(ground_truth_tokens)
            }
        
        # Use Counter to handle repeated tokens
        pred_counter = Counter(prediction_tokens)
        gt_counter = Counter(ground_truth_tokens)
        
        # Find common tokens (intersection)
        common = pred_counter & gt_counter  # Min counts for each token
        num_common = sum(common.values())
        
        # Calculate metrics
        num_predicted = len(prediction_tokens)
        num_ground_truth = len(ground_truth_tokens)
        
        precision = num_common / num_predicted if num_predicted > 0 else 0.0
        recall = num_common / num_ground_truth if num_ground_truth > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'common_tokens': num_common,
            'predicted_tokens': num_predicted,
            'ground_truth_tokens': num_ground_truth
        }
    
    def evaluate_single(
        self,
        question: Question,
        predicted_answer: Answer,
    ) -> Dict[str, Any]:
        """
        Evaluate using F1 score.
        
        Args:
            question: Question with ground truth answer
            predicted_answer: Predicted answer
            
        Returns:
            Dict with precision, recall, F1 scores, and token counts
        """
        # Get texts
        prediction_text = predicted_answer.answer_text
        ground_truth_text = question.ground_truth_answer
        
        # Normalize texts
        prediction_normalized = self._normalize_text(prediction_text)
        ground_truth_normalized = self._normalize_text(ground_truth_text)
        
        # Tokenize
        prediction_tokens = self._tokenize(prediction_normalized)
        ground_truth_tokens = self._tokenize(ground_truth_normalized)
        
        # Compute F1 score
        f1_results = self._compute_f1(prediction_tokens, ground_truth_tokens)
        
        # Add metadata
        result = {
            'question_id': question.question_id,
            'question_type': question.question_type,
            'precision': f1_results['precision'],
            'recall': f1_results['recall'],
            'f1': f1_results['f1'],
            'common_tokens': f1_results['common_tokens'],
            'predicted_tokens': f1_results['predicted_tokens'],
            'ground_truth_tokens': f1_results['ground_truth_tokens'],
            'prediction_text': prediction_text,
            'ground_truth_text': ground_truth_text,
        }
        
        return result
    
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        questions: List[Question],
    ) -> EvaluationResult:
        """
        Aggregate F1 scores across all questions.
        
        Computes average precision, recall, and F1 scores overall and per question type.
        
        Args:
            results: List of individual evaluation results
            questions: List of questions evaluated
            
        Returns:
            EvaluationResult with aggregated metrics
        """
        if not results:
            return EvaluationResult(
                evaluator_name=self.get_name(),
                overall_metrics={'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                per_type_metrics={},
                individual_results=[],
                metadata={}
            )
        
        # Overall metrics
        total_precision = sum(r['precision'] for r in results)
        total_recall = sum(r['recall'] for r in results)
        total_f1 = sum(r['f1'] for r in results)
        total_count = len(results)
        
        avg_precision = total_precision / total_count
        avg_recall = total_recall / total_count
        avg_f1 = total_f1 / total_count
        
        overall_metrics = {
            'f1': avg_f1,
            'precision': avg_precision,
            'recall': avg_recall,
            'total_questions': total_count,
        }
        
        # Per-type metrics
        type_results: Dict[str, List[Dict[str, Any]]] = {}
        for result in results:
            q_type = result['question_type']
            if q_type not in type_results:
                type_results[q_type] = []
            type_results[q_type].append(result)
        
        per_type_metrics: Dict[str, Dict[str, float]] = {}
        for q_type, type_res in type_results.items():
            type_count = len(type_res)
            type_precision = sum(r['precision'] for r in type_res) / type_count
            type_recall = sum(r['recall'] for r in type_res) / type_count
            type_f1 = sum(r['f1'] for r in type_res) / type_count
            
            per_type_metrics[q_type] = {
                'f1': type_f1,
                'precision': type_precision,
                'recall': type_recall,
                'total': type_count,
            }
        
        # Metadata
        metadata = {
            'tokenizer': self.config.get('tokenizer', 'whitespace'),
            'lowercase': self.config.get('lowercase', True),
            'remove_punctuation': self.config.get('remove_punctuation', True),
            'remove_stopwords': self.config.get('remove_stopwords', False),
            'question_types': list(type_results.keys()),
        }
        
        return EvaluationResult(
            evaluator_name=self.get_name(),
            overall_metrics=overall_metrics,
            per_type_metrics=per_type_metrics,
            individual_results=results,
            metadata=metadata
        )
    
    def get_name(self) -> str:
        """Get evaluator name"""
        return "F1Score"
