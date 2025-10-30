"""
Latency Evaluator

Evaluates the latency/performance metrics of answer generation.
Computes statistics like mean, median, P95, P99, min, max latency.
"""

from typing import List, Dict, Any
import numpy as np

from ..core.interfaces import Evaluator
from ..core.models import Question, Answer, EvaluationResult


class LatencyEvaluator(Evaluator):
    """
    Evaluator for analyzing answer generation latency.
    
    Computes various latency statistics:
    - Mean latency
    - Median latency
    - Standard deviation
    - Min/Max latency
    - Percentiles (P50, P95, P99)
    
    This helps evaluate the performance characteristics and consistency
    of different memory systems and LLM backends.
    """
    
    def __init__(self):
        """Initialize latency evaluator"""
        self.config: Dict[str, Any] = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize latency evaluator.
        
        Expected config:
            - percentiles: List of percentiles to compute (default: [50, 95, 99])
          
        Args:
            config: Configuration dictionary
        """
        self.config = config.copy()
        
        # Set defaults
        if 'percentiles' not in self.config:
            self.config['percentiles'] = [50, 95, 99]
        
    
    def evaluate_single(self, question: Question, predicted_answer: Answer) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair for latency.

        Args:
            question: The original question
            predicted_answer: The generated answer

        Returns:
            Dictionary with latency metrics
        """
        # Get latency information from the predicted answer
        return {
            "question_id": question.question_id,
            "latency": predicted_answer.processing_time
        }
    
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        questions: List[Question],
    ) -> EvaluationResult:
        """
        Evaluate latency metrics for generated answers.
        
        Args:
            results: List of question-answer evaluation results
            questions: List of questions (used for metadata)

        Returns:
            EvaluationResult with latency statistics
        """
        
        # Extract latencies from answers
        latencies = []
        per_question_results = []
        question_type_latencies: Dict[str, List[float]] = {}
        
        for question, result in zip(questions, results):
            # Get generation time 
            latency = result.get('latency', None)
            
            if latency is not None and latency > 0:
                latencies.append(latency)
                
                # Track by question type
                q_type = question.question_type
                if q_type not in question_type_latencies:
                    question_type_latencies[q_type] = []
                question_type_latencies[q_type].append(latency)
                
                # Store per-question result
                per_question_results.append({
                    'question_id': question.question_id,
                    'question_type': q_type,
                    'latency': latency,
                    'latency_unit': self.config.get('time_unit', 'seconds')
                })
        
        if not latencies:
            # No valid latencies found
            return EvaluationResult(
                overall_metrics={
                    'error': 'No valid latencies found',
                    'total_questions': len(questions)
                },
                per_question_results=[],
                per_type_metrics={},
                evaluator_name='LatencyEvaluator',
                metadata={
                    'time_unit': self.config.get('time_unit', 'seconds'),
                    'error': 'No latency data available'
                }
            )
    
        # Convert to numpy array for better performance
        latencies_np = np.array(latencies)
        
        # Compute overall metrics
        overall_metrics = {
            'count': len(latencies_np),
            'mean': np.mean(latencies_np),
            'median': np.median(latencies_np),
            'min': np.min(latencies_np),
            'max': np.max(latencies_np),
        }
        
        # Add standard deviation if we have enough data
        if len(latencies_np) >= 2:
            overall_metrics['std_dev'] = np.std(latencies_np, ddof=1)
        
        # Compute percentiles
        for percentile in self.config.get('percentiles', [50, 95, 99]):
            p_values = np.percentile(latencies_np, percentile)
            overall_metrics[f'p{percentile}'] = p_values
        
        # Compute per-type metrics
        per_type_metrics = {}
        for q_type, type_latencies in question_type_latencies.items():
            if not type_latencies:
                continue

            # Convert to numpy array for better performance
            type_latencies_np = np.array(type_latencies)
            
            type_metrics = {
                'count': len(type_latencies_np),
                'mean': np.mean(type_latencies_np),
                'median': np.median(type_latencies_np),
                'min': np.min(type_latencies_np),
                'max': np.max(type_latencies_np),
            }
            
            # Add standard deviation if we have enough data
            if len(type_latencies_np) >= 2:
                type_metrics['std_dev'] = np.std(type_latencies_np, ddof=1)
            
            # Compute percentiles for this type
            for percentile in self.config.get('percentiles', [50, 95, 99]):
                p_values = np.percentile(type_latencies_np, percentile)
                type_metrics[f'p{percentile}'] = p_values

            per_type_metrics[q_type] = type_metrics
        
        # Build metadata
        metadata = {
            'time_unit': self.config.get('time_unit', 'seconds'),
            'percentiles_computed': self.config.get('percentiles', [50, 95, 99]),
            'total_questions': len(questions),
            'questions_with_latency': len(latencies),
            'question_types': list(question_type_latencies.keys())
        }
        
        return EvaluationResult(
            overall_metrics=overall_metrics,
            per_question_results=per_question_results,
            per_type_metrics=per_type_metrics,
            evaluator_name='LatencyEvaluator',
            metadata=metadata
        )
    
    def get_name(self) -> str:
        """
        Get evaluator name.
        
        Returns:
            Evaluator name
        """
        return "LatencyEvaluator"
