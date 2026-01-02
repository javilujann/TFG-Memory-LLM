"""
Context Processing Time Evaluator

Evaluates the time taken to process context for each question.
Computes statistics like mean, median, P95, P99, min, max processing time.
"""

from typing import List, Dict, Any
import numpy as np

from ..core.interfaces import Evaluator
from ..core.models import Question, Answer, EvaluationResult


class ContextProcessingTimeEvaluator(Evaluator):
    """
    Evaluator for analyzing context processing time.
    
    Computes various processing time statistics:
    - Mean processing time
    - Median processing time
    - Standard deviation
    - Min/Max processing time
    - Percentiles (P50, P95, P99)
    
    This helps evaluate the performance characteristics and efficiency
    of different memory systems when processing context.
    """
    
    def __init__(self):
        """Initialize context processing time evaluator"""
        self.config: Dict[str, Any] = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize context processing time evaluator.
        
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
        Evaluate a single question for context processing time.

        Args:
            question: The original question
            predicted_answer: The generated answer (unused but required by interface)

        Returns:
            Dictionary with processing time metrics
        """
        # Get processing time from question metadata
        processing_time = question.metadata.get('context_processing_time', None)
        
        return {
            "question_id": question.question_id,
            "processing_time": processing_time
        }
    
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        questions: List[Question],
    ) -> EvaluationResult:
        """
        Evaluate context processing time metrics.
        
        Args:
            results: List of question-answer evaluation results
            questions: List of questions (used for metadata)

        Returns:
            EvaluationResult with processing time statistics
        """
        
        # Extract processing times
        processing_times = []
        per_question_results = []
        question_type_times: Dict[str, List[float]] = {}
        
        for question, result in zip(questions, results):
            # Get processing time 
            proc_time = result.get('processing_time', None)
            
            if proc_time is not None and proc_time >= 0:
                processing_times.append(proc_time)
                
                # Track by question type
                q_type = question.question_type
                if q_type not in question_type_times:
                    question_type_times[q_type] = []
                question_type_times[q_type].append(proc_time)
                
                # Store per-question result
                per_question_results.append({
                    'question_id': question.question_id,
                    'question_type': q_type,
                    'processing_time': proc_time,
                    'time_unit': self.config.get('time_unit', 'seconds')
                })
        
        if not processing_times:
            # No valid processing times found
            return EvaluationResult(
                overall_metrics={
                    'error': 'No valid processing times found',
                    'total_questions': len(questions)
                },
                per_question_results=[],
                per_type_metrics={},
                evaluator_name='ContextProcessingTimeEvaluator',
                metadata={
                    'time_unit': self.config.get('time_unit', 'seconds'),
                    'error': 'No processing time data available'
                }
            )
    
        # Convert to numpy array for better performance
        times_np = np.array(processing_times)
        
        # Compute overall metrics
        overall_metrics = {
            'count': len(times_np),
            'mean': np.mean(times_np),
            'median': np.median(times_np),
            'min': np.min(times_np),
            'max': np.max(times_np),
        }
        
        # Add standard deviation if we have enough data
        if len(times_np) >= 2:
            overall_metrics['std_dev'] = np.std(times_np, ddof=1)
        
        # Compute percentiles
        for percentile in self.config.get('percentiles', [50, 95, 99]):
            p_values = np.percentile(times_np, percentile)
            overall_metrics[f'p{percentile}'] = p_values
        
        # Compute per-type metrics
        per_type_metrics = {}
        for q_type, type_times in question_type_times.items():
            if not type_times:
                continue

            # Convert to numpy array for better performance
            type_times_np = np.array(type_times)
            
            type_metrics = {
                'count': len(type_times_np),
                'mean': np.mean(type_times_np),
                'median': np.median(type_times_np),
                'min': np.min(type_times_np),
                'max': np.max(type_times_np),
            }
            
            # Add standard deviation if we have enough data
            if len(type_times_np) >= 2:
                type_metrics['std_dev'] = np.std(type_times_np, ddof=1)
            
            # Compute percentiles for this type
            for percentile in self.config.get('percentiles', [50, 95, 99]):
                p_values = np.percentile(type_times_np, percentile)
                type_metrics[f'p{percentile}'] = p_values

            per_type_metrics[q_type] = type_metrics
        
        # Build metadata
        metadata = {
            'time_unit': self.config.get('time_unit', 'seconds'),
            'percentiles_computed': self.config.get('percentiles', [50, 95, 99]),
            'total_questions': len(questions),
            'questions_with_processing_time': len(processing_times),
            'question_types': list(question_type_times.keys())
        }
        
        return EvaluationResult(
            overall_metrics=overall_metrics,
            per_question_results=per_question_results,
            per_type_metrics=per_type_metrics,
            evaluator_name='ContextProcessingTimeEvaluator',
            metadata=metadata
        )
    
    def get_name(self) -> str:
        """
        Get evaluator name.
        
        Returns:
            Evaluator name
        """
        return "ContextProcessingTimeEvaluator"
