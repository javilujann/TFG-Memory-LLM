"""
Memory Audit Evaluator

This evaluator retrieves and displays memories for audit purposes.
Compares retrieved memories with all available memories for a question.
"""

from typing import List, Dict, Any, Optional

from ..core.interfaces import Evaluator, MemorySystem
from ..core.models import Question, Answer, EvaluationResult


class MemoryAuditEvaluator(Evaluator):
    """
    Evaluator that audits and displays memories for comparison.
    
    For each question, it:
    1. Gets all memories for that question from the memory system
    2. Extracts memories retrieved during answer generation
    3. Displays both sets for manual comparison
    """
    
    def __init__(self, memory_system: MemorySystem):
        """
        Initialize with memory system reference.
        
        Args:
            memory_system: The MemorySystem instance to audit
        """
        self.memory_system = memory_system
        self.config: Dict[str, Any] = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the memory audit evaluator.
        
        Expected config:
            - memory_key: Key in answer.metadata for retrieved memories (default: 'memoriesRetrieved')
            - filters: Additional filters to apply when retrieving all memories (optional)
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.copy()
    
    def evaluate_single(
        self,
        question: Question,
        predicted_answer: Answer,
    ) -> Dict[str, Any]:
        """
        Evaluate and display memories for a single question.
        
        Args:
            question: Question with ground truth
            predicted_answer: Answer with retrieved memories
        
        Returns:
            Dict with question info and memory lists
        """
        memory_key = self.config.get('memory_key', 'memoriesRetrieved')
        
        # Get retrieved memories from answer
        retrieved = predicted_answer.metadata.get(memory_key, [])
        if isinstance(retrieved, str):
            # Handle case where it's a formatted string instead of list
            retrieved = []
        
        # Build filters for get_all()
        filters = {'run_id': question.question_id}
        
        # Add additional filters if provided
        additional_filters = self.config.get('filters', {})
        if additional_filters:
            filters.update(additional_filters)
        
        try:
            # Get all memories for this question from memory system
            all_memories = self.memory_system.get_all(filters=filters)
        except Exception as e:
            print(f"⚠️ Warning: Could not retrieve all memories: {e}")
            all_memories = []
        
        # Prepare result
        result = {
            'question_id': question.question_id,
            'question_type': question.question_type,
            'retrieved_count': len(retrieved),
            'total_count': len(all_memories),
            'retrieved_memories': retrieved,
            'all_memories': all_memories,
        }
        
        return result
    
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        questions: List[Question],
    ) -> EvaluationResult:
        """
        Aggregate audit results across multiple questions.
        
        Args:
            results: List of evaluation results from evaluate_single
            questions: List of Question objects corresponding to the results
            
        Returns:
            EvaluationResult with aggregated audit statistics
        """
        if not results:
            return EvaluationResult(
                overall_metrics={},
                per_question_results=[],
                per_type_metrics={},
                evaluator_name=self.get_name(),
                metadata={}
            )
        
        total_retrieved = sum(r.get('retrieved_count', 0) for r in results)
        total_all = sum(r.get('total_count', 0) for r in results)
        
        # Calculate per-type metrics
        per_type_metrics: Dict[str, Dict[str, Any]] = {}
        for question, result in zip(questions, results):
            q_type = question.question_type
            if q_type not in per_type_metrics:
                per_type_metrics[q_type] = {
                    'count': 0,
                    'total_retrieved': 0,
                    'total_all': 0,
                }
            per_type_metrics[q_type]['count'] += 1
            per_type_metrics[q_type]['total_retrieved'] += result.get('retrieved_count', 0)
            per_type_metrics[q_type]['total_all'] += result.get('total_count', 0)
        
        overall_metrics = {
            'total_questions': len(results),
            'total_memories_retrieved': total_retrieved,
            'total_memories_all': total_all,
        }
        
        return EvaluationResult(
            overall_metrics=overall_metrics,
            per_question_results=results,
            per_type_metrics=per_type_metrics,
            evaluator_name=self.get_name(),
            metadata={
                'memory_key': self.config.get('memory_key', 'memoriesRetrieved'),
                'haystack_filters': self.config.get('haystack_filters', {}),
            }
        )

    def get_name(self) -> str:
        """
        Get evaluator name.
        
        Returns:
            Evaluator name
        """
        return "MemoryAuditEvaluator"