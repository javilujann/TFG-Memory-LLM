"""
Retrieval Precision Evaluators

These evaluators calculate precision, recall, and F1 scores for memory retrieval
at two levels:
1. Session-level: Whether retrieved memories come from the correct sessions
2. Turn-level: Whether retrieved memories come from the exact turns containing answers
"""

from typing import List, Dict, Any, Set, Tuple

from ..core.interfaces import Evaluator
from ..core.models import Question, Answer, EvaluationResult


class SessionPrecisionEvaluator(Evaluator):
    """
    Evaluator that calculates precision, recall, and F1 for session-level retrieval.
    
    Compares the session_ids of retrieved memories against answer_session_ids
    from the question metadata to determine:
    - Precision: What fraction of retrieved memories come from answer sessions
    - Recall: What fraction of answer sessions are represented in retrieved memories
    - F1: Harmonic mean of precision and recall
    """
    
    def __init__(self):
        """Initialize session precision evaluator."""
        self.config: Dict[str, Any] = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize session precision evaluator.
        
        Expected config:
            - memory_key: Key in answer.metadata to access retrieved memories (default: 'memoriesRetrieved')
        """
        self.config = config.copy()
    
    def _get_memories_from_answer(self, answer: Answer) -> List[Dict[str, Any]]:
        """
        Extract retrieved memories from answer metadata.
        
        Args:
            answer: Answer object with metadata containing memories
            
        Returns:
            List of memory dictionaries
        """
        memory_key = self.config.get('memory_key', 'memoriesRetrieved')
        memories = answer.metadata.get(memory_key, [])
        
        if not memories:
            return []
        
        # Ensure memories are dicts
        if isinstance(memories[0], dict):
            return memories
        
        return []
    
    def _extract_session_ids(self, memories: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract unique session IDs from retrieved memories.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            Set of session IDs
        """
        session_ids = set()
        for memory in memories:
            # Memory metadata should contain session_id
            if 'metadata' in memory and isinstance(memory['metadata'], dict):
                session_id = memory['metadata'].get('session_id')
                if session_id:
                    session_ids.add(session_id)
        
        return session_ids
    
    def _calculate_metrics(
        self,
        retrieved_sessions: Set[str],
        answer_sessions: Set[str],
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            retrieved_sessions: Set of session IDs from retrieved memories
            answer_sessions: Set of session IDs that contain answers
            
        Returns:
            Dict with precision, recall, and f1 scores
        """
        # Calculate true positives (intersection)
        true_positives = len(retrieved_sessions & answer_sessions)
        
        # Calculate precision
        if len(retrieved_sessions) > 0:
            precision = true_positives / len(retrieved_sessions)
        else:
            precision = 0.0
        
        # Calculate recall
        if len(answer_sessions) > 0:
            recall = true_positives / len(answer_sessions)
        else:
            recall = 0.0
        
        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def evaluate_single(
        self,
        question: Question,
        predicted_answer: Answer,
    ) -> Dict[str, Any]:
        """
        Evaluate session-level precision for a single answer.
        
        Args:
            question: Question with ground truth and metadata
            predicted_answer: Predicted answer with retrieved memories
            
        Returns:
            Dict with evaluation results including F1 score
        """
        # Extract memories from answer
        memories = self._get_memories_from_answer(predicted_answer)
        
        # If no memories, return zero scores
        if not memories:
            return {
                'question_id': question.question_id,
                'question_type': question.question_type,
                'evaluable': False,
                'num_memories': 0,
                'num_retrieved_sessions': 0,
                'num_answer_sessions': 0,
                'num_correct_sessions': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }
        
        # Extract session IDs from retrieved memories
        retrieved_sessions = self._extract_session_ids(memories)
        
        # Get answer session IDs from question metadata
        answer_session_ids = question.metadata.get('answer_session_ids', [])
        answer_sessions = set(answer_session_ids) if answer_session_ids else set()
        
        # Calculate metrics
        metrics = self._calculate_metrics(retrieved_sessions, answer_sessions)
        
        # Count correct sessions (true positives)
        correct_sessions = retrieved_sessions & answer_sessions
        
        return {
            'question_id': question.question_id,
            'question_type': question.question_type,
            'evaluable': True,
            'num_memories': len(memories),
            'num_retrieved_sessions': len(retrieved_sessions),
            'num_answer_sessions': len(answer_sessions),
            'num_correct_sessions': len(correct_sessions),
            'retrieved_sessions': sorted(list(retrieved_sessions)),
            'answer_sessions': sorted(list(answer_sessions)),
            'correct_sessions': sorted(list(correct_sessions)),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
        }
    
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        questions: List[Question],
    ) -> EvaluationResult:
        """
        Aggregate individual results into overall metrics.
        
        Args:
            results: List of per-question results
            questions: List of original questions
            
        Returns:
            EvaluationResult with aggregated metrics
        """
        if not results:
            return EvaluationResult(
                overall_metrics={},
                per_question_results=[],
                per_type_metrics={},
                evaluator_name=self.get_name()
            )
        
        # Filter evaluable results
        evaluable_results = [r for r in results if r.get('evaluable', True)]
        non_evaluable_count = len(results) - len(evaluable_results)
        
        if not evaluable_results:
            return EvaluationResult(
                overall_metrics={
                    'total_questions': len(results),
                    'evaluable_questions': 0,
                    'non_evaluable_questions': non_evaluable_count,
                },
                per_question_results=results,
                per_type_metrics={},
                evaluator_name=self.get_name()
            )
        
        # Calculate overall averages (macro)
        total_evaluable = len(evaluable_results)
        avg_precision = sum(r['precision'] for r in evaluable_results) / total_evaluable
        avg_recall = sum(r['recall'] for r in evaluable_results) / total_evaluable
        avg_f1 = sum(r['f1'] for r in evaluable_results) / total_evaluable
        
        # Calculate micro metrics (aggregated across all questions)
        total_retrieved_sessions = sum(r['num_retrieved_sessions'] for r in evaluable_results)
        total_answer_sessions = sum(r['num_answer_sessions'] for r in evaluable_results)
        total_correct_sessions = sum(r['num_correct_sessions'] for r in evaluable_results)
        
        micro_precision = (total_correct_sessions / total_retrieved_sessions) if total_retrieved_sessions > 0 else 0.0
        micro_recall = (total_correct_sessions / total_answer_sessions) if total_answer_sessions > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0
        
        overall_metrics = {
            'macro_f1': avg_f1,
            'macro_precision': avg_precision,
            'macro_recall': avg_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'total_retrieved_sessions': total_retrieved_sessions,
            'total_answer_sessions': total_answer_sessions,
            'total_correct_sessions': total_correct_sessions,
            'evaluable_questions': total_evaluable,
            'non_evaluable_questions': non_evaluable_count,
            'total_questions': len(results),
        }
        
        # Per-type metrics
        type_stats: Dict[str, List[Dict[str, Any]]] = {}
        for result in evaluable_results:
            q_type = result.get('question_type', 'unknown')
            if q_type not in type_stats:
                type_stats[q_type] = []
            type_stats[q_type].append(result)
        
        per_type_metrics = {}
        for q_type, type_results in type_stats.items():
            n = len(type_results)
            
            # Macro metrics (average per question)
            type_macro_f1 = sum(r['f1'] for r in type_results) / n
            type_macro_precision = sum(r['precision'] for r in type_results) / n
            type_macro_recall = sum(r['recall'] for r in type_results) / n
            
            # Micro metrics (aggregated)
            type_retrieved = sum(r['num_retrieved_sessions'] for r in type_results)
            type_answer = sum(r['num_answer_sessions'] for r in type_results)
            type_correct = sum(r['num_correct_sessions'] for r in type_results)
            
            type_micro_precision = (type_correct / type_retrieved) if type_retrieved > 0 else 0.0
            type_micro_recall = (type_correct / type_answer) if type_answer > 0 else 0.0
            type_micro_f1 = (2 * type_micro_precision * type_micro_recall / (type_micro_precision + type_micro_recall)) if (type_micro_precision + type_micro_recall) > 0 else 0.0
            
            per_type_metrics[q_type] = {
                'macro_f1': type_macro_f1,
                'macro_precision': type_macro_precision,
                'macro_recall': type_macro_recall,
                'micro_f1': type_micro_f1,
                'micro_precision': type_micro_precision,
                'micro_recall': type_micro_recall,
                'total_questions': n,
            }
        
        return EvaluationResult(
            overall_metrics=overall_metrics,
            per_question_results=results,
            per_type_metrics=per_type_metrics,
            evaluator_name=self.get_name(),
            metadata={
                'question_types': list(type_stats.keys()),
                'memory_key': self.config.get('memory_key', 'memoriesRetrieved'),
            }
        )
    
    def get_name(self) -> str:
        """Get evaluator name"""
        return "SessionPrecisionEvaluator"


class TurnPrecisionEvaluator(Evaluator):
    """
    Evaluator that calculates precision, recall, and F1 for turn-level retrieval.
    
    Compares the (session_id, turn_index) pairs of retrieved memories against
    answer_locations from the question metadata to determine if memories come
    from the exact turns that contain answers.
    
    Note: Both memory metadata and answer_locations use pair indices (turn_idx // 2)
    to represent user-assistant pairs, ensuring direct comparison.
    """
    
    def __init__(self):
        """Initialize turn precision evaluator."""
        self.config: Dict[str, Any] = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize turn precision evaluator.
        
        Expected config:
            - memory_key: Key in answer.metadata to access retrieved memories (default: 'memoriesRetrieved')
        """
        self.config = config.copy()
    
    def _get_memories_from_answer(self, answer: Answer) -> List[Dict[str, Any]]:
        """
        Extract retrieved memories from answer metadata.
        
        Args:
            answer: Answer object with metadata containing memories
            
        Returns:
            List of memory dictionaries
        """
        memory_key = self.config.get('memory_key', 'memoriesRetrieved')
        memories = answer.metadata.get(memory_key, [])
        
        if not memories:
            return []
        
        # Ensure memories are dicts
        if isinstance(memories[0], dict):
            return memories
        
        return []
    
    def _extract_turn_locations(self, memories: List[Dict[str, Any]]) -> Set[Tuple[str, int]]:
        """
        Extract (session_id, turn_index) pairs from retrieved memories.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            Set of (session_id, turn_index) tuples
        """
        turn_locations = set()
        for memory in memories:
            # Memory metadata should contain session_id and turn_index
            if 'metadata' in memory and isinstance(memory['metadata'], dict):
                session_id = memory['metadata'].get('session_id')
                turn_index = memory['metadata'].get('turn_index')
                if session_id is not None and turn_index is not None:
                    turn_locations.add((session_id, turn_index))
        
        return turn_locations
    
    def _calculate_metrics(
        self,
        retrieved_turns: Set[Tuple[str, int]],
        answer_turns: Set[Tuple[str, int]],
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            retrieved_turns: Set of (session_id, turn_index) from retrieved memories
            answer_turns: Set of (session_id, turn_index) that contain answers
            
        Returns:
            Dict with precision, recall, and f1 scores
        """
        # Calculate true positives (intersection)
        true_positives = len(retrieved_turns & answer_turns)
        
        # Calculate precision
        if len(retrieved_turns) > 0:
            precision = true_positives / len(retrieved_turns)
        else:
            precision = 0.0
        
        # Calculate recall
        if len(answer_turns) > 0:
            recall = true_positives / len(answer_turns)
        else:
            recall = 0.0
        
        # Calculate F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def evaluate_single(
        self,
        question: Question,
        predicted_answer: Answer,
    ) -> Dict[str, Any]:
        """
        Evaluate turn-level precision for a single answer.
        
        Args:
            question: Question with ground truth and metadata
            predicted_answer: Predicted answer with retrieved memories
            
        Returns:
            Dict with evaluation results including F1 score
        """
        # Extract memories from answer
        memories = self._get_memories_from_answer(predicted_answer)
        
        # If no memories, return zero scores
        if not memories:
            return {
                'question_id': question.question_id,
                'question_type': question.question_type,
                'evaluable': False,
                'num_memories': 0,
                'num_retrieved_turns': 0,
                'num_answer_turns': 0,
                'num_correct_turns': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }
        
        # Extract turn locations from retrieved memories
        retrieved_turns = self._extract_turn_locations(memories)
        
        # Get answer locations from question metadata (already in pair index format)
        answer_locations = question.metadata.get('answer_locations', [])
        answer_turns = set(answer_locations) if answer_locations else set()
        
        # Calculate metrics
        metrics = self._calculate_metrics(retrieved_turns, answer_turns)
        
        # Count correct turns (true positives)
        correct_turns = retrieved_turns & answer_turns
        
        return {
            'question_id': question.question_id,
            'question_type': question.question_type,
            'evaluable': True,
            'num_memories': len(memories),
            'num_retrieved_turns': len(retrieved_turns),
            'num_answer_turns': len(answer_turns),
            'num_correct_turns': len(correct_turns),
            'retrieved_turns': sorted(list(retrieved_turns)),
            'answer_turns': sorted(list(answer_turns)),
            'correct_turns': sorted(list(correct_turns)),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
        }
    
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        questions: List[Question],
    ) -> EvaluationResult:
        """
        Aggregate individual results into overall metrics.
        
        Args:
            results: List of per-question results
            questions: List of original questions
            
        Returns:
            EvaluationResult with aggregated metrics
        """
        if not results:
            return EvaluationResult(
                overall_metrics={},
                per_question_results=[],
                per_type_metrics={},
                evaluator_name=self.get_name()
            )
        
        # Filter evaluable results
        evaluable_results = [r for r in results if r.get('evaluable', True)]
        non_evaluable_count = len(results) - len(evaluable_results)
        
        if not evaluable_results:
            return EvaluationResult(
                overall_metrics={
                    'total_questions': len(results),
                    'evaluable_questions': 0,
                    'non_evaluable_questions': non_evaluable_count,
                },
                per_question_results=results,
                per_type_metrics={},
                evaluator_name=self.get_name()
            )
        
        # Calculate overall averages (macro)
        total_evaluable = len(evaluable_results)
        avg_precision = sum(r['precision'] for r in evaluable_results) / total_evaluable
        avg_recall = sum(r['recall'] for r in evaluable_results) / total_evaluable
        avg_f1 = sum(r['f1'] for r in evaluable_results) / total_evaluable
        
        # Calculate micro metrics (aggregated across all questions)
        total_retrieved_turns = sum(r['num_retrieved_turns'] for r in evaluable_results)
        total_answer_turns = sum(r['num_answer_turns'] for r in evaluable_results)
        total_correct_turns = sum(r['num_correct_turns'] for r in evaluable_results)
        
        micro_precision = (total_correct_turns / total_retrieved_turns) if total_retrieved_turns > 0 else 0.0
        micro_recall = (total_correct_turns / total_answer_turns) if total_answer_turns > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0
        
        overall_metrics = {
            'macro_f1': avg_f1,
            'macro_precision': avg_precision,
            'macro_recall': avg_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'total_retrieved_turns': total_retrieved_turns,
            'total_answer_turns': total_answer_turns,
            'total_correct_turns': total_correct_turns,
            'evaluable_questions': total_evaluable,
            'non_evaluable_questions': non_evaluable_count,
            'total_questions': len(results),
        }
        
        # Per-type metrics
        type_stats: Dict[str, List[Dict[str, Any]]] = {}
        for result in evaluable_results:
            q_type = result.get('question_type', 'unknown')
            if q_type not in type_stats:
                type_stats[q_type] = []
            type_stats[q_type].append(result)
        
        per_type_metrics = {}
        for q_type, type_results in type_stats.items():
            n = len(type_results)
            
            # Macro metrics (average per question)
            type_macro_f1 = sum(r['f1'] for r in type_results) / n
            type_macro_precision = sum(r['precision'] for r in type_results) / n
            type_macro_recall = sum(r['recall'] for r in type_results) / n
            
            # Micro metrics (aggregated)
            type_retrieved = sum(r['num_retrieved_turns'] for r in type_results)
            type_answer = sum(r['num_answer_turns'] for r in type_results)
            type_correct = sum(r['num_correct_turns'] for r in type_results)
            
            type_micro_precision = (type_correct / type_retrieved) if type_retrieved > 0 else 0.0
            type_micro_recall = (type_correct / type_answer) if type_answer > 0 else 0.0
            type_micro_f1 = (2 * type_micro_precision * type_micro_recall / (type_micro_precision + type_micro_recall)) if (type_micro_precision + type_micro_recall) > 0 else 0.0
            
            per_type_metrics[q_type] = {
                'macro_f1': type_macro_f1,
                'macro_precision': type_macro_precision,
                'macro_recall': type_macro_recall,
                'micro_f1': type_micro_f1,
                'micro_precision': type_micro_precision,
                'micro_recall': type_micro_recall,
                'total_questions': n,
            }
        
        return EvaluationResult(
            overall_metrics=overall_metrics,
            per_question_results=results,
            per_type_metrics=per_type_metrics,
            evaluator_name=self.get_name(),
            metadata={
                'question_types': list(type_stats.keys()),
                'memory_key': self.config.get('memory_key', 'memoriesRetrieved'),
            }
        )
    
    def get_name(self) -> str:
        """Get evaluator name"""
        return "TurnPrecisionEvaluator"
