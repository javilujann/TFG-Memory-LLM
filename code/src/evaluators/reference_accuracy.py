"""
Reference Accuracy Evaluator using LLM

This evaluator calculates the F1 score of reference accuracy for extracted memories
using an LLM. It splits the LLM usage into two parts:
1. Evaluate whether each memory is relevant to the answer (True Positive vs False Positive)
2. Evaluate whether the set of memories is sufficient to answer the question (detecting False Negatives)
"""

from typing import List, Dict, Any, Optional

from ..core.interfaces import Evaluator, LLMBackend
from ..core.models import Question, Answer, EvaluationResult


class ReferenceAccuracyEvaluator(Evaluator):
    """
    Evaluator that calculates F1 score for reference accuracy using LLM.
    
    This evaluator uses an LLM to:
    1. Judge if each retrieved memory is relevant to the answer (TP/FP classification)
    2. Judge if the set of memories is sufficient to answer the question (FN estimation)
    
    The F1 score is then calculated from:
    - True Positives: Relevant memories retrieved
    - False Positives: Irrelevant memories retrieved
    - False Negatives: Derived from sufficiency score (1 - sufficiency = FN ratio)
    """
    
    # Prompt templates for LLM evaluation
    RELEVANCE_PROMPT_TEMPLATE = """You are evaluating whether a piece of retrieved memory is relevant for answering a specific question.

Question: {question}

Ground Truth Answer: {answer}

Retrieved Memory: {memory}

Is this memory relevant for answering the question correctly? Consider a memory relevant if it contains information that would help answer the question or is directly related to the answer.

Answer ONLY with "yes" or "no"."""

    SUFFICIENCY_PROMPT_TEMPLATE = """You are evaluating whether a set of retrieved memories provides sufficient information to answer a question correctly.

Question: {question}

Ground Truth Answer: {answer}

Retrieved Memories:
{memories}

On a scale from 0 to 1, how sufficient is this set of memories to answer the question correctly?
- 1.0 means the memories contain all necessary information to answer the question
- 0.5 means the memories contain about half of the necessary information
- 0.0 means the memories contain no useful information for answering the question

Consider both the completeness and relevance of the information.

Answer ONLY with a number between 0 and 1 (e.g., 0.8, 0.5, 0.2)."""
    
    def __init__(self, judge_backend: LLMBackend):
        """
        Initialize with LLM backend for judging.
        
        Args:
            judge_backend: LLM backend to use for evaluation
        """
        self.judge_backend = judge_backend
        self.config: Dict[str, Any] = {}
        self.relevance_prompt_template: str = ""
        self.sufficiency_prompt_template: str = ""
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize reference accuracy evaluator.
        
        Expected config:
            - relevance_prompt_template: Custom template for relevance evaluation (optional)
            - sufficiency_prompt_template: Custom template for sufficiency evaluation (optional)
            - temperature: Temperature for judge LLM (default: 0.0)
            - max_tokens_relevance: Max tokens for relevance response (default: 10)
            - max_tokens_sufficiency: Max tokens for sufficiency response (default: 20)
            - memory_key: Key in answer.metadata to access retrieved memories (default: 'memoriesRetrieved')
        
        Raises:
            RuntimeError: If judge backend is not set
        """
        if self.judge_backend is None:
            raise RuntimeError("Judge backend not set")
        
        self.config = config.copy()
        
        # Use custom templates if provided, otherwise use defaults
        self.relevance_prompt_template = config.get(
            'relevance_prompt_template',
            self.RELEVANCE_PROMPT_TEMPLATE
        )
        self.sufficiency_prompt_template = config.get(
            'sufficiency_prompt_template',
            self.SUFFICIENCY_PROMPT_TEMPLATE
        )
    
    def _get_memories_from_answer(self, answer: Answer) -> List[str]:
        """
        Extract retrieved memories from answer metadata.
        
        Args:
            answer: Answer object with metadata containing memories
            
        Returns:
            List of memory strings
        """
        memory_key = self.config.get('memory_key', 'memoriesRetrieved')
        memories = answer.metadata.get(memory_key, [])
        
        # Handle different memory formats
        if not memories:
            return []
        
        # If memories are strings, return as-is
        if isinstance(memories[0], str):
            return memories
        
        # If memories are dicts, extract text field
        if isinstance(memories[0], dict):
            # Try common keys for memory text
            for key in ['text', 'content', 'memory', 'message']:
                if key in memories[0]:
                    return [mem.get(key, '') for mem in memories]
            # If no standard key found, convert to string
            return [str(mem) for mem in memories]
        
        # Default: convert to string
        return [str(mem) for mem in memories]
    
    def _evaluate_memory_relevance(
        self,
        question: Question,
        memory: str,
    ) -> bool:
        """
        Evaluate if a single memory is relevant for answering the question.
        
        Args:
            question: Question with ground truth
            memory: Memory string to evaluate
            
        Returns:
            True if memory is relevant, False otherwise
        """
        # Format the prompt
        prompt = self.relevance_prompt_template.format(
            question=question.question_text,
            answer=question.ground_truth_answer,
            memory=memory
        )
        
        try:
            # Get judgment from LLM
            response = self.judge_backend.generate(
                prompt,
                temperature=self.config.get('temperature', 0.0),
                max_tokens=self.config.get('max_tokens_relevance', 10)
            )
            
            # Parse yes/no
            response_lower = response.lower().strip()
            if 'yes' in response_lower:
                return True
            elif 'no' in response_lower:
                return False
            else:
                # If unclear, be conservative and mark as irrelevant
                return False
                
        except Exception as e:
            # If evaluation fails, mark as irrelevant
            print(f"⚠️ Error evaluating memory relevance: {e}")
            return False
    
    def _evaluate_sufficiency(
        self,
        question: Question,
        memories: List[str],
    ) -> float:
        """
        Evaluate if the set of memories is sufficient to answer the question.
        
        Args:
            question: Question with ground truth
            memories: List of memory strings
            
        Returns:
            Sufficiency score between 0 and 1
        """
        # Format memories for prompt
        if not memories:
            memories_text = "(No memories retrieved)"
        else:
            memories_text = "\n".join([f"{i+1}. {mem}" for i, mem in enumerate(memories)])
        
        # Format the prompt
        prompt = self.sufficiency_prompt_template.format(
            question=question.question_text,
            answer=question.ground_truth_answer,
            memories=memories_text
        )
        
        try:
            # Get judgment from LLM
            response = self.judge_backend.generate(
                prompt,
                temperature=self.config.get('temperature', 0.0),
                max_tokens=self.config.get('max_tokens_sufficiency', 20)
            )
            
            # Parse numeric value
            response_clean = response.strip()
            # Extract first number found
            import re
            numbers = re.findall(r'0?\.\d+|1\.0|1|0', response_clean)
            
            if numbers:
                score = float(numbers[0])
                # Ensure score is in [0, 1]
                score = max(0.0, min(1.0, score))
                return score
            else:
                # If no number found, return 0
                print(f"⚠️ Could not parse sufficiency score from: {response}")
                return 0.0
                
        except Exception as e:
            # If evaluation fails, return 0
            print(f"⚠️ Error evaluating sufficiency: {e}")
            return 0.0
    
    def _calculate_f1(
        self,
        true_positives: int,
        false_positives: int,
        false_negatives: float,
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            true_positives: Number of relevant memories retrieved
            false_positives: Number of irrelevant memories retrieved
            false_negatives: Number of missing relevant information (can be fractional)
            
        Returns:
            Dict with precision, recall, and f1 scores
        """
        # Calculate precision
        total_retrieved = true_positives + false_positives
        if total_retrieved > 0:
            precision = true_positives / total_retrieved
        else:
            precision = 0.0
        
        # Calculate recall
        total_relevant = true_positives + false_negatives
        if total_relevant > 0:
            recall = true_positives / total_relevant
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
        Evaluate reference accuracy for a single answer.
        
        Args:
            question: Question with ground truth
            predicted_answer: Predicted answer with retrieved memories
            
        Returns:
            Dict with evaluation results including F1 score.
            If no memories are retrieved (e.g., baseline systems), returns
            a result marked as not evaluable for reference accuracy metrics.
        """
        # Extract memories from answer
        memories = self._get_memories_from_answer(predicted_answer)
        
        # If no memories, mark as not evaluable for reference accuracy
        # This is appropriate for baseline systems that don't use memory retrieval
        if not memories:
            return {
                'question_id': question.question_id,
                'question_type': question.question_type,
                'evaluable': False,
                'num_memories': 0,
                # Don't include F1 metrics for non-retrieval systems
            }
        
        # Step 1: Evaluate relevance of each memory (TP vs FP)
        relevance_judgments = []
        for memory in memories:
            is_relevant = self._evaluate_memory_relevance(question, memory)
            relevance_judgments.append(is_relevant)
        
        true_positives = sum(relevance_judgments)
        false_positives = len(relevance_judgments) - true_positives
        
        # Step 2: Evaluate sufficiency of all memories (to estimate FN)
        sufficiency_score = self._evaluate_sufficiency(question, memories)
        
        # Convert sufficiency to false negatives
        # If sufficiency is 1.0, FN = 0 (we have everything)
        # If sufficiency is 0.0, FN = TP (we're missing as much as we have)
        # Linear interpolation: FN = TP * (1 - sufficiency)
        if true_positives > 0:
            false_negatives = true_positives * (1 - sufficiency_score)
        else:
            # If no relevant memories, FN is at least 1
            false_negatives = max(1.0, 1.0 - sufficiency_score)
        
        # Calculate F1 metrics
        metrics = self._calculate_f1(true_positives, false_positives, false_negatives)
        
        # Prepare result
        result = {
            'question_id': question.question_id,
            'question_type': question.question_type,
            'evaluable': True,
            'num_memories': len(memories),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'sufficiency_score': sufficiency_score,
            'relevance_judgments': relevance_judgments,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
        }
        
        return result
    
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
        
        # Filter evaluable results (only those with memory retrieval)
        evaluable_results = [r for r in results if r.get('evaluable', True)]
        non_evaluable_count = len(results) - len(evaluable_results)
        
        # If no evaluable results, return empty metrics
        if not evaluable_results:
            return EvaluationResult(
                overall_metrics={
                    'total_questions': len(results),
                    'evaluable_questions': 0,
                    'non_evaluable_questions': non_evaluable_count,
                },
                per_question_results=results,
                per_type_metrics={},
                evaluator_name=self.get_name(),
                metadata={
                    'judge_backend': self.judge_backend.get_model_info().get('model', 'unknown'),
                    'memory_key': self.config.get('memory_key', 'retrieved_memories'),
                }
            )
        
        # Calculate overall averages (only for evaluable results)
        total_evaluable = len(evaluable_results)
        avg_precision = sum(r['precision'] for r in evaluable_results) / total_evaluable
        avg_recall = sum(r['recall'] for r in evaluable_results) / total_evaluable
        avg_f1 = sum(r['f1'] for r in evaluable_results) / total_evaluable
        avg_sufficiency = sum(r['sufficiency_score'] for r in evaluable_results) / total_evaluable
        
        total_tp = sum(r['true_positives'] for r in evaluable_results)
        total_fp = sum(r['false_positives'] for r in evaluable_results)
        total_fn = sum(r['false_negatives'] for r in evaluable_results)
        
        # Calculate macro F1 (average of per-question F1s)
        macro_f1 = avg_f1
        
        # Calculate micro F1 (F1 over all aggregated TPs, FPs, FNs)
        micro_metrics = self._calculate_f1(total_tp, total_fp, total_fn)
        micro_f1 = micro_metrics['f1']
        micro_precision = micro_metrics['precision']
        micro_recall = micro_metrics['recall']
        
        overall_metrics = {
            'macro_f1': macro_f1,
            'macro_precision': avg_precision,
            'macro_recall': avg_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'avg_sufficiency_score': avg_sufficiency,
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'evaluable_questions': total_evaluable,
            'non_evaluable_questions': non_evaluable_count,
            'total_questions': len(results),
        }
        
        # Per-type metrics (only for evaluable results)
        type_stats: Dict[str, List[Dict[str, Any]]] = {}
        for result in evaluable_results:
            q_type = result.get('question_type', 'unknown')
            if q_type not in type_stats:
                type_stats[q_type] = []
            type_stats[q_type].append(result)
        
        per_type_metrics = {}
        for q_type, type_results in type_stats.items():
            n = len(type_results)
            type_tp = sum(r['true_positives'] for r in type_results)
            type_fp = sum(r['false_positives'] for r in type_results)
            type_fn = sum(r['false_negatives'] for r in type_results)
            
            type_metrics = self._calculate_f1(type_tp, type_fp, type_fn)
            
            per_type_metrics[q_type] = {
                'macro_f1': sum(r['f1'] for r in type_results) / n,
                'macro_precision': sum(r['precision'] for r in type_results) / n,
                'macro_recall': sum(r['recall'] for r in type_results) / n,
                'micro_f1': type_metrics['f1'],
                'micro_precision': type_metrics['precision'],
                'micro_recall': type_metrics['recall'],
                'avg_sufficiency_score': sum(r['sufficiency_score'] for r in type_results) / n,
                'total_questions': n,
            }
        
        return EvaluationResult(
            overall_metrics=overall_metrics,
            per_question_results=results,
            per_type_metrics=per_type_metrics,
            evaluator_name=self.get_name(),
            metadata={
                'judge_backend': self.judge_backend.get_model_info().get('model', 'unknown'),
                'question_types': list(type_stats.keys()),
                'memory_key': self.config.get('memory_key', 'retrieved_memories'),
            }
        )
    
    def get_name(self) -> str:
        """Get evaluator name"""
        model = self.judge_backend.get_model_info().get('model', 'unknown')
        return f"ReferenceAccuracyEvaluator-{model}"
