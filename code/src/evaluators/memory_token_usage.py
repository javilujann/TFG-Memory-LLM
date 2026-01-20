"""
Memory Token Usage Evaluator

Evaluates the number of tokens used in the context/prompt for each question.
For memory systems: counts tokens in retrieved memories.
For full context systems: counts tokens in the entire chat history.
Uses tiktoken to count tokens based on the embedder model configuration.
"""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    import tiktoken
except ImportError:
    tiktoken = None

from ..core.interfaces import Evaluator
from ..core.models import Question, Answer, EvaluationResult


class MemoryTokenUsageEvaluator(Evaluator):
    """
    Evaluator for analyzing token usage in context/prompts.
    
    Computes various token usage statistics:
    - Mean token count
    - Median token count
    - Standard deviation
    - Min/Max token count
    - Percentiles (P50, P95, P99)
    - Total tokens used across all questions
    
    For memory systems: counts tokens in retrieved memories.
    For full context systems: counts tokens in the entire chat history.
    This provides a fair comparison of context usage across all systems.
    """
    
    def __init__(self):
        """Initialize memory token usage evaluator"""
        self.config: Dict[str, Any] = {}
        self.encoding: Optional[Any] = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize memory token usage evaluator.
        
        Expected config:
            - embedder_model: Model name to use for token counting (required)
                              Can be tiktoken model name or OpenAI model name
            - memory_key: Key in answer.metadata to access retrieved memories (default: 'memoriesRetrieved')
            - percentiles: List of percentiles to compute (default: [50, 95, 99])
            - count_relations: Whether to also count tokens in relations (default: False)
          
        Args:
            config: Configuration dictionary
            
        Raises:
            ImportError: If tiktoken is not installed
            ValueError: If embedder_model is not provided
        """
        if tiktoken is None:
            raise ImportError("tiktoken is required for MemoryTokenUsageEvaluator. Install with: pip install tiktoken")
        
        self.config = config.copy()
        
        # Set defaults
        if 'percentiles' not in self.config:
            self.config['percentiles'] = [50, 95, 99]
        
        if 'memory_key' not in self.config:
            self.config['memory_key'] = 'memoriesRetrieved'
        
        if 'count_relations' not in self.config:
            self.config['count_relations'] = False
        
        # Get embedder model
        embedder_model = self.config.get('embedder_model')
        if not embedder_model:
            raise ValueError("embedder_model is required in config for token counting")
        
        # Initialize tiktoken encoding
        # Try to get encoding for the model, fallback to cl100k_base for OpenAI models
        try:
            self.encoding = tiktoken.encoding_for_model(embedder_model)
        except KeyError:
            # If model not found, try common encodings
            # For most embedding models, cl100k_base is a good approximation
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                raise ValueError(f"Could not initialize tiktoken encoding for model '{embedder_model}': {e}")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text or not isinstance(text, str):
            return 0
        
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            print(f"Warning: Failed to count tokens: {e}")
            return 0
    
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
        if isinstance(memories, list) and len(memories) > 0:
            if isinstance(memories[0], dict):
                return memories
        
        return []
    
    def _get_relations_from_answer(self, answer: Answer) -> List[Dict[str, Any]]:
        """
        Extract retrieved relations from answer metadata.
        
        Args:
            answer: Answer object with metadata containing relations
            
        Returns:
            List of relation dictionaries
        """
        relations = answer.metadata.get('relationsRetrieved', [])
        
        if not relations:
            return []
        
        # Ensure relations are dicts
        if isinstance(relations, list) and len(relations) > 0:
            if isinstance(relations[0], dict):
                return relations
        
        return []
    
    def evaluate_single(self, question: Question, predicted_answer: Answer) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair for prompt token usage.
        Counts tokens in the full prompt sent to the LLM.

        Args:
            question: The original question
            predicted_answer: The generated answer with memory metadata

        Returns:
            Dictionary with token usage metrics
        """
        # Get the full prompt from answer metadata
        full_prompt = predicted_answer.metadata.get('full_prompt', '')
        
        if not full_prompt:
            # Fallback: no prompt stored
            return {
                "question_id": question.question_id,
                "prompt_tokens": 0,
                "error": "No full_prompt found in metadata"
            }
        
        # Count tokens in the full prompt
        prompt_tokens = self._count_tokens(full_prompt)
        
        # Get number of memories for additional context
        memories = self._get_memories_from_answer(predicted_answer)
        num_memories = len(memories)
        
        return {
            "question_id": question.question_id,
            "prompt_tokens": prompt_tokens,
            "num_memories": num_memories,
        }
    
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        questions: List[Question],
    ) -> EvaluationResult:
        """
        Evaluate memory token usage metrics.
        
        Args:
            results: List of question-answer evaluation results
            questions: List of questions (used for metadata)

        Returns:
            EvaluationResult with token usage statistics
        """
        
        # Extract token counts
        prompt_tokens_list = []
        per_question_results = []
        question_type_tokens: Dict[str, List[int]] = {}
        
        for question, result in zip(questions, results):
            prompt_tokens = result.get('prompt_tokens', 0)
            num_memories = result.get('num_memories', 0)
            
            prompt_tokens_list.append(prompt_tokens)
            
            # Track by question type
            q_type = question.question_type
            if q_type not in question_type_tokens:
                question_type_tokens[q_type] = []
            question_type_tokens[q_type].append(prompt_tokens)
            
            # Store per-question result
            per_question_results.append({
                'question_id': question.question_id,
                'question_type': q_type,
                'prompt_tokens': prompt_tokens,
                'num_memories': num_memories,
            })
        
        if not prompt_tokens_list:
            # No valid token counts found
            return EvaluationResult(
                overall_metrics={
                    'error': 'No valid token counts found',
                    'total_questions': len(questions)
                },
                per_question_results=[],
                per_type_metrics={},
                evaluator_name='MemoryTokenUsageEvaluator',
                metadata={
                    'embedder_model': self.config.get('embedder_model'),
                    'error': 'No token data available'
                }
            )
        
        # Convert to numpy arrays for better performance
        prompt_tokens_np = np.array(prompt_tokens_list)
        
        # Compute overall metrics (convert numpy types to native Python types)
        overall_metrics = {
            'count': len(prompt_tokens_np),
            'prompt_tokens_sum': int(np.sum(prompt_tokens_np)),
            'mean_tokens': float(np.mean(prompt_tokens_np)),
            'median_tokens': float(np.median(prompt_tokens_np)),
            'min_tokens': int(np.min(prompt_tokens_np)),
            'max_tokens': int(np.max(prompt_tokens_np)),
        }
        
        # Add standard deviation if we have enough data
        if len(prompt_tokens_np) >= 2:
            overall_metrics['std_dev'] = float(np.std(prompt_tokens_np, ddof=1))
        
        # Compute percentiles
        for percentile in self.config.get('percentiles', [50, 95, 99]):
            p_values = np.percentile(prompt_tokens_np, percentile)
            overall_metrics[f'p{percentile}'] = float(p_values)
        
        # Compute per-type metrics
        per_type_metrics = {}
        for q_type, type_tokens in question_type_tokens.items():
            if not type_tokens:
                continue

            # Convert to numpy array for better performance
            type_tokens_np = np.array(type_tokens)
            
            type_metrics = {
                'count': len(type_tokens_np),
                'prompt_tokens_sum': int(np.sum(type_tokens_np)),
                'mean_tokens': float(np.mean(type_tokens_np)),
                'median_tokens': float(np.median(type_tokens_np)),
                'min_tokens': int(np.min(type_tokens_np)),
                'max_tokens': int(np.max(type_tokens_np)),
            }
            
            # Add standard deviation if we have enough data
            if len(type_tokens_np) >= 2:
                type_metrics['std_dev'] = float(np.std(type_tokens_np, ddof=1))
            
            # Compute percentiles for this type
            for percentile in self.config.get('percentiles', [50, 95, 99]):
                p_values = np.percentile(type_tokens_np, percentile)
                type_metrics[f'p{percentile}'] = float(p_values)

            per_type_metrics[q_type] = type_metrics
        
        # Build metadata
        metadata = {
            'embedder_model': self.config.get('embedder_model'),
            'memory_key': self.config.get('memory_key', 'memoriesRetrieved'),
            'count_relations': self.config.get('count_relations', False),
            'percentiles_computed': self.config.get('percentiles', [50, 95, 99]),
            'total_questions': len(questions),
            'question_types': list(question_type_tokens.keys())
        }
        
        return EvaluationResult(
            overall_metrics=overall_metrics,
            per_question_results=per_question_results,
            per_type_metrics=per_type_metrics,
            evaluator_name='MemoryTokenUsageEvaluator',
            metadata=metadata
        )
    
    def get_name(self) -> str:
        """
        Get evaluator name.
        
        Returns:
            Evaluator name
        """
        return "MemoryTokenUsageEvaluator"
