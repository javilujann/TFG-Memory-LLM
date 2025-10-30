"""
Abstract Interfaces for Pipeline Components

This module defines the interfaces (abstract base classes) that all pipeline
components must implement. This enables easy swapping of implementations
for datasets, memory systems, LLMs, and evaluators.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import Question, Answer, ChatTurn, EvaluationResult


class DatasetReader(ABC):
    """
    Abstract base class for reading different datasets.
    
    Each dataset format (LongMemEval, custom formats, etc.) should implement
    this interface to provide questions in a standardized format.
    
    Example implementations:
        - LongMemEvalReader
        - CustomJSONReader
        - CSVReader
    """
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the dataset reader with configuration.
        
        Args:
            config: Configuration dictionary with reader-specific settings
        """
        pass
    
    @abstractmethod
    def load(self) -> List[Question]:
        """
        Load dataset from path and return list of Question objects.
        
        Args:
            path: Path to the dataset file
            max_questions: Maximum number of questions to load (None for all)
            
        Returns:
            List of Question objects with standardized format
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset format is invalid
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the dataset.
        
        Returns:
            Dictionary with metadata like:
            - name: Dataset name
            - size: Number of questions
            - question_types: List of question types
            - source: Where the dataset came from
            - version: Dataset version
        """
        pass
    
    def validate_question(self, question: Question) -> bool:
        """
        Validate that a question has all required fields.
        
        Can be overridden by subclasses for custom validation.
        
        Args:
            question: Question to validate
            
        Returns:
            True if valid, False otherwise
        """
        return (
            question.question_id and
            question.question_text and
            question.ground_truth_answer and
            len(question.context) > 0
        )


class LLMBackend(ABC):
    """
    Abstract base class for different LLM providers.
    
    This allows the pipeline to work with different LLM backends
    (Ollama, OpenAI, vLLM, etc.) through a unified interface.
    
    Example implementations:
        - OllamaBackend
        - OpenAIBackend
        - VLLMBackend
        - HuggingFaceBackend
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the LLM backend with configuration.
        
        Args:
            config: Configuration dictionary with backend-specific settings
                   (e.g., model name, host, API key, temperature, etc.)
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM given a prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If generation fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information (name, version, parameters, etc.)
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if the backend is available and ready to use.
        
        Returns:
            True if backend can be used, False otherwise
        """
        try:
            self.generate("test", max_tokens=1)
            return True
        except Exception:
            return False


class MemorySystem(ABC):
    """
    Abstract base class for LLM + Memory systems.
    
    A memory system handles how context/chat history is processed, stored,
    and used to answer questions. It may use an external LLMBackend or
    handle LLM calls internally.
    
    Example implementations:
        - FullContextMemorySystem (baseline: feed all context)
        - Mem0MemorySystem (using mem0 library)
        - RAGMemorySystem (retrieval-augmented generation)
        - SummaryMemorySystem (context summarization)
        - MCPMemorySystem (memory via Model Context Protocol)
    """
    
    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        """
        Initialize memory system.
        
        Args:
            llm_backend: Optional LLM backend to use. If None, the memory
                        system should handle LLM calls internally.
        """
        self.llm_backend = llm_backend
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the memory system with configuration.
        
        Args:
            config: Configuration dictionary with system-specific settings
        """
        pass
    
    @abstractmethod
    def process_context(self, context: List[List[ChatTurn]]) -> None:
        """
        Process and store context for later retrieval.
        
        This is where the memory system does its work: storing, indexing,
        summarizing, or otherwise processing the chat history.
        
        Args:
            context: List of sessions, where each session is a list of ChatTurns
        """
        pass
    
    @abstractmethod
    def answer_question(self, question: str, question_id: str) -> Answer:
        """
        Generate an answer to the question using the processed context.
        
        This method should use the memory system to retrieve relevant context
        and generate an answer (either using self.llm_backend or internally).
        
        Args:
            question: The question text
            question_id: Unique identifier for the question
            
        Returns:
            Answer object with the generated response
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Clear the memory system for the next question/session.
        
        This ensures that each question starts with a clean slate
        (unless the system is designed to maintain cross-question memory).
        """
        pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current memory state.
        
        Returns:
            Dictionary with memory statistics (size, retrievals, etc.)
            Default implementation returns empty dict.
        """
        return {}


class Evaluator(ABC):
    """
    Abstract base class for evaluation strategies.
    
    Evaluators compare predicted answers with ground truth and compute metrics.
    Multiple evaluators can be used simultaneously in the pipeline.
    
    Example implementations:
        - LLMAsJudgeEvaluator (uses GPT/other LLM to judge correctness)
        - ExactMatchEvaluator (simple string matching)
        - F1ScoreEvaluator (token-level F1 score)
        - ROUGEEvaluator (ROUGE metrics)
        - SemanticSimilarityEvaluator (embedding-based similarity)
        - CompositeEvaluator (combines multiple evaluators)
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Configuration dictionary with evaluator-specific settings
        """
        pass
    
    @abstractmethod
    def evaluate_single(
        self,
        question: Question,
        predicted_answer: Answer,
    ) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair.
        
        Args:
            question: The Question object (contains ground truth)
            predicted_answer: The Answer object from the memory system
            
        Returns:
            Dictionary with metrics and metadata for this question.
            Example: {
                'question_id': 'q1',
                'correct': True,
                'score': 0.95,
                'explanation': 'Answer matches ground truth',
                # ... any other metrics this evaluator computes
            }
        """
        pass
    
    @abstractmethod
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        questions: List[Question],
    ) -> EvaluationResult:
        """
        Aggregate individual results into overall metrics.
        
        Args:
            results: List of per-question results from evaluate_single()
            questions: List of original Question objects for context
            
        Returns:
            EvaluationResult object with aggregated metrics
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this evaluator.
        
        Returns:
            Evaluator name (e.g., 'LLM-as-Judge-GPT4', 'ExactMatch', 'F1')
        """
        pass