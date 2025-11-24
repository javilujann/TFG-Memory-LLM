"""
Data Models for the Evaluation Pipeline

This module contains all the data classes used throughout the pipeline
to ensure standardized data flow between components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class ChatTurn:
    """
    Represents a single turn in a conversation.
    
    Attributes:
        role: The role of the speaker ('user' or 'assistant')
        content: The text content of the turn
        has_answer: Whether this turn contains the answer to the question (optional)
        timestamp: When this turn occurred (optional)
        metadata: Additional metadata for this turn
    """
    role: str
    content: str
    has_answer: Optional[bool] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate role"""
        if self.role not in ['user', 'assistant', 'system']:
            raise ValueError(f"Invalid role: {self.role}. Must be 'user', 'assistant', or 'system'")


@dataclass
class Question:
    """
    Standardized representation of a question with context and ground truth.
    
    This is the output format from DatasetReader implementations, ensuring
    that all datasets are normalized to the same structure.
    
    Attributes:
        question_id: Unique identifier for this question
        question_text: The actual question text
        question_type: Type/category of the question (e.g., 'temporal-reasoning', 'knowledge-update')
        ground_truth_answer: The correct answer
        context: The context needed to answer the question (can be single or multiple sessions)
        metadata: Additional information (dates, session_ids, difficulty, etc.)
    """
    question_id: str
    question_text: str
    question_type: str
    ground_truth_answer: str
    context: List[List[ChatTurn]]  # List of sessions, each session is a list of turns
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_turns(self) -> List[ChatTurn]:
        """Flatten all sessions into a single list of turns"""
        return [turn for session in self.context for turn in session]
    
    def get_context_size(self) -> int:
        """Get total number of turns across all sessions"""
        return len(self.get_all_turns())


@dataclass
class Answer:
    """
    Standardized answer format from memory systems.
    
    Attributes:
        question_id: ID of the question being answered
        answer_text: The generated answer text
        confidence: Confidence score if available (0-1)
        processing_time: Time taken to generate answer in seconds
        metadata: Additional information (tokens used, retrieval results, etc.)
    """
    question_id: str
    answer_text: str
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'question_id': self.question_id,
            'hypothesis': self.answer_text,  # Using 'hypothesis' to match LongMemEval format
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'metadata': self.metadata,
        }


@dataclass
class EvaluationResult:
    """
    Aggregated evaluation results from one or more evaluators.
    
    Attributes:
        overall_metrics: Overall metrics across all questions (e.g., {'accuracy': 0.85})
        per_question_results: Detailed results for each question
        per_type_metrics: Metrics broken down by question type
        evaluator_name: Name of the evaluator that produced these results
        timestamp: When the evaluation was performed
        metadata: Additional information about the evaluation run
    """
    overall_metrics: Dict[str, float]
    per_question_results: List[Dict[str, Any]]
    per_type_metrics: Dict[str, Dict[str, float]]
    evaluator_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'overall_metrics': self.overall_metrics,
            'per_question_results': self.per_question_results,
            'per_type_metrics': self.per_type_metrics,
            'evaluator_name': self.evaluator_name,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }
    
    def summary(self) -> str:
        """Get a human-readable summary of results"""
        lines = [f"\n{'='*60}"]
        lines.append(f"Evaluation Results - {self.evaluator_name}")
        lines.append(f"{'='*60}")
        lines.append("\nOverall Metrics:")
        for metric, value in self.overall_metrics.items():
            lines.append(f"  {metric}: {value:.4f}")
        
        if self.per_type_metrics:
            lines.append("\nPer-Type Metrics:")
            for qtype, metrics in self.per_type_metrics.items():
                lines.append(f"  {qtype}:")
                for metric, value in metrics.items():
                    lines.append(f"    {metric}: {value:.4f}")
        
        lines.append(f"\nTotal Questions: {len(self.per_question_results)}")
        lines.append(f"Timestamp: {self.timestamp}")
        lines.append(f"{'='*60}\n")
        return '\n'.join(lines)


@dataclass
class PipelineConfig:
    """
    Configuration for the entire evaluation pipeline.
    
    This can be loaded from YAML/JSON files for reproducible experiments.
    
    Attributes:
        experiment_name: Name for this experiment run
        dataset_config: Configuration for the dataset reader
        memory_system_config: Configuration for the memory system
        llm_config: Configuration for the LLM backend (if separate)
        evaluation_config: Configuration for evaluators
        output_config: Configuration for output handling (where to save, formats, etc.)
        process_context_only: If True, only load dataset and process context without answering questions
    """
    experiment_name: str
    dataset_config: Dict[str, Any]
    memory_system_config: Dict[str, Any]
    llm_config: Dict[str, Any] = field(default_factory=dict)
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    output_config: Dict[str, Any] = field(default_factory=dict)
    process_context_only: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'experiment_name': self.experiment_name,
            'dataset_config': self.dataset_config,
            'memory_system_config': self.memory_system_config,
            'llm_config': self.llm_config,
            'evaluation_config': self.evaluation_config,
            'output_config': self.output_config,
            'process_context_only': self.process_context_only,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary"""
        return cls(**data)
