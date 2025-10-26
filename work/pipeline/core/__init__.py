"""
Pipeline Core Module

This module contains the core interfaces and data models for the LLM Memory Evaluation Pipeline.
"""

from .interfaces import (
    DatasetReader,
    MemorySystem,
    LLMBackend,
    Evaluator,
)

from .models import (
    ChatTurn,
    Question,
    Answer,
    EvaluationResult,
    PipelineConfig,
)

from .pipeline import EvaluationPipeline

__all__ = [
    # Interfaces
    'DatasetReader',
    'MemorySystem',
    'LLMBackend',
    'Evaluator',
    # Models
    'ChatTurn',
    'Question',
    'Answer',
    'EvaluationResult',
    'PipelineConfig',
    # Pipeline
    'EvaluationPipeline',
]
