"""
LLM Memory Evaluation Pipeline

A modular pipeline for evaluating different memory systems with Large Language Models.

Main components:
- Core: Abstract interfaces and data models
- Readers: Dataset loaders
- Backends: LLM providers
- Memory Systems: Different memory strategies
- Evaluators: Answer evaluation methods
- Utils: Helper functions and utilities
"""

from .core import (
    DatasetReader,
    MemorySystem,
    LLMBackend,
    Evaluator,
    Question,
    Answer,
    EvaluationResult,
    PipelineConfig,
    EvaluationPipeline,
)

__version__ = "0.1.0"

__all__ = [
    # Core interfaces
    'DatasetReader',
    'MemorySystem',
    'LLMBackend',
    'Evaluator',
    # Data models
    'Question',
    'Answer',
    'EvaluationResult',
    'PipelineConfig',
    # Pipeline
    'EvaluationPipeline',
]
