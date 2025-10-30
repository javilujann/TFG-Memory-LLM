"""
Configuration Utilities

Functions for loading, saving, and managing pipeline configurations.
Includes factory functions to create pipeline components from config.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from ..core.models import PipelineConfig
from ..core.pipeline import EvaluationPipeline


def load_config(config_path: str) -> PipelineConfig:
    """
    Load pipeline configuration from file.
    
    Supports JSON and YAML formats.
    
    Args:
        config_path: Path to configuration file (.json or .yaml/.yml)
        
    Returns:
        PipelineConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
        
    Example:
        config = load_config('experiments/baseline.yaml')
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Detect format from extension
    suffix = config_file.suffix.lower()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if suffix == '.json':
                config_dict = json.load(f)
            elif suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {suffix}. Use .json, .yaml, or .yml")
        
        # Validate basic structure
        validate_config(config_dict)
        
        # Create PipelineConfig from dict
        return PipelineConfig.from_dict(config_dict)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")


def save_config(config: PipelineConfig, output_path: str) -> None:
    """
    Save pipeline configuration to file.
    
    Args:
        config: PipelineConfig to save
        output_path: Path to save configuration (.json or .yaml/.yml)
        
    Example:
        save_config(my_config, 'experiments/my_experiment.yaml')
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    config_dict = config.to_dict()
    
    # Determine format from extension
    suffix = output_file.suffix.lower()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if suffix == '.json':
            json.dump(config_dict, f, indent=2)
        elif suffix in ['.yaml', '.yml']:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        else:
            # Default to JSON
            json.dump(config_dict, f, indent=2)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    required_fields = ['experiment_name', 'dataset_config', 'memory_system_config']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Check types
    if not isinstance(config['experiment_name'], str):
        raise ValueError("experiment_name must be a string")
    
    if not isinstance(config.get('dataset_config', {}), dict):
        raise ValueError("dataset_config must be a dictionary")
    
    if not isinstance(config.get('memory_system_config', {}), dict):
        raise ValueError("memory_system_config must be a dictionary")
    
    # Validate optional fields
    if 'max_questions' in config and config['max_questions'] is not None:
        if not isinstance(config['max_questions'], int) or config['max_questions'] <= 0:
            raise ValueError("max_questions must be a positive integer or None")
    
    if 'batch_size' in config:
        if not isinstance(config['batch_size'], int) or config['batch_size'] <= 0:
            raise ValueError("batch_size must be a positive integer")
    
    return True


def create_pipeline_from_config(config: PipelineConfig) -> EvaluationPipeline:
    """
    Create an EvaluationPipeline from a PipelineConfig.
    
    This is a factory function that automatically constructs all pipeline components
    from the configuration. It creates the reader, backends, memory system, and
    evaluators based on the types specified in the config.
    
    Args:
        config: PipelineConfig with all settings
        
    Returns:
        Fully configured EvaluationPipeline ready to run
        
    Raises:
        ValueError: If config is missing required information or has invalid types
        
    Example:
        config = load_config('experiments/baseline.yaml')
        pipeline = create_pipeline_from_config(config)
        results = pipeline.run('data/dataset.json')
        
    Supported dataset_config types:
        - dataset_type: "longmemeval" -> LongMemEvalReader
        
    Supported llm_config backend types:
        - backend_type: "ollama" -> OllamaBackend
        - backend_type: "openai" -> OpenAIBackend
        
    Supported memory_system_config types:
        - system_type: "full_context" -> FullContextMemorySystem
        - system_type: "mem0_api" -> Mem0ApiMemorySystem
        - system_type: "mem0_local" -> Mem0LocalMemorySystem
        
    Supported evaluation_config evaluator types:
        - evaluator_type: "llm_judge" -> LLMJudgeEvaluator
    """
    # Import here to avoid circular dependencies
    from ..readers import LongMemEvalReader
    from ..backends import OllamaBackend, OpenAIBackend
    from ..memory_systems import FullContextMemorySystem, Mem0ApiMemorySystem, Mem0LocalMemorySystem
    from ..evaluators import LLMJudgeEvaluator, F1ScoreEvaluator
    
    # 1. Create Reader
    dataset_type = config.dataset_config.get('dataset_type', 'longmemeval')
    
    if dataset_type == 'longmemeval':
        reader = LongMemEvalReader()
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Supported: 'longmemeval'")
    
    # 2. Create LLM Backend for answering
    backend_type = config.llm_config.get('backend_type', 'ollama')
    
    if backend_type == 'ollama':
        answer_backend = OllamaBackend()
        answer_backend.initialize({
            'host': config.llm_config.get('host', 'http://localhost:11434'),
            'model': config.llm_config.get('model', 'llama3.3:latest'),
            'temperature': config.llm_config.get('temperature', 0.7),
            'top_p': config.llm_config.get('top_p', 0.9),
        })
    elif backend_type == 'openai':
        answer_backend = OpenAIBackend()
        answer_backend.initialize({
            'api_key': config.llm_config.get('api_key'),  # Can be None, will use env var
            'model': config.llm_config.get('model', 'gpt-4o-mini'),
            'temperature': config.llm_config.get('temperature', 0.7),
            'max_tokens': config.llm_config.get('max_tokens'),
        })
    else:
        raise ValueError(f"Unknown backend_type: {backend_type}. Supported: 'ollama', 'openai'")
    
    # 3. Create Memory System
    system_type = config.memory_system_config.get('system_type', 'full_context')
    
    if system_type == 'full_context':
        memory_system = FullContextMemorySystem(llm_backend=answer_backend)
        memory_system.initialize({
            'include_session_separators': config.memory_system_config.get('include_session_separators', True),
            'max_context_length': config.memory_system_config.get('max_context_length'),
        })
    elif system_type == 'mem0_api':
        memory_system = Mem0ApiMemorySystem(llm_backend=answer_backend)
        memory_system.initialize({
            'api_key': config.memory_system_config.get('api_key'),
            'user_id': config.memory_system_config.get('user_id', 'default_user'),
            'agent_id': config.memory_system_config.get('agent_id'),
            'org_id': config.memory_system_config.get('org_id'),
            'project_id': config.memory_system_config.get('project_id'),
            'prompt_template': config.memory_system_config.get('prompt_template'),
            'search_limit': config.memory_system_config.get('search_limit', 5),
            'enable_graph': config.memory_system_config.get('enable_graph', False),
        })
    elif system_type == 'mem0_local':
        memory_system = Mem0LocalMemorySystem(llm_backend=answer_backend)
        memory_system.initialize({
            'user_id': config.memory_system_config.get('user_id', 'default_user'),
            'llm': config.memory_system_config.get('llm'),
            'embedder': config.memory_system_config.get('embedder'),
            'vector_store': config.memory_system_config.get('vector_store'),
            'reranker': config.memory_system_config.get('reranker'),
            'prompt_template': config.memory_system_config.get('prompt_template'),
            'search_limit': config.memory_system_config.get('search_limit', 5),
            'version': config.memory_system_config.get('version', 'v1.1'),
        })
    else:
        raise ValueError(f"Unknown system_type: {system_type}. Supported: 'full_context', 'mem0_api', 'mem0_local'")
    
    # 4. Create Evaluators
    evaluators = []
    
    # Support multiple evaluators (can be a list or single value)
    evaluator_types = config.evaluation_config.get('evaluator_types', [config.evaluation_config.get('evaluator_type', 'llm_judge')])
    if not isinstance(evaluator_types, list):
        evaluator_types = [evaluator_types]
    
    for evaluator_type in evaluator_types:
        if evaluator_type == 'llm_judge':
            # Create judge backend (can be different from answer backend)
            judge_backend_type = config.evaluation_config.get('judge_backend_type', backend_type)
            
            if judge_backend_type == 'ollama':
                judge_backend = OllamaBackend()
                judge_backend.initialize({
                    'host': config.evaluation_config.get('judge_host', config.llm_config.get('host', 'http://localhost:11434')),
                    'model': config.evaluation_config.get('judge_model', config.llm_config.get('model', 'llama3.3:latest')),
                    'temperature': config.evaluation_config.get('judge_temperature', 0.0),
                })
            elif judge_backend_type == 'openai':
                judge_backend = OpenAIBackend()
                judge_backend.initialize({
                    'api_key': config.evaluation_config.get('judge_api_key', config.llm_config.get('api_key')),
                    'model': config.evaluation_config.get('judge_model', 'gpt-4o-mini'),
                    'temperature': config.evaluation_config.get('judge_temperature', 0.0),
                    'max_tokens': config.evaluation_config.get('max_tokens', 10),
                })
            else:
                raise ValueError(f"Unknown judge_backend_type: {judge_backend_type}")
            
            # Create evaluator
            evaluator = LLMJudgeEvaluator(judge_backend=judge_backend)
            evaluator.initialize({
                'temperature': config.evaluation_config.get('judge_temperature', 0.0),
                'max_tokens': config.evaluation_config.get('max_tokens', 10),
            })
            evaluators.append(evaluator)
            
        elif evaluator_type == 'f1_score':
            # Create F1 score evaluator
            evaluator = F1ScoreEvaluator()
            evaluator.initialize({
                'lowercase': config.evaluation_config.get('lowercase', True),
                'remove_punctuation': config.evaluation_config.get('remove_punctuation', True),
                'remove_stopwords': config.evaluation_config.get('remove_stopwords', False),
            })
            evaluators.append(evaluator)
            
        else:
            raise ValueError(f"Unknown evaluator_type: {evaluator_type}. Supported: 'llm_judge', 'f1_score'")
    
    # 5. Create and return pipeline
    pipeline = EvaluationPipeline(
        reader=reader,
        memory_system=memory_system,
        evaluators=evaluators,
        config=config,
    )
    
    return pipeline
