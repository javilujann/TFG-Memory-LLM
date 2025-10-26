"""
Configuration Utilities

Functions for loading, saving, and managing pipeline configurations.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from ..core.models import PipelineConfig


def load_config(config_path: str) -> PipelineConfig:
    """
    Load pipeline configuration from file.
    
    Supports JSON and YAML formats.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        PipelineConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    # TODO: Implement config loading
    # 1. Detect file format (json/yaml)
    # 2. Load file
    # 3. Validate structure
    # 4. Create PipelineConfig
    pass


def save_config(config: PipelineConfig, output_path: str) -> None:
    """
    Save pipeline configuration to file.
    
    Args:
        config: PipelineConfig to save
        output_path: Path to save configuration
    """
    # TODO: Implement config saving
    # 1. Convert to dict
    # 2. Determine format from extension
    # 3. Write to file
    pass


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    # TODO: Implement deep merge of configs
    pass


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # TODO: Implement validation
    # Check required fields, types, etc.
    pass


def create_default_config(experiment_name: str) -> PipelineConfig:
    """
    Create a default pipeline configuration.
    
    Args:
        experiment_name: Name for the experiment
        
    Returns:
        Default PipelineConfig
    """
    # TODO: Implement default config creation
    pass
