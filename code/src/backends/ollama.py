"""
Ollama Backend

LLM backend implementation for Ollama local models.
"""

from typing import Dict, Any, Optional
import ollama
import time

from ..core.interfaces import LLMBackend


class OllamaBackend(LLMBackend):
    """
    Backend for Ollama local LLM models.
    
    Supports any model available in Ollama (llama, qwen, mistral, etc.)
    """
    
    def __init__(self):
        """Initialize Ollama backend (configuration happens in initialize())"""
        self.client: Optional[ollama.Client] = None
        self.model: Optional[str] = None
        self.config: Dict[str, Any] = {}
        self._default_options: Dict[str, Any] = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize Ollama backend.
        
        Expected config:
            - host: Ollama server host (default: http://localhost:11434)
            - model: Model name (e.g., 'llama3.3:latest', 'qwen2.5:32b')
            - temperature: Temperature for generation (default: 0.7)
            - top_p: Top-p sampling parameter (default: 0.9)
            - timeout: Request timeout in seconds (default: 120)
            - max_tokens: Maximum tokens to generate (optional)
        
        Raises:
            ValueError: If required config is missing
            RuntimeError: If cannot connect to Ollama server
        """
        # Validate required config
        if 'model' not in config:
            raise ValueError("Config must include 'model' parameter")
        
        # Extract configuration
        self.model = config['model']
        host = config.get('host', 'http://localhost:11434')
        
        # Store full config
        self.config = config.copy()
        
        # Create Ollama client
        try:
            self.client = ollama.Client(host=host)
        except Exception as e:
            raise RuntimeError(f"Failed to create Ollama client: {e}")
        
        # Set default generation options
        self._default_options = {
            'temperature': config.get('temperature', 0.7),
            'top_p': config.get('top_p', 0.9),
        }
        
        # Add max_tokens if specified
        if 'max_tokens' in config:
            self._default_options['num_predict'] = config['max_tokens']
        
        # Validate model availability
        try:
            models_response = self.client.list()
            available_models = [m.model for m in models_response.models]
            
            # Check if exact model exists, or if base model exists
            model_exists = any(
                self.model == m or self.model.split(':')[0] == m.split(':')[0]
                for m in available_models
            )
            
            if not model_exists:
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available models: {', '.join(available_models)}"
                )
        except Exception as e:
            # Don't fail initialization if we can't list models (might be network issue)
            # but warn the user
            print(f"⚠️ Warning: Could not verify model availability: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Ollama.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters (override defaults)
                - temperature: Override temperature
                - top_p: Override top_p
                - max_tokens: Override max tokens
                - system: System prompt (optional)
                
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If backend not initialized or generation fails
        """
        if self.client is None or self.model is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        
        # Merge default options with kwargs
        options = self._default_options.copy()
        
        # Handle special kwargs
        if 'temperature' in kwargs:
            options['temperature'] = kwargs.pop('temperature')
        if 'top_p' in kwargs:
            options['top_p'] = kwargs.pop('top_p')
        if 'max_tokens' in kwargs:
            options['num_predict'] = kwargs.pop('max_tokens')
        
        # Add any other options passed
        options.update(kwargs)
        
        # Generate response
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options=options
            )
            
            return response['response'].strip()
            
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current Ollama model.
        
        Returns:
            Dictionary with model information
        """
        if self.client is None or self.model is None:
            return {
                'initialized': False,
                'model': None,
                'host': None,
            }
        
        info = {
            'initialized': True,
            'model': self.model,
            'host': self.config.get('host', 'http://localhost:11434'),
            'temperature': self._default_options.get('temperature'),
            'top_p': self._default_options.get('top_p'),
        }
        
        return info
    
    def is_available(self) -> bool:
        """
        Check if the Ollama backend is available and ready to use.
        
        Returns:
            True if backend can generate text, False otherwise
        """
        if self.client is None or self.model is None:
            return False
        
        try:
            # Try a minimal generation
            response = self.client.generate(
                model=self.model,
                prompt="test",
                options={'num_predict': 1}
            )
            return 'response' in response
        except Exception:
            return False
