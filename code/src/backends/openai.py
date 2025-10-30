"""
OpenAI Backend

LLM backend implementation for OpenAI API (GPT-4, GPT-3.5, etc.)
"""

from typing import Dict, Any, Optional
import os
from openai import OpenAI, OpenAIError, APIError, RateLimitError
import backoff

from ..core.interfaces import LLMBackend


class OpenAIBackend(LLMBackend):
    """
    Backend for OpenAI API models.
    
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    Includes automatic retry logic for rate limits and transient errors.
    """
    
    def __init__(self):
        """Initialize OpenAI backend (configuration happens in initialize())"""
        self.client: Optional[OpenAI] = None
        self.model: Optional[str] = None
        self.config: Dict[str, Any] = {}
        self._default_params: Dict[str, Any] = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize OpenAI backend.
        
        Expected config:
            - api_key: OpenAI API key (or from OPENAI_API_KEY environment variable)
            - model: Model name (e.g., 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo')
            - temperature: Temperature for generation (default: 0.7)
            - max_tokens: Maximum tokens to generate (optional)
            - organization: OpenAI organization ID (optional)
            - base_url: Custom API base URL (optional, for Azure OpenAI or proxies)
        
        Raises:
            ValueError: If required config is missing
            RuntimeError: If cannot create OpenAI client
        """
        # Validate required config
        if 'model' not in config:
            raise ValueError("Config must include 'model' parameter")
        
        # Get API key from config or environment
        api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Provide 'api_key' in config "
                "or set OPENAI_API_KEY environment variable."
            )
        
        # Extract configuration
        self.model = config['model']
        self.config = config.copy()
        
        # Create OpenAI client
        try:
            client_kwargs = {'api_key': api_key}
            
            # Add optional parameters
            if 'organization' in config:
                client_kwargs['organization'] = config['organization']
            if 'base_url' in config:
                client_kwargs['base_url'] = config['base_url']
            
            self.client = OpenAI(**client_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create OpenAI client: {e}")
        
        # Set default generation parameters
        self._default_params = {
            'temperature': config.get('temperature', 0.7),
        }
        
        # Add max_tokens if specified
        if 'max_tokens' in config:
            self._default_params['max_tokens'] = config['max_tokens']
        
        # Add other optional parameters
        if 'top_p' in config:
            self._default_params['top_p'] = config['top_p']
        if 'frequency_penalty' in config:
            self._default_params['frequency_penalty'] = config['frequency_penalty']
        if 'presence_penalty' in config:
            self._default_params['presence_penalty'] = config['presence_penalty']
    
    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIError),
        max_tries=5,
        max_time=300
    )
    def _generate_with_retry(self, messages: list, params: dict) -> str:
        """
        Internal method to generate with automatic retry on rate limits.
        
        Args:
            messages: List of message dictionaries
            params: Generation parameters
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **params
        )
        
        return response.choices[0].message.content.strip()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            prompt: Input prompt (will be formatted as user message)
            **kwargs: Additional generation parameters (override defaults)
                - temperature: Override temperature
                - max_tokens: Override max tokens
                - top_p: Top-p sampling parameter
                - system: System prompt (optional)
                - frequency_penalty: Frequency penalty
                - presence_penalty: Presence penalty
                
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If backend not initialized or generation fails
        """
        if self.client is None or self.model is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        
        # Merge default params with kwargs
        params = self._default_params.copy()
        
        # Extract system prompt if provided
        system_prompt = kwargs.pop('system', None)
        
        # Update params with any overrides
        for key in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
            if key in kwargs:
                params[key] = kwargs.pop(key)
        
        # Format messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Generate response with retry logic
        try:
            return self._generate_with_retry(messages, params)
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during generation: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current OpenAI model.
        
        Returns:
            Dictionary with model information
        """
        if self.client is None or self.model is None:
            return {
                'initialized': False,
                'model': None,
                'provider': 'OpenAI',
            }
        
        info = {
            'initialized': True,
            'provider': 'OpenAI',
            'model': self.model,
            'temperature': self._default_params.get('temperature'),
            'max_tokens': self._default_params.get('max_tokens', 'unlimited'),
        }
        
        # Add optional parameters if set
        if 'top_p' in self._default_params:
            info['top_p'] = self._default_params['top_p']
        if 'frequency_penalty' in self._default_params:
            info['frequency_penalty'] = self._default_params['frequency_penalty']
        if 'presence_penalty' in self._default_params:
            info['presence_penalty'] = self._default_params['presence_penalty']
        
        # Add organization if configured
        if 'organization' in self.config:
            info['organization'] = self.config['organization']
        
        # Add base URL if custom
        if 'base_url' in self.config:
            info['base_url'] = self.config['base_url']
        
        return info
    
    def is_available(self) -> bool:
        """
        Check if the OpenAI backend is available and ready to use.
        
        Returns:
            True if backend can generate text, False otherwise
        """
        if self.client is None or self.model is None:
            return False
        
        try:
            # Try a minimal generation
            messages = [{"role": "user", "content": "test"}]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1
            )
            return response.choices[0].message.content is not None
        except Exception:
            return False
