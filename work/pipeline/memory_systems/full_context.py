"""
Full Context Memory System

Baseline system that feeds all context to the LLM without any memory mechanism.
"""

from typing import List, Dict, Any, Optional
import time

from ..core.interfaces import MemorySystem, LLMBackend
from ..core.models import ChatTurn, Answer


class FullContextMemorySystem(MemorySystem):
    """
    Baseline memory system that provides full chat history as context.
    
    This is the simplest approach: concatenate all chat turns and
    pass them to the LLM with the question.
    """
    
    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        super().__init__(llm_backend)
        # TODO: Initialize system
        self.context = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the full context system.
        
        Expected config:
            - prompt_template: Template for formatting context and question (optional)
            - max_context_length: Maximum context length to use (optional)
        """
        # TODO: Implement initialization
        pass
    
    def process_context(self, context: List[List[ChatTurn]]) -> None:
        """
        Store the full context for later use.
        
        Args:
            context: List of sessions, each containing ChatTurns
        """
        # TODO: Implement context processing
        # Just store it for now, format when answering
        pass
    
    def answer_question(self, question: str, question_id: str) -> Answer:
        """
        Answer question using full context.
        
        Args:
            question: Question text
            question_id: Question identifier
            
        Returns:
            Answer object with generated response
        """
        # TODO: Implement answer generation
        # 1. Format context into a prompt
        # 2. Add question to prompt
        # 3. Call LLM backend
        # 4. Return Answer object
        pass
    
    def reset(self) -> None:
        """Clear stored context"""
        # TODO: Implement reset
        pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about context"""
        # TODO: Return stats (number of turns, tokens, etc.)
        pass
