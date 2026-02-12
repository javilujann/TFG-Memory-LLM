"""
Full Context Memory System

Baseline system that feeds all context to the LLM without any memory mechanism.
"""

from typing import List, Dict, Any, Optional
import time

from ..core.interfaces import MemorySystem, LLMBackend
from ..core.models import ChatTurn, Answer, Question


class FullContextMemorySystem(MemorySystem):
    """
    Baseline memory system that provides full chat history as context.
    
    This is the simplest approach: concatenate all chat turns and
    pass them to the LLM with the question.
    """
    
    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        """
        Initialize with optional LLM backend.
        
        Args:
            llm_backend: LLM backend to use for generation
        """
        super().__init__(llm_backend)
        self.context: Optional[List[List[ChatTurn]]] = None
        self.config: Dict[str, Any] = {}
        self._prompt_template: str = self._default_prompt_template()
    
    def _default_prompt_template(self) -> str:
        """
        Default prompt template for full context system.
        
        Returns:
            Template string with {context} and {question} placeholders
        """
        return """{context}

=== QUESTION ===
Based on the chat history above, please answer the following question:

{question}

=== INSTRUCTIONS ===
- Use only information from the chat history above
- Be specific and accurate
- If you cannot find the answer in the chat history, say "I don't have enough information to answer this question"
- Keep your answer concise and direct

Answer:"""
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the full context system.
        
        Expected config:
            - prompt_template: Template for formatting context and question (optional)
            - max_context_length: Maximum number of turns to include (optional)
            - include_session_separators: Whether to show session boundaries (default: True)
        
        Raises:
            RuntimeError: If LLM backend is not set
        """
        if self.llm_backend is None:
            raise RuntimeError("FullContextMemorySystem requires an LLM backend")
        
        self.config = config.copy()
        
        # Use custom prompt template if provided
        if 'prompt_template' in config:
            self._prompt_template = config['prompt_template']

    def process_context(self, question: Question) -> None:
        """
        Store the full context for later use.
        
        Args:
            question: The Question object containing context to process
        """
        # Start timing
        start_time = time.time()
        
        context = question.context

        # Apply max context length if configured
        max_length = self.config.get('max_context_length')
        
        if max_length is not None:
            # Flatten, truncate, then reconstruct sessions
            all_turns = [turn for session in context for turn in session]
            if len(all_turns) > max_length:
                all_turns = all_turns[-max_length:]  # Keep most recent turns
                # Put all truncated turns in a single session
                self.context = [all_turns]
            else:
                self.context = context
        else:
            self.context = context
        
        # Calculate and store processing time
        processing_time = time.time() - start_time
        question.metadata['context_processing_time'] = processing_time
    
    def _format_context(self) -> str:
        """
        Format stored context into a readable string.
        
        Returns:
            Formatted context string
        """
        if not self.context:
            return "No chat history available."
        
        include_separators = self.config.get('include_session_separators', True)
        
        formatted = "=== CHAT HISTORY ===\n\n"
        
        for session_idx, session in enumerate(self.context, 1):
            if include_separators and len(self.context) > 1:
                formatted += f"Session {session_idx}:\n"
            
            for turn in session:
                role_label = turn.role.capitalize()
                formatted += f"{role_label}: {turn.content}\n"
            
            if include_separators and len(self.context) > 1:
                formatted += "\n"
        
        return formatted.strip()
    
    def answer_question(self, question: Question) -> Answer:
        """
        Answer question using full context.
        
        Args:
            question: Question object containing text and metadata
                        
        Returns:
            Answer object with generated response
            
        Raises:
            RuntimeError: If context not processed or LLM backend not available
        """
        if self.context is None:
            raise RuntimeError("Context not processed. Call process_context() first.")
        
        if self.llm_backend is None:
            raise RuntimeError("LLM backend not set")
        
        # Format context and create prompt
        context_str = self._format_context()
        prompt = self._prompt_template.format(
            context=context_str,
            question=question.question_text
        )
        
        # Generate answer
        start_time = time.time()
        
        try:
            answer_text = self.llm_backend.generate(prompt)
            processing_time = time.time() - start_time
            
            # Create Answer object
            answer = Answer(
                question_id=question.question_id,
                answer_text=answer_text,
                processing_time=processing_time,
                metadata={
                    'memory_system': 'FullContext',
                    'context_turns': sum(len(session) for session in self.context),
                    'context_sessions': len(self.context),
                    'prompt_length': len(prompt),
                    'full_prompt': prompt,
                }
            )
            
            return answer
            
        except Exception as e:
            # Return error as answer
            processing_time = time.time() - start_time
            return Answer(
                question_id=question.question_id,
                answer_text=f"Error generating answer: {e}",
                processing_time=processing_time,
                metadata={
                    'memory_system': 'FullContext',
                    'error': str(e),
                }
            )
    
    def reset(self) -> None:
        """Clear stored context"""
        self.context = None
    
    def get_all(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get all memories (not applicable for FullContext system).
        
        The full context system doesn't use discrete memories,
        so this returns an empty list.
        
        Args:
            filters: Ignored for this system
        
        Returns:
            Empty list (memories not applicable)
        """
        return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored context.
        
        Returns:
            Dictionary with context statistics
        """
        if self.context is None:
            return {
                'context_processed': False,
                'total_turns': 0,
                'total_sessions': 0,
                'total_characters': 0,
            }
        
        total_turns = sum(len(session) for session in self.context)
        total_chars = sum(
            len(turn.content)
            for session in self.context
            for turn in session
        )
        
        return {
            'context_processed': True,
            'total_turns': total_turns,
            'total_sessions': len(self.context),
            'total_characters': total_chars,
            'average_turn_length': total_chars / total_turns if total_turns > 0 else 0,
            'max_context_length': self.config.get('max_context_length', 'unlimited'),
        }
