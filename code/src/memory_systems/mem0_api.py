"""
Mem0 API Memory System

Memory system that uses the Mem0 hosted platform (API-based).
This is a minimal approach requiring only an API key.
"""

from typing import List, Dict, Any, Optional
import time

from mem0 import MemoryClient

from ..core.interfaces import MemorySystem, LLMBackend
from ..core.models import  Answer, Question


class Mem0ApiMemorySystem(MemorySystem):
    """
    Memory system using Mem0's hosted platform API.
    
    This is the simplest approach - memories are stored and retrieved
    via Mem0's cloud service. Requires only an API key.
    
    Features:
    - Automatic memory extraction and storage
    - Semantic search across memories
    - No local infrastructure needed
    - Enterprise features (webhooks, analytics, etc.)
    """
    
    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        """
        Initialize with optional LLM backend.
        
        Args:
            llm_backend: LLM backend to use for question answering
        """
        super().__init__(llm_backend)
        self.client: Optional[MemoryClient] = None
        self.config: Dict[str, Any] = {}
        self._prompt_template: str = self._default_prompt_template()
        self._user_id: Optional[str] = None
        self.ActivateReset: bool = False
    
    def _default_prompt_template(self) -> str:
        """
        Default prompt template for Mem0 API system.
        
        Returns:
            Template string with {memories} and {question} placeholders
        """
        return """You are a helpful AI assistant that answers questions based on relevant memories.

=== RELEVANT MEMORIES ===
{memories}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
- Use the relevant memories above to answer the question
- Be specific and accurate
- If the memories don't contain enough information, say "I don't have enough information to answer this question"
- Keep your answer concise and direct

Answer:"""
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Mem0 API client.
        
        Expected config:
            - api_key: Mem0 API key (required)
            - user_id: User identifier for memory isolation (required)
            - prompt_template: Custom prompt template (optional)
            - search_limit: Number of memories to retrieve (default: 5)
            - enable_graph: Enable graph memory features (default: False)
        
        Raises:
            ValueError: If required config is missing
            RuntimeError: If LLM backend is not set or API client initialization fails
        """
        if self.llm_backend is None:
            raise RuntimeError("Mem0ApiMemorySystem requires an LLM backend")
        
        # Validate required fields
        if 'api_key' not in config:
            raise ValueError("Mem0 API configuration requires 'api_key'")
        if 'user_id' not in config:
            raise ValueError("Mem0 API configuration requires 'user_id'")
        
        self.config = config.copy()
        self._user_id = config['user_id']
        
        # Initialize Mem0 client
        try:
            client_kwargs = {'api_key': config['api_key']}
            
            self.client = MemoryClient(**client_kwargs)
            
            # Enable graph memory if configured
            if config.get('enable_graph', False) and 'project_id' in config:
                self.client.project.update(enable_graph=True)
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mem0 API client: {e}")
        
        # Use custom prompt template if provided
        if 'prompt_template' in config and config['prompt_template']:
            self._prompt_template = config['prompt_template'] 

        if 'reset' in config and config['reset']:
            self.ActivateReset = True

    def process_context(self, question: Question) -> None:
        """
        Process context by adding it to Mem0's memory storage.
        
        Converts chat sessions to messages format and stores them in Mem0.
        The platform automatically extracts and indexes memories.
        
        Args:
            context: List of sessions, each containing ChatTurns
        """
        if self.client is None:
            raise RuntimeError("Mem0 client not initialized. Call initialize() first.")
        
        # Check if pair dataSet-question has already been processed
        memories = self._search_memories(question)
        if memories and len(memories) > 0:
            return
        
        # Exctract context from question
        context = question.context

        # Extract user and run IDs from metadata to later filter memories
        user_id = self._user_id
        run_id = question.question_id

        # Convert context to user-assistant pairs and add to Mem0
        for session_idx, session in enumerate(context):
            # Group turns into user-assistant pairs
            i = 0
            while i < len(session):
                messages = []
                
                # Get user message
                if i < len(session) and session[i].role == "user":
                    messages.append({
                        "role": session[i].role,
                        "content": session[i].content
                    })
                    i += 1
                
                # Get assistant response
                if i < len(session) and session[i].role == "assistant":
                    messages.append({
                        "role": session[i].role,
                        "content": session[i].content
                    })
                    i += 1
                
                # Add pair to Mem0 (skip if incomplete)
                if len(messages) == 2:
                    try:
                        add_kwargs = {
                            'messages': messages,
                            'user_id': user_id,
                            'run_id': run_id,
                            'version': 'v2',
                            'async_mode': False,  # ⚠️ CRITICAL: Force synchronous processing
                            'metadata': {
                                'session_idx': session_idx,
                                'turn_index': i // 2
                            }
                        }
                        
                        # Add memories 
                        self.client.add(**add_kwargs)
                        
                    except Exception as e:
                        # Log error but continue processing other pairs
                        print(f"Warning: Failed to add turn {i//2} from session {session_idx} to Mem0: {e}")
                else:
                    # Skip incomplete pairs
                    if messages:
                        print(f"Warning: Incomplete pair in session {session_idx}, skipping")
    
    def _search_memories(self, question: Question) -> List:
        """
        Search for relevant memories using the question.
        
        Args:
            question: The question to search for
            
        Returns:
            Formatted string of relevant memories
        """
        if self.client is None:
            raise RuntimeError("Mem0 client not initialized")
        
        search_limit = self.config.get('search_limit', 5)
        user_id = self._user_id
        run_id = question.question_id
        
        # Build filters
        filters = {
            "AND": [{
                "user_id": user_id,
                "run_id": run_id
            }],
        }
        
        try:
            # Search memories
            search_response = self.client.search(
                query=question.question_text,
                filters=filters,
                version='v2',
                limit=search_limit
            )

            # Extract results list from response
            # Mem0 API returns a dict with 'results' key
            results = search_response.get('results', []) if isinstance(search_response, dict) else search_response
            return results
        
        except Exception as e:
            print(f"Warning: Memory search failed: {e}")
            return "Memory search unavailable."
        
    def format_memories(self, results : List) -> str:
        # Format memories
        if results and len(results) > 0:
            memory_lines = [f"- {mem['memory']}" for mem in results]
            memories_str = "\n".join(memory_lines)
            return memories_str
        else:
            return "No relevant memories found."
    
    def answer_question(self, question: Question) -> Answer:
        """
        Answer a question using relevant memories from Mem0.
        
        Process:
        1. Search for relevant memories using the question
        2. Format memories into a prompt
        3. Generate answer using LLM backend
        
        Args:
            question: The question to answer
            
        Returns:
            Answer object with generated response and metadata
        """
        if self.llm_backend is None:
            raise RuntimeError("LLM backend not set")
        
        start_time = time.time()
        
        # Search for relevant memories
        results = self._search_memories(question)
        memories_str = self.format_memories(results)
        
        # Format prompt with memories and question
        prompt = self._prompt_template.format(
            memories=memories_str,
            question=question.question_text 
        )
        
        # Generate answer
        answer_text = self.llm_backend.generate(prompt)
        processing_time = time.time() - start_time
        
        return Answer(
            question_id=question.question_id,
            answer_text=answer_text,
            processing_time=processing_time,
            metadata={
                'memory_system': 'mem0_api',
                'num_memories_retrieved': len(memories_str.split('\n')) if memories_str != "No relevant memories found." else 0,
                'memoriesRetrieved': memories_str,
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Mem0 API memory system.
        
        Returns:
            Dictionary with memory stats
        """
        if self.client is None:
            return {'status': 'not_initialized'}
        
        try:
            # Get all memories for this user
            all_memories = self.client.get_all(
                user_id=self._user_id,
                page=1,
                page_size=1000  # Adjust based on expected size
            )
            
            return {
                'status': 'active',
                'mode': 'api',
                'total_memories': len(all_memories) if all_memories else 0,
                'search_limit': self.config.get('search_limit', 5),
                'enable_graph': self.config.get('enable_graph', False)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def reset(self) -> None:
        """
        Delete all memories for this user.
        
        Warning: This is irreversible!
        """
        if self.client is None:
            raise RuntimeError("Mem0 client not initialized")
        
        # Only proceed if configured to do so
        if not self.ActivateReset:
            return
        
        try:
            self.client.delete_all(user_id=self._user_id)
            print(f"✅ All memories deleted for user: {self._user_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to reset memories: {e}")
