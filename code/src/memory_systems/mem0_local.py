"""
Mem0 Local Memory System

Memory system using the Mem0 SDK in local/self-hosted mode.
Requires configuration of LLM, embedder, and vector store.
"""

from typing import List, Dict, Any, Optional
import time

from mem0 import Memory

from ..core.interfaces import MemorySystem, LLMBackend
from ..core.models import  Answer, Question


class Mem0LocalMemorySystem(MemorySystem):
    """
    Memory system using Mem0's local SDK.
    
    This approach gives full control over the memory infrastructure.
    Requires configuration of:
    - LLM provider (for memory extraction)
    - Embedder (for semantic search)
    - Vector store (for memory storage)
    - Optional graph database for relationships
    
    Features:
    - Full control over infrastructure
    - Self-hosted, no external API calls
    - Customizable components
    - Works offline
    """
    
    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        """
        Initialize with optional LLM backend.
        
        Note: The Mem0 SDK uses its own LLM for memory extraction,
        but we still need an LLM backend for answering questions.
        
        Args:
            llm_backend: LLM backend to use for question answering
        """
        super().__init__(llm_backend)
        self.memory: Optional[Memory] = None
        self.config: Dict[str, Any] = {}
        self._prompt_template: str = self._default_prompt_template()
        self._user_id: Optional[str] = None
        self.ActivateReset: bool = False
        self.enableGraph: bool = False
    
    def _default_prompt_template(self) -> str:
        """
        Default prompt template for Mem0 local system.
        
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
- If a knowledge graph is provided, use the relationships to understand connections between entities
- Be specific and accurate
- If the memories don't contain enough information, say "I don't have enough information to answer this question"
- Keep your answer concise and direct

Answer:"""
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Mem0 local SDK.
        
        Expected config:
            - user_id: User identifier for memory isolation (required)
            - llm: LLM configuration for memory extraction (required)
                - provider: e.g., 'ollama', 'openai'
                - config: provider-specific settings
            - embedder: Embedder configuration for semantic search (required)
                - provider: e.g., 'ollama', 'openai'
                - config: provider-specific settings
            - vector_store: Vector DB configuration (required)
                - provider: e.g., 'chroma', 'qdrant', 'supabase'
                - config: provider-specific settings
            - prompt_template: Custom prompt template (optional)
            - search_limit: Number of memories to retrieve (default: 5)
            - search_threshold: Minimum similarity score for retrieval (default: 0.3, range: 0.0-1.0)
            - version: Mem0 config version (default: 'v1.1')
        
        Raises:
            ValueError: If required config is missing
            RuntimeError: If LLM backend is not set or Memory initialization fails
        """
        if self.llm_backend is None:
            raise RuntimeError("Mem0LocalMemorySystem requires an LLM backend")
        
        # Validate required fields
        if 'user_id' not in config:
            raise ValueError("Mem0 local configuration requires 'user_id'")
        if 'llm' not in config:
            raise ValueError("Mem0 local configuration requires 'llm' settings")
        if 'embedder' not in config:
            raise ValueError("Mem0 local configuration requires 'embedder' settings")
        if 'vector_store' not in config:
            raise ValueError("Mem0 local configuration requires 'vector_store' settings")
        if 'enableGraph' in config and config['enableGraph']:
            self.enableGraph = True
            if 'graph_store' not in config:
                raise ValueError("Mem0 local configuration requires 'graph_store' settings when 'enableGraph' is True")
        
        self.config = config.copy()
        self._user_id = config['user_id']
        
        # Build Mem0 config
        mem0_config = {
            "enableGraph": self.enableGraph,
            "version": config.get('version', 'v1.1'),
            "llm": config['llm'],
            "embedder": config['embedder'],
            "vector_store": config['vector_store']
        }
        
        # Add graph_store only if enableGraph is True and graph_store is configured
        if self.enableGraph and 'graph_store' in config and config['graph_store']:
            mem0_config['graph_store'] = config['graph_store']
        
        # Add optional reranker if provided
        if 'reranker' in config and config['reranker']:
            mem0_config['reranker'] = config['reranker']
        
        # Initialize Mem0 Memory instance
        try:
            self.memory = Memory.from_config(mem0_config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mem0 local memory: {e}")
        
        # Use custom prompt template if provided
        if 'prompt_template' in config and config['prompt_template']:
            self._prompt_template = config['prompt_template']

        if 'reset' in config and config['reset']:
            self.ActivateReset = True

    def process_context(self, question: Question) -> None:
        """
        Process context by adding it to local Mem0 storage.
        
        Converts chat sessions to messages format and stores them locally.
        Mem0 SDK extracts and indexes memories using configured LLM and embedder.
        
        Args:
            context: List of sessions, each containing ChatTurns
        """
        # Start timing
        start_time = time.time()
        
        if self.memory is None:
            raise RuntimeError("Mem0 Memory not initialized. Call initialize() first.")
        
        # Check if pair dataSet-question has already been processed
        dict = self._search_memories(question, fromContext=True)
        results = dict.get('results', [])
        if results and len(results) > 0:
            # Calculate and store processing time (cached case)
            processing_time = time.time() - start_time
            question.metadata['context_processing_time'] = processing_time
            return
        
        # Exctract context from question
        context = question.context

        # Extract user and run IDs from metadata to later filter memories
        run_id = question.question_id

        # Convert context to messages format and add in user-assistant pairs
        for session in context:
            # Group turns into user-assistant pairs
            i = -1 # Start before first turn
            session_id = session[0].metadata.get('session_id', 'unknown_session')
            while i < len(session):
                messages = []
                i += 1 # Move to next turn

                # Get user message
                if i < len(session) and session[i].role == "user":
                    messages.append({
                        "role": session[i].role,
                        "content": session[i].content
                    })
                    i += 1 # Advance into assistant paired response
                
                # Get assistant response
                if i < len(session) and session[i].role == "assistant":
                    messages.append({
                        "role": session[i].role,
                        "content": session[i].content
                    })
                
                # Add pair to Mem0 (skip if incomplete)
                if len(messages) == 2:
                    try:
                        self.memory.add(
                            messages=messages,
                            user_id=self._user_id,
                            run_id=run_id,
                            metadata={
                                'session_id': session_id,
                                'turn_index': i // 2
                            }
                        )
                    except Exception as e:
                        # Log error but continue processing other pairs
                        print(f"Warning: Failed to add turn {i//2} from session {session_id} to Mem0: {e}")
                else:
                    # Skip incomplete pairs
                    if messages:
                        print(f"Warning: Incomplete pair in session {session_id}, skipping")
        
        # Calculate and store processing time
        processing_time = time.time() - start_time
        question.metadata['context_processing_time'] = processing_time
    
    def _search_memories(self, question: Question, fromContext: bool = False) -> Dict[str, Any]:
        """
        Search for relevant memories using the question.
        
        Args:
            question: The question to search for
            know: If False, use standard threshold; if True, set threshold to 0.0 to retrieve all memories.
            
        Returns:
            Formatted string of relevant memories
        """
        if self.memory is None:
            raise RuntimeError("Mem0 Memory not initialized")
        
        if not fromContext:
            threshold=self.config.get('search_threshold', 0.3)
        else:
            threshold=0.0
        
        try:
            # Search memories (local SDK returns dict with 'results' key)
            search_results = self.memory.search(
                query=question.question_text,
                user_id=self._user_id,
                run_id=question.question_id,
                limit=self.config.get('search_limit', 5),
                threshold=threshold
            )
            
            # Extract results and relations (if graph is enabled)
            if isinstance(search_results, dict):
                results = search_results.get('results', [])
                relations = search_results.get('relations', []) if self.enableGraph else []
                return {'results': results, 'relations': relations}
            else:
                return {'results': search_results, 'relations': []}
            
        except Exception as e:
            print(f"Warning: Memory search failed: {e}")
            return "Memory search unavailable."
        
    def format_memories(self, search_data: dict) -> str:
        """Format memories and relations into a string for the prompt."""
        results = search_data.get('results', [])
        relations = search_data.get('relations', [])
        
        output_parts = []
        
        # Format memories
        if results and len(results) > 0:
            memory_lines = [f"- {mem['memory']}" for mem in results]
            output_parts.append("\n".join(memory_lines))
        else:
            output_parts.append("No relevant memories found.")
        
        # Format relations (graph data) if available
        if relations and len(relations) > 0:
            output_parts.append("\n\n=== KNOWLEDGE GRAPH ===")
            relation_lines = []
            for rel in relations:
                source = rel.get('source', '').replace('_', ' ')
                relationship = rel.get('relationship', '').replace('_', ' ')
                destination = rel.get('destination', '').replace('_', ' ')
                relation_lines.append(f"- {source} {relationship} {destination}")
            output_parts.append("\n".join(relation_lines))
        
        return "\n".join(output_parts)
            
    
    def answer_question(self, question: Question) -> Answer:
        """
        Answer a question using relevant memories from local Mem0.
        
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
        
        # Search for relevant memories (returns dict with results and relations)
        search_data = self._search_memories(question)
        memories_str = self.format_memories(search_data)
        
        # Extract results for metadata
        results = search_data.get('results', [])
        relations = search_data.get('relations', [])

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
                'memory_system': 'mem0_local',
                'user_id': self._user_id,
                'num_memories_retrieved': len(results),
                'num_relations_retrieved': len(relations) if self.enableGraph else 0,
                'memoriesRetrieved': results,
                'relationsRetrieved': relations if self.enableGraph else [],
                'full_prompt': prompt,
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Mem0 local memory system.
        
        Returns:
            Dictionary with memory stats
        """
        if self.memory is None:
            return {'status': 'not_initialized'}
        
        try:
            # Get all memories for this user
            all_memories = self.memory.get_all(user_id=self._user_id)
            
            # Extract results
            results = all_memories.get('results', []) if isinstance(all_memories, dict) else all_memories
            
            return {
                'status': 'active',
                'mode': 'local',
                'user_id': self._user_id,
                'total_memories': len(results),
                'search_limit': self.config.get('search_limit', 5),
                'search_threshold': self.config.get('search_threshold', 0.3),
                'llm_provider': self.config.get('llm', {}).get('provider'),
                'embedder_provider': self.config.get('embedder', {}).get('provider'),
                'vector_store_provider': self.config.get('vector_store', {}).get('provider'),
                'graph_store_provider': self.config.get('graph_store', {}).get('provider') if self.enableGraph else None
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
        if self.memory is None:
            raise RuntimeError("Mem0 Memory not initialized")
        
        # Only proceed if configured to do so
        if not self.ActivateReset:
            return
        
        try:
            self.memory.reset(user_id=self._user_id)
            print(f"✅ All memories deleted for user: {self._user_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to reset memories: {e}")