"""
LongMemEval Dataset Reader

Reads the LongMemEval dataset format and converts it to standardized Question objects.
"""

from importlib.resources import path
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from ..core.interfaces import DatasetReader
from ..core.models import Question, ChatTurn


class LongMemEvalReader(DatasetReader):
    """
    Reader for LongMemEval dataset format.
    
    Handles the oracle, small (s), and medium (m) dataset variants.
    """
    
    def __init__(self):
        """Initialize the LongMemEval reader."""
        self._total_questions: int = 0
        self._question_types: Dict[str, int] = {}
        self.config: Dict[str, Any] = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize reader with configuration """
        
        if 'dataset_path' not in config or config['dataset_path'] is None:
            raise ValueError("LongMemEvalReader requires 'dataset_path' in config")
        
        self.config = config
       
    
    def load(self) -> List[Question]:
        """
        Load LongMemEval dataset from JSON file.
        
        Args:
            path: Path to the dataset JSON file
            max_questions: Maximum number of questions to load (None for all)
            
        Returns:
            List of Question objects
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        # Load dataset file
        dataset_path = Path(self.config['dataset_path'])
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.config['dataset_path']}")

        # Load JSON data from file
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of questions, got {type(data)}")
    
        # Apply dataset filters if provided
        filters = self.config.get('filters') or {}
        if filters:
            data = self._apply_filters(data, filters)

        # Limit questions if requested
        max_questions = self.config.get('max_questions', None)
        if max_questions is not None:
            data = data[:max_questions]

        # Store metadata after filters/limits are applied
        self._total_questions = len(data)
        
        # Convert to Question objects
        questions = []
        for entry in data:
            question = self._parse_entry(entry)
            questions.append(question)
            
            # Track question types
            q_type = question.question_type
            self._question_types[q_type] = self._question_types.get(q_type, 0) + 1
        
        return questions
    
    def _parse_entry(self, entry: Dict[str, Any]) -> Question:
        """
        Parse a single LongMemEval entry into a Question object.
        
        Args:
            entry: Raw dictionary entry from the dataset
            
        Returns:
            Question object
        """
        # Extract basic fields
        question_id = entry['question_id']
        question_text = entry['question']
        question_type = entry['question_type']
        ground_truth = entry['answer']
        
        # Parse haystack sessions (context)
        haystack_sessions = entry.get('haystack_sessions', [])
        haystack_session_ids = entry.get('haystack_session_ids', [])
        context, answer_locations = self._parse_sessions(haystack_sessions, haystack_session_ids)
        
        # Build metadata
        metadata = {
            'question_date': entry.get('question_date'),
            'haystack_dates': entry.get('haystack_dates', []),
            'answer_session_ids': entry.get('answer_session_ids', []),
            'answer_locations': answer_locations,
        }
        
        return Question(
            question_id=question_id,
            question_text=question_text,
            question_type=question_type,
            ground_truth_answer=ground_truth,
            context=context,
            metadata=metadata
        )
    
    def _parse_sessions(self, sessions: List[List[Dict[str, Any]]], session_ids: List[str]) -> tuple[List[List[ChatTurn]], List[tuple[str, int]]]:
        """
        Parse haystack sessions into ChatTurn objects.
        
        Args:
            sessions: List of sessions, each containing a list of message dictionaries
            session_ids: List of session IDs corresponding to each session
            
        Returns:
            Tuple of:
            - List of sessions, each containing ChatTurn objects
            - List of (session_id, pair_idx) tuples for turns with has_answer=True
              where pair_idx corresponds to user-assistant pair index (turn_idx // 2)
        """
        parsed_sessions = []
        answer_locations = []
        
        for session_idx, session in enumerate(sessions):
            parsed_turns = []
            # Get session_id if available, otherwise use index
            session_id = session_ids[session_idx] if session_idx < len(session_ids) else f"session_{session_idx}"
            
            for turn_idx, turn in enumerate(session):
                has_answer = turn.get('has_answer')
                
                chat_turn = ChatTurn(
                    role=turn['role'],
                    content=turn['content'],
                    has_answer=has_answer,
                    timestamp=None,  # LongMemEval doesn't include per-turn timestamps
                    metadata={'session_id': session_id}
                )
                parsed_turns.append(chat_turn)
                
                # Track answer locations with pair index (to match memory storage format)
                if has_answer:
                    pair_idx = turn_idx // 2  # Convert to user-assistant pair index
                    answer_locations.append((session_id, pair_idx))
            
            parsed_sessions.append(parsed_turns)
        
        return parsed_sessions, answer_locations

    def _apply_filters(self, data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply dataset filters before parsing entries.
        Order: question_index_range (on original data) -> question_ids(on filtered data)

        Supported filters:
            - question_index_range: [start, end] (inclusive, 1-based on original dataset; e.g., [10, 15] selects positions 10-15)
            - question_ids: list[str] or str (applied after index range)
        """
        filtered = data

        # Step 1: Apply index range FIRST (on original data)
        index_range = filters.get('question_index_range')
        if index_range is not None:
            if not isinstance(index_range, (list, tuple)) or len(index_range) != 2:
                raise ValueError("filters.question_index_range must be [start, end]")
            start, end = index_range
            if start is None:
                start = 1
            if end is None:
                end = len(filtered)
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("filters.question_index_range values must be integers or None")
            if start < 1 or end < start or end > len(filtered):
                raise ValueError(f"filters.question_index_range must be 1-based within list bounds (1 to {len(filtered)}), start <= end")
            # Convert from 1-based to 0-based for slicing
            filtered = filtered[start - 1:end]

        # Step 2: Apply question_ids filter
        question_ids = filters.get('question_ids')
        if isinstance(question_ids, str):
            question_ids = [question_ids]
        if question_ids:
            question_id_set = set(question_ids)
            filtered = [entry for entry in filtered if entry.get('question_id') in question_id_set]

        return filtered
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded LongMemEval dataset.
        
        Returns:
            Dictionary with dataset statistics and information
        """
        return {
            'dataset_path': self.config.get('dataset_path', ''),
            'total_questions': self._total_questions,
            'question_types': self._question_types.copy(),
            'dataset_name': 'LongMemEval',
            'format_version': '1.0',
        }
