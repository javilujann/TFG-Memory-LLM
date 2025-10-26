"""
LongMemEval Dataset Reader

Reads the LongMemEval dataset format and converts it to standardized Question objects.
"""

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
        self._dataset_path: Optional[Path] = None
        self._total_questions: int = 0
        self._question_types: Dict[str, int] = {}
    
    def load(self, path: str, max_questions: Optional[int] = None) -> List[Question]:
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
        dataset_path = Path(path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        # Load JSON data
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of questions, got {type(data)}")
        
        # Store metadata
        self._dataset_path = dataset_path
        self._total_questions = len(data)
        
        # Limit questions if requested
        if max_questions is not None:
            data = data[:max_questions]
        
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
        context = self._parse_sessions(haystack_sessions)
        
        # Build metadata
        metadata = {
            'question_date': entry.get('question_date'),
            'haystack_dates': entry.get('haystack_dates', []),
            'haystack_session_ids': entry.get('haystack_session_ids', []),
            'answer_session_ids': entry.get('answer_session_ids', []),
        }
        
        return Question(
            question_id=question_id,
            question_text=question_text,
            question_type=question_type,
            ground_truth_answer=ground_truth,
            context=context,
            metadata=metadata
        )
    
    def _parse_sessions(self, sessions: List[List[Dict[str, Any]]]) -> List[List[ChatTurn]]:
        """
        Parse haystack sessions into ChatTurn objects.
        
        Args:
            sessions: List of sessions, each containing a list of message dictionaries
            
        Returns:
            List of sessions, each containing ChatTurn objects
        """
        parsed_sessions = []
        
        for session in sessions:
            parsed_turns = []
            for turn in session:
                chat_turn = ChatTurn(
                    role=turn['role'],
                    content=turn['content'],
                    has_answer=turn.get('has_answer'),
                    timestamp=None,  # LongMemEval doesn't include per-turn timestamps
                    metadata={}
                )
                parsed_turns.append(chat_turn)
            
            parsed_sessions.append(parsed_turns)
        
        return parsed_sessions
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded LongMemEval dataset.
        
        Returns:
            Dictionary with dataset statistics and information
        """
        return {
            'dataset_path': str(self._dataset_path) if self._dataset_path else None,
            'total_questions': self._total_questions,
            'question_types': self._question_types.copy(),
            'dataset_name': 'LongMemEval',
            'format_version': '1.0',
        }
