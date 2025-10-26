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
        # TODO: Initialize reader
        pass
    
    def load(self, path: str, max_questions: Optional[int] = None) -> List[Question]:
        """
        Load LongMemEval dataset from JSON file.
        
        Args:
            path: Path to the dataset JSON file
            max_questions: Maximum number of questions to load
            
        Returns:
            List of Question objects
        """
        # TODO: Implement loading logic
        # 1. Load JSON file
        # 2. Parse each entry
        # 3. Convert to Question objects with ChatTurn context
        # 4. Return list of questions
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the LongMemEval dataset"""
        # TODO: Return dataset metadata
        pass
