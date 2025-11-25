"""
Data loading module
Load ScienceQA dataset
"""
import json
from pathlib import Path
from typing import Dict, Optional

import config


class ScienceQADataLoader:
    """ScienceQA data loader"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader
        
        Args:
            data_dir: Data directory path
        """
        self.data_dir = data_dir or config.SCIENCEQA_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_json(self, file_path: Path) -> Dict:
        """
        Load JSON file
        
        Args:
            file_path: JSON file path
            
        Returns:
            Loaded JSON data dictionary
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_problems(self) -> Dict:
        """Load problems.json"""
        return self.load_json(config.PROBLEMS_JSON)
    
    def load_captions(self) -> Dict:
        """Load captions.json"""
        return self.load_json(config.CAPTIONS_JSON)
    
    def load_pid_splits(self) -> Dict:
        """Load pid_splits.json"""
        return self.load_json(config.PID_SPLITS_JSON)
    
    def get_split_problems(self, split: str = "train") -> Dict:
        """
        Get problems for specified split
        
        Args:
            split: Dataset split (train/val/test)
            
        Returns:
            Problem dictionary
        """
        problems = self.load_problems()
        splits = self.load_pid_splits()
        
        if split not in splits:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(splits.keys())}")
        
        split_ids = splits[split]
        return {pid: problems[pid] for pid in split_ids if pid in problems}

