"""
Configuration loader for Privacy-Preserving Agentic RAG.
Loads settings from config.json at the project root.
"""

import json
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager that loads from config.json."""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from config.json."""
        # Find project root (parent of src directory)
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found at {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'ingestion.chunk_size')."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_path(self, key: str) -> Path:
        """Get a path configuration value as a Path object relative to project root."""
        project_root = Path(__file__).parent.parent
        relative_path = self.get(key)
        if relative_path:
            return project_root / relative_path
        return project_root
    
    @property
    def data_dir(self) -> Path:
        return self.get_path("paths.data_dir")
    
    @property
    def chroma_dir(self) -> Path:
        return self.get_path("paths.chroma_dir")
    
    @property
    def embedding_model(self) -> str:
        return self.get("ingestion.embedding_model")
    
    @property
    def chunk_size(self) -> int:
        return self.get("ingestion.chunk_size")
    
    @property
    def chunk_overlap(self) -> int:
        return self.get("ingestion.chunk_overlap")
    
    @property
    def text_splitter_separators(self) -> list:
        return self.get("ingestion.text_splitter_separators")
    
    @property
    def top_k(self) -> int:
        return self.get("retrieval.top_k")
    
    @property
    def retrieval_embedding_model(self) -> str:
        return self.get("retrieval.embedding_model")
    
    @property
    def local_llm_model(self) -> str:
        return self.get("models.local_llm_model")
    
    @property
    def cloud_llm_model(self) -> str:
        return self.get("models.cloud_llm_model")
    
    @property
    def pii_entity_types(self) -> list:
        return self.get("pii_masking.entity_types")
    
    @property
    def pii_token_format(self) -> str:
        return self.get("pii_masking.token_format")
    
    @property
    def ollama_base_url(self) -> str:
        return self.get("streamlit.ollama_base_url")


# Global config instance
config = Config()