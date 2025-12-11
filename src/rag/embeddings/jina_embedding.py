"""
Jina v2 Embedding Model Integration
Supports jina-ai/jina-embeddings-v2-base-en and fine-tuned models
"""
import sys
from pathlib import Path
from typing import List, Optional, Union
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import config

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Please install: pip install transformers")


class JinaEmbedding:
    """
    Jina v2 Embedding Model Wrapper
    Compatible with LangChain's embedding interface
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize Jina embedding model
        
        Args:
            model_name: HuggingFace model name (default: jinaai/jina-embeddings-v2-base-en)
            model_path: Path to fine-tuned model (optional)
            device: Device to use ('cuda' or 'cpu')
            normalize_embeddings: Whether to normalize embeddings
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        self.model_name = model_name or config.JINA_BASE_MODEL
        self.model_path = model_path or config.JINA_MODEL_PATH
        self.normalize_embeddings = normalize_embeddings
        
        # Device configuration
        if device is None:
            if torch.cuda.is_available() and config.NUM_GPUS > 0:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"🚀 Loading Jina embedding model: {self.model_name}")
        print(f"💻 Using device: {self.device}")
        
        # Load model
        model_path_to_load = self.model_path if self.model_path else self.model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_to_load,
                trust_remote_code=True,
                cache_dir=str(config.MODEL_DIR)
            )
            
            self.model = AutoModel.from_pretrained(
                model_path_to_load,
                trust_remote_code=True,
                cache_dir=str(config.MODEL_DIR)
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Jina model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Jina model: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        Compatible with LangChain interface
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        return self.embed_documents([text])[0]
    
    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Embed a list of documents with automatic batching for memory efficiency
        Compatible with LangChain interface
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (default: from config.EMBEDDING_BATCH_SIZE)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Use config batch size if not specified
        if batch_size is None:
            batch_size = getattr(config, 'EMBEDDING_BATCH_SIZE', 16)
        
        # For small batches, process directly
        if len(texts) <= batch_size:
            return self._embed_batch(texts)
        
        # For large batches, process in chunks
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Clear GPU cache periodically (every 10 batches)
            if self.device == 'cuda' and (i // batch_size + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        return all_embeddings
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method to embed a single batch of texts
        
        Args:
            texts: List of texts to embed (should be <= batch_size)
            
        Returns:
            List of embedding vectors
        """
        # Tokenize
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.TRAINING_MAX_LENGTH,
            return_tensors='pt'
        )
        
        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Jina models typically use mean pooling
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize if needed
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to CPU and list immediately to free GPU memory
        embeddings_list = embeddings.cpu().numpy().tolist()
        
        return embeddings_list
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling for sentence embeddings
        
        Args:
            model_output: Model output containing token embeddings
            attention_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Embed texts in batches (for large datasets)
        This is now a wrapper around embed_documents which handles batching internally
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (default: 16)
            
        Returns:
            List of embedding vectors
        """
        return self.embed_documents(texts, batch_size=batch_size)

