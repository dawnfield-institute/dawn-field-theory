"""
Embedding Generation for PAC Nodes

Provides methods for generating high-dimensional embeddings from information states.
Supports multiple embedding strategies: pretrained models, custom learned, synthetic.
"""

import numpy as np
from typing import Dict, Optional, List, Callable
from abc import ABC, abstractmethod
import warnings
import json
import subprocess

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available - only synthetic embeddings will work")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .pac_hierarchy import PACNode, PACHierarchy


class EmbeddingStrategy(ABC):
    """Abstract base class for embedding generation strategies."""
    
    @abstractmethod
    def embed(self, text: str, node: PACNode) -> np.ndarray:
        """Generate embedding for a node."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimensionality."""
        pass


class SyntheticEmbedding(EmbeddingStrategy):
    """
    Synthetic embedding that respects PAC conservation by construction.
    
    Generates embeddings where parent = weighted sum of children embeddings.
    Useful for validation with ground truth.
    """
    
    def __init__(self, dimension: int = 128, seed: Optional[int] = None):
        """
        Initialize synthetic embedder.
        
        Args:
            dimension: Embedding dimensionality
            seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.rng = np.random.RandomState(seed)
        self._cache: Dict[str, np.ndarray] = {}
    
    def embed(self, text: str, node: PACNode) -> np.ndarray:
        """
        Generate synthetic embedding.
        
        For leaf nodes: random vector normalized by value.
        For parent nodes: computed from children to satisfy conservation.
        """
        if node.id in self._cache:
            return self._cache[node.id]
        
        if not node.children:
            # Leaf node: generate random embedding scaled by value
            embedding = self.rng.randn(self.dimension)
            embedding = embedding / np.linalg.norm(embedding) * np.sqrt(node.value)
        else:
            # Parent node: compute from children
            child_embeddings = [self.embed("", child) for child in node.children]
            weights = [
                child.ownership_weights.get(node.id, 1.0) 
                for child in node.children
            ]
            embedding = sum(w * e for w, e in zip(weights, child_embeddings))
        
        self._cache[node.id] = embedding
        return embedding
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()


class OllamaEmbedding(EmbeddingStrategy):
    """
    Use Ollama API for embeddings.
    
    Supports any Ollama model with embedding capabilities.
    Generates embeddings for leaf nodes, computes parent embeddings
    as weighted sums (PAC-preserving composition).
    """
    
    def __init__(self, model_name: str = 'llama3.2:latest', ollama_host: str = 'http://localhost:11434'):
        """
        Initialize Ollama embedder.
        
        Args:
            model_name: Ollama model name (e.g., 'llama3.2:latest', 'phi3:medium')
            ollama_host: Ollama API endpoint
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self._cache: Dict[str, np.ndarray] = {}
        self._dimension: Optional[int] = None
        
        # Test connection and get dimension
        test_embedding = self._get_ollama_embedding("test")
        if test_embedding is not None:
            self._dimension = len(test_embedding)
            print(f"Ollama {model_name}: {self._dimension}D embeddings")
        else:
            raise RuntimeError(f"Failed to connect to Ollama at {ollama_host}")
    
    def _get_ollama_embedding(self, text: str) -> Optional[np.ndarray]:
        """Call Ollama API for embedding."""
        try:
            import requests
            response = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return np.array(data['embedding'], dtype=np.float32)
            else:
                print(f"Ollama API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Ollama embedding failed: {e}")
            return None
    
    def embed(self, text: str, node: PACNode) -> np.ndarray:
        """
        Generate embedding for node.
        
        For leaf nodes: get embedding from Ollama
        For parent nodes: weighted sum of children (PAC-preserving)
        """
        if node.id in self._cache:
            return self._cache[node.id]
        
        if not node.children:
            # Leaf node: get embedding from Ollama
            if not text:
                text = f"concept_{node.id}"
            
            embedding = self._get_ollama_embedding(text)
            if embedding is None:
                # Fallback to random if API fails
                embedding = np.random.randn(self._dimension or 4096).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding) * np.sqrt(node.value)
        else:
            # Parent node: weighted sum of children (preserves PAC)
            child_embeddings = [self.embed("", child) for child in node.children]
            weights = [
                child.ownership_weights.get(node.id, 1.0) 
                for child in node.children
            ]
            embedding = sum(w * e for w, e in zip(weights, child_embeddings)).astype(np.float32)
        
        self._cache[node.id] = embedding
        return embedding
    
    def get_dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Dimension not initialized")
        return self._dimension
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()


class PretrainedEmbedding(EmbeddingStrategy):
    """
    Use pretrained language model embeddings.
    
    Supports BERT, GPT, and other transformers models.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', layer: int = -1):
        """
        Initialize pretrained embedder.
        
        Args:
            model_name: HuggingFace model identifier
            layer: Which layer to extract embeddings from (-1 = last)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required for PretrainedEmbedding")
        
        self.model_name = model_name
        self.layer = layer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self._cache: Dict[str, np.ndarray] = {}
    
    def embed(self, text: str, node: PACNode) -> np.ndarray:
        """Generate embedding from text using pretrained model."""
        cache_key = f"{node.id}:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        # Get hidden states
        import torch
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer]
        
        # Use [CLS] token or mean pooling
        embedding = hidden_states[0, 0, :].cpu().numpy()  # [CLS] token
        
        self._cache[cache_key] = embedding
        return embedding
    
    def get_dimension(self) -> int:
        return self.model.config.hidden_size
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()


class SentenceEmbedding(EmbeddingStrategy):
    """
    Use sentence-transformers for semantic embeddings.
    
    More efficient than full transformer models for sentence-level embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize sentence embedder.
        
        Args:
            model_name: sentence-transformers model name
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package required for SentenceEmbedding")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._cache: Dict[str, np.ndarray] = {}
    
    def embed(self, text: str, node: PACNode) -> np.ndarray:
        """Generate sentence embedding."""
        cache_key = f"{node.id}:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        self._cache[cache_key] = embedding
        return embedding
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()


class EmbeddingGenerator:
    """
    Main interface for generating embeddings for PAC hierarchies.
    
    Supports multiple strategies and provides convenience methods.
    """
    
    def __init__(
        self,
        strategy: Optional[EmbeddingStrategy] = None,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: int = 128,
        seed: Optional[int] = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            strategy: Custom embedding strategy (overrides other args)
            model: Model type ('sentence-transformers', 'bert', 'synthetic', 'ollama')
            model_name: Specific model name for the chosen type
            dimension: For synthetic embeddings
            seed: Random seed
        """
        if strategy is not None:
            self.strategy = strategy
        elif model == 'sentence-transformers':
            model_name = model_name or 'all-MiniLM-L6-v2'
            self.strategy = SentenceEmbedding(model_name=model_name)
        elif model == 'ollama':
            model_name = model_name or 'llama3.2:latest'
            self.strategy = OllamaEmbedding(model_name=model_name)
        elif model == 'synthetic' or model is None:
            self.strategy = SyntheticEmbedding(dimension=dimension, seed=seed)
        elif model.startswith('bert') or model.startswith('gpt'):
            self.strategy = PretrainedEmbedding(model_name=model)
        else:
            # Try sentence-transformers
            self.strategy = SentenceEmbedding(model_name=model)
    
    def embed_node(self, node: PACNode, text: Optional[str] = None) -> np.ndarray:
        """
        Generate embedding for a single node.
        
        Args:
            node: Node to embed
            text: Optional text content (defaults to node.id)
        
        Returns:
            Embedding vector
        """
        if text is None:
            text = node.id
        
        embedding = self.strategy.embed(text, node)
        node.embedding = embedding
        return embedding
    
    def embed_hierarchy(
        self,
        hierarchy: PACHierarchy,
        text_fn: Optional[Callable[[PACNode], str]] = None
    ):
        """
        Generate embeddings for all nodes in hierarchy.
        
        Args:
            hierarchy: Hierarchy to embed
            text_fn: Optional function to extract text from node
                    Defaults to using node.id or node.metadata['text']
        """
        if text_fn is None:
            text_fn = lambda n: n.metadata.get('text', n.id)
        
        # For synthetic embeddings, process bottom-up to ensure conservation
        if isinstance(self.strategy, SyntheticEmbedding):
            # Get levels in reverse order (deepest first)
            levels = hierarchy.get_levels()
            for level in reversed(levels):
                for node in level:
                    text = text_fn(node)
                    self.embed_node(node, text)
        else:
            # For other strategies, order doesn't matter
            for node in hierarchy.nodes.values():
                text = text_fn(node)
                self.embed_node(node, text)
    
    def get_dimension(self) -> int:
        """Return embedding dimensionality."""
        return self.strategy.get_dimension()
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.strategy.clear_cache()


def create_synthetic_hierarchy_with_embeddings(
    depth: int = 3,
    branching: int = 3,
    dimension: int = 128,
    seed: Optional[int] = None
) -> PACHierarchy:
    """
    Create synthetic hierarchy with PAC-compliant embeddings.
    
    Useful for testing where ground truth is known.
    
    Args:
        depth: Maximum depth of tree
        branching: Number of children per node
        dimension: Embedding dimensionality
        seed: Random seed
    
    Returns:
        PACHierarchy with embeddings already generated
    """
    rng = np.random.RandomState(seed)
    
    def create_subtree(node_id: str, value: float, current_depth: int) -> PACNode:
        node = PACNode(id=node_id, value=value, depth=current_depth)
        
        if current_depth < depth:
            # Create children
            child_values = rng.dirichlet(np.ones(branching)) * value
            
            for i, child_value in enumerate(child_values):
                child_id = f"{node_id}_{i}"
                child = create_subtree(child_id, child_value, current_depth + 1)
                node.add_child(child)
        
        return node
    
    root = create_subtree("root", 1.0, 0)
    hierarchy = PACHierarchy(root)
    
    # Collect all nodes
    def collect_nodes(node: PACNode):
        hierarchy.nodes[node.id] = node
        for child in node.children:
            collect_nodes(child)
    
    collect_nodes(root)
    
    # Generate embeddings bottom-up to ensure conservation
    embedder = EmbeddingGenerator(model='synthetic', dimension=dimension, seed=seed)
    embedder.embed_hierarchy(hierarchy)
    
    return hierarchy
