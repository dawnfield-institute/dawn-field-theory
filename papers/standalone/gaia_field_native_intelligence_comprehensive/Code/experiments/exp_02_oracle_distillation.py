"""
Oracle-Bootstrapped Hierarchical PAC Training
==============================================

Use source models (GPT-2, Pythia) as ORACLES to learn:
- What IS language (embedding structure)
- What IS grammar (attention patterns)
- How to compose (layer structure)

The PAC-Lazy layers grow based on oracle matching, NOT training data.
We only use training data AFTER bootstrapping from oracles.

NO BACKPROP on the student - we use SEC-PAC dynamics.
The oracle just provides the LOSS FUNCTION to guide growth.
"""

import sys
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "poc_017_pac_import"))
sys.path.insert(0, r"c:\Users\peter\repos\core_workspace\fracton")

# Constants
PHI = (1 + np.sqrt(5)) / 2
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


class OracleProber:
    """Generates probes to learn from oracle models using their actual vocabularies"""
    
    def __init__(self, vocab_size: int, device, tokenizers: Dict[str, any] = None):
        self.vocab_size = vocab_size
        self.device = device
        self.tokenizers = tokenizers or {}
        
        # Extract vocabularies from each model
        self.model_vocabs = {}
        self.shared_tokens = set()
        self.common_words = []
        
        self._extract_vocabularies()
        
    def _extract_vocabularies(self):
        """Extract actual vocabulary from each model's tokenizer"""
        if not self.tokenizers:
            return
            
        print("\n  Extracting model vocabularies...")
        
        all_tokens = []
        for name, tokenizer in self.tokenizers.items():
            if tokenizer is None:
                continue
                
            # Get vocabulary
            vocab = tokenizer.get_vocab()
            self.model_vocabs[name] = {
                'tokens': vocab,
                'size': len(vocab),
                'common': []  # Will store common/frequent tokens
            }
            
            # Extract tokens that look like real words (no weird symbols)
            word_tokens = []
            for token, idx in vocab.items():
                # Clean token (remove special prefixes like Ġ for GPT-2)
                clean = token.replace('Ġ', ' ').replace('▁', ' ').strip()
                # Keep if it's a real-looking word
                if clean and len(clean) > 1 and clean.isalpha():
                    word_tokens.append((idx, clean.lower()))
                    
            self.model_vocabs[name]['word_tokens'] = word_tokens
            all_tokens.append(set(t[1] for t in word_tokens))
            print(f"    {name}: {len(vocab)} tokens, {len(word_tokens)} word-like tokens")
            
        # Find tokens shared across models
        if len(all_tokens) >= 2:
            self.shared_tokens = all_tokens[0]
            for tokens in all_tokens[1:]:
                self.shared_tokens = self.shared_tokens.intersection(tokens)
            print(f"    Shared across models: {len(self.shared_tokens)} tokens")
            
        # Build common English words list (high frequency)
        self.common_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "can", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "about", "like", "through", "after",
            "over", "between", "out", "against", "during", "before", "under",
            "and", "but", "or", "if", "because", "that", "which", "who", "what",
            "this", "these", "those", "it", "its", "they", "them", "their",
            "he", "she", "his", "her", "we", "our", "you", "your", "I", "my",
            "not", "no", "yes", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "only", "own", "same", "so", "than",
            "very", "just", "now", "then", "here", "there", "when", "where", "how",
            "time", "year", "people", "way", "day", "man", "woman", "child", "world",
            "life", "hand", "part", "place", "case", "week", "company", "system",
            "program", "question", "work", "government", "number", "night", "point",
            "home", "water", "room", "mother", "area", "money", "story", "fact",
            "month", "lot", "right", "study", "book", "eye", "job", "word", "business",
            "issue", "side", "kind", "head", "house", "service", "friend", "father",
            "power", "hour", "game", "line", "end", "member", "law", "car", "city",
            "name", "president", "team", "minute", "idea", "kid", "body", "information",
            "back", "parent", "face", "others", "level", "office", "door", "health",
            "person", "art", "war", "history", "party", "result", "change", "morning",
        ]
        
        # Real text prompts for learning language structure
        self.text_prompts = [
            "The cat sat on the mat.",
            "Scientists study the natural world.",
            "Language is a tool for communication.",
            "In the future, technology will advance.",
            "The ocean is vast and deep.",
            "Music brings people together.",
            "Books contain knowledge and wisdom.",
            "The sun shines brightly in the sky.",
            "Children learn through play and exploration.",
            "History teaches us important lessons.",
            "Art expresses human creativity.",
            "Nature is beautiful and diverse.",
            "Education opens doors to opportunity.",
            "Friendship is a valuable treasure.",
            "Time passes quickly when we are busy.",
            "Health is more important than wealth.",
            "Dreams inspire us to achieve greatness.",
            "Love connects people across distances.",
            "Wisdom comes from experience and reflection.",
            "Curiosity drives scientific discovery.",
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "A journey of a thousand miles begins with a single step.",
        ]
        
    def vocabulary_tokens(self, model_name: str, batch_size: int = 16, seq_len: int = 32) -> torch.Tensor:
        """Probe with actual vocabulary tokens from a specific model"""
        if model_name not in self.model_vocabs or model_name not in self.tokenizers:
            return self.common_tokens(batch_size, seq_len)
            
        word_tokens = self.model_vocabs[model_name].get('word_tokens', [])
        if not word_tokens:
            return self.common_tokens(batch_size, seq_len)
            
        # Sample from the model's word-like tokens
        token_ids = [t[0] for t in word_tokens]
        result = []
        for _ in range(batch_size):
            seq = [random.choice(token_ids) for _ in range(seq_len)]
            result.append(seq)
            
        return torch.tensor(result, device=self.device)
        
    def shared_vocabulary_tokens(self, batch_size: int = 16, seq_len: int = 32) -> torch.Tensor:
        """Probe with tokens shared across all models - these are universally understood"""
        if not self.shared_tokens or not self.tokenizers:
            return self.common_tokens(batch_size, seq_len)
            
        # Use first tokenizer to convert shared words to IDs
        tokenizer = list(self.tokenizers.values())[0]
        if tokenizer is None:
            return self.common_tokens(batch_size, seq_len)
            
        shared_list = list(self.shared_tokens)
        result = []
        for _ in range(batch_size):
            words = [random.choice(shared_list) for _ in range(seq_len)]
            text = ' '.join(words)
            tokens = tokenizer.encode(text)[:seq_len]
            if len(tokens) < seq_len:
                tokens = tokens + [tokenizer.eos_token_id or 0] * (seq_len - len(tokens))
            result.append(tokens)
            
        return torch.tensor(result, device=self.device)
        
    def common_english_tokens(self, batch_size: int = 16, seq_len: int = 32) -> torch.Tensor:
        """Probe with common English words - universal language patterns"""
        if not self.tokenizers:
            return self.common_tokens(batch_size, seq_len)
            
        tokenizer = list(self.tokenizers.values())[0]
        if tokenizer is None:
            return self.common_tokens(batch_size, seq_len)
            
        result = []
        for _ in range(batch_size):
            words = [random.choice(self.common_words) for _ in range(seq_len // 2)]
            text = ' '.join(words)
            tokens = tokenizer.encode(text)[:seq_len]
            if len(tokens) < seq_len:
                tokens = tokens + [tokenizer.eos_token_id or 0] * (seq_len - len(tokens))
            result.append(tokens)
            
        return torch.tensor(result, device=self.device)
        
    def random_tokens(self, batch_size: int = 16, seq_len: int = 32) -> torch.Tensor:
        """Random token sequences for broad coverage"""
        return torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        
    def common_tokens(self, batch_size: int = 16, seq_len: int = 32) -> torch.Tensor:
        """Common tokens (low IDs = frequent)"""
        return torch.randint(0, min(1000, self.vocab_size), (batch_size, seq_len), device=self.device)
        
    def structured_tokens(self, batch_size: int = 16, seq_len: int = 32) -> torch.Tensor:
        """Repeating patterns"""
        base = torch.randint(0, min(1000, self.vocab_size), (batch_size, seq_len // 4), device=self.device)
        return base.repeat(1, 4)
        
    def real_text_tokens(self, batch_size: int = 16, seq_len: int = 32) -> torch.Tensor:
        """Real text tokenized - this is crucial for learning real language"""
        if not self.tokenizers:
            return self.common_tokens(batch_size, seq_len)
            
        tokenizer = list(self.tokenizers.values())[0]
        if tokenizer is None:
            return self.common_tokens(batch_size, seq_len)
            
        tokens_list = []
        for i in range(batch_size):
            text = self.text_prompts[i % len(self.text_prompts)]
            tokens = tokenizer.encode(text)
            # Pad or truncate to seq_len
            if len(tokens) < seq_len:
                tokens = tokens + [tokenizer.eos_token_id or 0] * (seq_len - len(tokens))
            else:
                tokens = tokens[:seq_len]
            tokens_list.append(tokens)
            
        return torch.tensor(tokens_list, device=self.device)


class HierarchicalSECCollapse:
    """SEC collapse at different hierarchy levels"""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.level_scales = {0: 1.0, 1: PHI, 2: PHI**2, 3: PHI**3}
        
    def collapse(self, x: torch.Tensor, level: int = 0) -> torch.Tensor:
        scale = self.level_scales.get(level, 1.0)
        norm = x.norm(dim=-1, keepdim=True) + 1e-8
        return (x / norm) * scale


class PACLazyLayer:
    """PAC-Lazy layer that materializes based on oracle matching"""
    
    def __init__(self, dim: int, num_heads: int, layer_idx: int, device):
        self.dim = dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.device = device
        
        self.materialized = False
        self.oracle_pattern = None  # Learned from oracle
        self.activations = 0
        
        # Attention weights (learned from oracle without backprop)
        self.attn_weights = None
        
    def materialize_from_oracle(self, oracle_attention: torch.Tensor):
        """Materialize layer from oracle attention pattern"""
        with torch.no_grad():
            # Store the oracle's attention pattern
            self.oracle_pattern = oracle_attention.detach().cpu()
            
            # Create attention weights that mimic oracle
            # Average across batch and heads
            if len(oracle_attention.shape) == 4:  # [batch, heads, seq, seq]
                avg_attn = oracle_attention.mean(dim=(0, 1))
            else:
                avg_attn = oracle_attention.mean(dim=0)
                
            self.attn_weights = avg_attn.to(self.device)
            self.materialized = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.materialized:
            return x
            
        self.activations += 1
        batch, seq, dim = x.shape
        
        # Use learned attention weights
        if self.attn_weights is not None and self.attn_weights.shape[0] >= seq:
            attn = self.attn_weights[:seq, :seq]
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, x)
        else:
            # Fallback: simple self-attention
            attn = torch.softmax(torch.matmul(x, x.transpose(-2, -1)) / np.sqrt(dim), dim=-1)
            out = torch.matmul(attn, x)
            
        return out


class OracleBootstrappedModel:
    """Model that learns from oracles using SEC-PAC dynamics"""
    
    def __init__(self, device):
        self.device = device
        self.vocab_size = 50257
        self.embed_dim = 256
        self.num_heads = 8
        self.max_layers = 13  # Fibonacci target
        
        # Embeddings
        self.embeddings = torch.randn(self.vocab_size, self.embed_dim, device=device) * 0.02
        
        # PAC-Lazy layers
        self.layers = [PACLazyLayer(self.embed_dim, self.num_heads, i, device) 
                       for i in range(self.max_layers)]
        self.current_layers = 1
        
        # Confluence trees
        self.token_confluence = {}
        self.attention_confluence = {}  # Learned from oracle attention
        
        # SEC collapse
        self.sec_collapse = HierarchicalSECCollapse(self.embed_dim)
        
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embeddings[input_ids]
        
        for i in range(self.current_layers):
            if self.layers[i].materialized:
                x = self.layers[i].forward(x)
                
        logits = torch.matmul(x, self.embeddings.T)
        return x, logits
        
    def grow_layer(self) -> bool:
        if self.current_layers >= self.max_layers:
            return False
        self.current_layers += 1
        return True


class OracleDistillationTrainer:
    """
    Train GAIA by distilling from oracle models.
    
    Key insight: Use oracle as LOSS FUNCTION, but don't backprop.
    Instead, use SEC-PAC dynamics to:
    1. Probe oracle with diverse inputs
    2. Record oracle's attention patterns
    3. Materialize layers that match oracle behavior
    4. Build confluence from oracle predictions
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load oracles
        self.oracles = {}
        self._load_oracles()
        
        # Create model
        self.model = OracleBootstrappedModel(self.device)
        
        # Transfer embeddings from oracles
        self._transfer_embeddings()
        
        # Prober - pass all tokenizers for vocabulary-based probing
        tokenizers = {name: oracle['tokenizer'] for name, oracle in self.oracles.items()}
        self.prober = OracleProber(self.model.vocab_size, self.device, tokenizers)
        
    def _load_oracles(self):
        """Load oracle models"""
        print("\nLoading oracle models...")
        
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self.oracles['gpt2'] = {
                'model': GPT2LMHeadModel.from_pretrained('gpt2').to(self.device).eval(),
                'tokenizer': GPT2Tokenizer.from_pretrained('gpt2')
            }
            for p in self.oracles['gpt2']['model'].parameters():
                p.requires_grad = False
            print(f"  GPT-2: loaded")
        except Exception as e:
            print(f"  GPT-2 failed: {e}")
            
        try:
            from transformers import GPTNeoXForCausalLM, AutoTokenizer
            self.oracles['pythia'] = {
                'model': GPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-70m').to(self.device).eval(),
                'tokenizer': AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
            }
            for p in self.oracles['pythia']['model'].parameters():
                p.requires_grad = False
            print(f"  Pythia: loaded")
        except Exception as e:
            print(f"  Pythia failed: {e}")
            
    def _transfer_embeddings(self):
        """Transfer embeddings from oracles"""
        print("\nTransferring embeddings from oracles...")
        embeddings = []
        
        if 'gpt2' in self.oracles:
            gpt2_emb = self.oracles['gpt2']['model'].transformer.wte.weight.detach()
            gpt2_emb = gpt2_emb[:self.model.vocab_size, :self.model.embed_dim]
            embeddings.append(gpt2_emb)
            print(f"  GPT-2: {gpt2_emb.shape}")
            
        if 'pythia' in self.oracles:
            pythia_emb = self.oracles['pythia']['model'].gpt_neox.embed_in.weight.detach()
            if pythia_emb.shape[1] < self.model.embed_dim:
                pad = torch.zeros(pythia_emb.shape[0], self.model.embed_dim - pythia_emb.shape[1], device=self.device)
                pythia_emb = torch.cat([pythia_emb.to(self.device), pad], dim=1)
            else:
                pythia_emb = pythia_emb[:, :self.model.embed_dim]
            pythia_emb = pythia_emb[:self.model.vocab_size]
            embeddings.append(pythia_emb.to(self.device))
            print(f"  Pythia: {pythia_emb.shape}")
            
        if embeddings:
            with torch.no_grad():
                avg = torch.stack([e.to(self.device) for e in embeddings]).mean(dim=0)
                self.model.embeddings = avg
                print(f"  Transferred avg of {len(embeddings)} models")
                
    def get_oracle_attention(self, oracle_name: str, tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention patterns from oracle"""
        if oracle_name not in self.oracles:
            return None
            
        oracle = self.oracles[oracle_name]['model']
        
        with torch.no_grad():
            if oracle_name == 'gpt2':
                outputs = oracle(tokens, output_attentions=True)
                # Stack all layer attentions
                attentions = outputs.attentions  # Tuple of [batch, heads, seq, seq]
                return torch.stack(attentions)  # [layers, batch, heads, seq, seq]
            elif oracle_name == 'pythia':
                outputs = oracle(tokens, output_attentions=True)
                attentions = outputs.attentions
                return torch.stack(attentions)
                
        return None
        
    def get_oracle_predictions(self, oracle_name: str, tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """Get next-token predictions from oracle"""
        if oracle_name not in self.oracles:
            return None
            
        oracle = self.oracles[oracle_name]['model']
        
        with torch.no_grad():
            outputs = oracle(tokens)
            return outputs.logits
            
    def compute_oracle_loss(self, student_logits: torch.Tensor, 
                            oracle_logits: torch.Tensor) -> float:
        """Compute how well student matches oracle (for growth decisions)"""
        with torch.no_grad():
            # Match vocab sizes
            min_vocab = min(student_logits.shape[-1], oracle_logits.shape[-1])
            student_logits = student_logits[:, :, :min_vocab]
            oracle_logits = oracle_logits[:, :, :min_vocab]
            
            # KL divergence
            student_probs = F.log_softmax(student_logits / 2.0, dim=-1)
            oracle_probs = F.softmax(oracle_logits / 2.0, dim=-1)
            kl = F.kl_div(student_probs, oracle_probs, reduction='batchmean')
            return kl.item()
            
    def learn_from_oracle_attention(self, oracle_name: str, probe_tokens: torch.Tensor):
        """Learn attention patterns from oracle - NO BACKPROP"""
        attentions = self.get_oracle_attention(oracle_name, probe_tokens)
        if attentions is None:
            return
            
        # Materialize layers based on oracle attention
        num_oracle_layers = attentions.shape[0]
        
        for i in range(min(self.model.current_layers, num_oracle_layers)):
            if not self.model.layers[i].materialized:
                layer_attn = attentions[i]  # [batch, heads, seq, seq]
                self.model.layers[i].materialize_from_oracle(layer_attn)
                print(f"  Layer {i} materialized from {oracle_name}")
                
    def learn_from_oracle_predictions(self, oracle_name: str, probe_tokens: torch.Tensor):
        """Learn token transitions from oracle predictions - NO BACKPROP"""
        oracle_logits = self.get_oracle_predictions(oracle_name, probe_tokens)
        if oracle_logits is None:
            return
            
        with torch.no_grad():
            # Get oracle's predicted next tokens
            oracle_preds = oracle_logits.argmax(dim=-1)  # [batch, seq]
            
            # Build confluence from oracle predictions
            for b in range(probe_tokens.shape[0]):
                for t in range(probe_tokens.shape[1] - 1):
                    # Store with different context lengths for flexible lookup
                    next_token = oracle_preds[b, t].item()
                    
                    for ctx_len in [5, 4, 3, 2]:
                        if t + 1 >= ctx_len:
                            context = tuple(probe_tokens[b, t+1-ctx_len:t+1].cpu().tolist())
                            if context not in self.model.token_confluence:
                                self.model.token_confluence[context] = {}
                            self.model.token_confluence[context][next_token] = \
                                self.model.token_confluence[context].get(next_token, 0) + 1
                        
    def bootstrap_from_oracles(self, num_probes: int = 100):
        """Bootstrap model structure from oracles"""
        print("\n" + "="*60)
        print("ORACLE DISTILLATION (NO BACKPROP)")
        print("="*60)
        
        # Probe types: prioritize vocabulary-based probes
        probe_types = [
            ('real_text', lambda: self.prober.real_text_tokens()),
            ('common_english', lambda: self.prober.common_english_tokens()),
            ('shared_vocab', lambda: self.prober.shared_vocabulary_tokens()),
            ('gpt2_vocab', lambda: self.prober.vocabulary_tokens('gpt2')),
            ('pythia_vocab', lambda: self.prober.vocabulary_tokens('pythia')),
            ('common', lambda: self.prober.common_tokens()),
        ]
        
        for probe_idx in range(num_probes):
            # Cycle through probe types - vocabulary-based first
            probe_type, probe_fn = probe_types[probe_idx % len(probe_types)]
            tokens = probe_fn()
                
            # Learn from each oracle
            for oracle_name in self.oracles:
                # Learn attention patterns
                self.learn_from_oracle_attention(oracle_name, tokens)
                
                # Learn token transitions
                self.learn_from_oracle_predictions(oracle_name, tokens)
                
            # Check if we should grow
            if (probe_idx + 1) % 20 == 0:
                # Compute oracle matching loss
                student_hidden, student_logits = self.model.forward(tokens)
                
                losses = []
                for oracle_name in self.oracles:
                    oracle_logits = self.get_oracle_predictions(oracle_name, tokens)
                    if oracle_logits is not None:
                        loss = self.compute_oracle_loss(student_logits, oracle_logits)
                        losses.append(loss)
                        
                avg_loss = np.mean(losses) if losses else 0
                
                # Grow if loss is high and we can grow
                if avg_loss > 2.0 and self.model.current_layers < self.model.max_layers:
                    self.model.grow_layer()
                    print(f"  🌱 Grew to {self.model.current_layers} layers (oracle loss: {avg_loss:.2f})")
                    
            if (probe_idx + 1) % 25 == 0:
                mat = sum(1 for l in self.model.layers[:self.model.current_layers] if l.materialized)
                print(f"  Probe {probe_idx + 1}/{num_probes}: {mat} layers materialized, "
                      f"{len(self.model.token_confluence)} confluence contexts")
                      
        return {
            'layers': self.model.current_layers,
            'materialized': sum(1 for l in self.model.layers[:self.model.current_layers] if l.materialized),
            'confluence_contexts': len(self.model.token_confluence)
        }
        
    def generate(self, prompt: str, max_tokens: int = 30, verbose: bool = False) -> str:
        """Generate using confluence learned from oracles"""
        tokenizer = self.oracles['gpt2']['tokenizer'] if 'gpt2' in self.oracles else None
        if tokenizer is None:
            return prompt
            
        tokens = tokenizer.encode(prompt)
        
        # Track hit/miss for analysis
        hits = 0
        misses = 0
        
        for _ in range(max_tokens):
            # Try confluence first (learned from oracles)
            found = False
            for ctx_len in [5, 4, 3, 2]:
                if len(tokens) >= ctx_len:
                    context = tuple(tokens[-ctx_len:])
                    
                    if context in self.model.token_confluence:
                        candidates = self.model.token_confluence[context]
                        if candidates:
                            # Sample proportionally
                            total = sum(candidates.values())
                            items = list(candidates.items())
                            weights = [v/total for _, v in items]
                            
                            r = np.random.random()
                            cumsum = 0
                            next_token = items[0][0]
                            for (tok, _), w in zip(items, weights):
                                cumsum += w
                                if r < cumsum:
                                    next_token = tok
                                    break
                                    
                            tokens.append(next_token)
                            found = True
                            hits += 1
                            break
            
            if not found:
                misses += 1
                # Fall back to oracle (we haven't learned this yet)
                # This shows what we SHOULD have learned
                input_ids = torch.tensor([tokens[-32:]], device=self.device)
                
                if 'gpt2' in self.oracles:
                    oracle_logits = self.get_oracle_predictions('gpt2', input_ids)
                    if oracle_logits is not None:
                        next_token = oracle_logits[0, -1].argmax().item()
                        tokens.append(next_token)
                        
                        # Learn this on the fly!
                        for ctx_len in [5, 4, 3, 2]:
                            if len(tokens) > ctx_len:
                                context = tuple(tokens[-(ctx_len+1):-1])
                                if context not in self.model.token_confluence:
                                    self.model.token_confluence[context] = {}
                                self.model.token_confluence[context][next_token] = \
                                    self.model.token_confluence[context].get(next_token, 0) + 1
                        continue
                
                # Last resort: student model
                _, logits = self.model.forward(input_ids)
                next_token = logits[0, -1].argmax().item()
                tokens.append(next_token)
                
            # Stop on period
            if tokenizer.decode([tokens[-1]]).strip() == '.':
                break
        
        result = tokenizer.decode(tokens)
        
        if verbose:
            total = hits + misses
            hit_rate = hits / total * 100 if total > 0 else 0
            print(f"    [Confluence: {hits}/{total} = {hit_rate:.1f}% hit rate]")
                
        return result


def main():
    print("="*60)
    print("Oracle-Bootstrapped Hierarchical PAC Training")
    print("NO BACKPROP - Oracle as Loss Function")
    print("="*60)
    
    trainer = OracleDistillationTrainer()
    
    # Bootstrap from oracles
    summary = trainer.bootstrap_from_oracles(num_probes=100)
    
    print("\n" + "="*60)
    print("BOOTSTRAP COMPLETE")
    print("="*60)
    
    print(f"\nModel Structure:")
    print(f"  Layers: {summary['layers']}")
    print(f"  Materialized: {summary['materialized']}")
    print(f"  Confluence contexts: {summary['confluence_contexts']}")
    
    # Fibonacci check
    print(f"\nFibonacci Check:")
    for fib in FIBONACCI:
        if summary['layers'] == fib:
            print(f"  ✓ {summary['layers']} layers = F({FIBONACCI.index(fib)})")
            break
    else:
        print(f"  ○ {summary['layers']} layers not Fibonacci-aligned")
        
    # Layer status
    print(f"\nLayer Status:")
    for i in range(trainer.model.current_layers):
        layer = trainer.model.layers[i]
        status = "✓ materialized" if layer.materialized else "○ lazy"
        acts = layer.activations
        print(f"  Layer {i}: {status}, {acts} activations")
        
    # Generation test
    print("\n" + "="*60)
    print("GENERATION TEST (from oracle-learned knowledge)")
    print("="*60)
    
    prompts = [
        "The cat",
        "Scientists study",
        "Language is",
        "In the future"
    ]
    
    for prompt in prompts:
        print(f"\n'{prompt}' →")
        generated = trainer.generate(prompt, max_tokens=25, verbose=True)
        print(f"    {generated}")
        
    # Save results
    results = {
        'layers': summary['layers'],
        'materialized': summary['materialized'],
        'confluence_contexts': summary['confluence_contexts'],
        'oracles_used': list(trainer.oracles.keys())
    }
    
    output_path = Path(__file__).parent.parent / "results" / "oracle_distillation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
