"""
Baseline Stochastic Generation Module

Implements standard random generation with mutation for baseline comparison.
"""

import random
import itertools
from typing import List, Tuple, Dict, Any
from collections import Counter
import numpy as np
import zlib
import math

class BaselineGenerator:
    """Baseline stochastic text generation with mutation"""
    
    def __init__(self, 
                 base_alphabet: List[str],
                 mutation_alphabet: List[str],
                 input_length: int,
                 output_length_range: Tuple[int, int],
                 mutation_rate: float = 0.2):
        
        self.base_alphabet = base_alphabet
        self.mutation_alphabet = mutation_alphabet
        self.full_alphabet = base_alphabet + mutation_alphabet
        self.input_length = input_length
        self.output_min_len, self.output_max_len = output_length_range
        self.mutation_rate = mutation_rate
        
        # Generate input space
        self.input_space = list(itertools.product(base_alphabet, repeat=input_length))
        self.input_strings = ["".join(seq) for seq in self.input_space]
    
    def generate_output(self, seed: str = None) -> str:
        """Generate output with mutations and variable length"""
        length = random.randint(self.output_min_len, self.output_max_len)
        
        # Option to use seed for tracing transformations
        if seed and len(seed) <= length:
            seq = list(seed) + [random.choice(self.base_alphabet) 
                               for _ in range(length - len(seed))]
        else:
            seq = []
            
        for i in range(len(seq), length):
            if random.random() < self.mutation_rate:
                seq.append(random.choice(self.mutation_alphabet))
            else:
                seq.append(random.choice(self.base_alphabet))
                
        # Additional transformation: local rearrangements
        if random.random() < 0.1:  # 10% chance of local swap
            if len(seq) > 1:
                i = random.randint(0, len(seq) - 2)
                seq[i], seq[i+1] = seq[i+1], seq[i]
                
        return "".join(seq)
    
    def generate_batch(self, num_outputs: int) -> List[str]:
        """Generate batch of outputs"""
        return [self.generate_output() for _ in range(num_outputs)]
    
    def analyze_outputs(self, outputs: List[str]) -> Dict[str, Any]:
        """Analyze generated outputs"""
        unique_outputs = list(set(outputs))
        
        # Basic metrics
        input_space_size = len(self.input_space)
        output_space_observed = len(unique_outputs)
        combinatorial_amplification = output_space_observed / input_space_size
        
        # Complexity analysis
        input_concat = "".join(self.input_strings)
        output_concat = "".join(outputs)
        
        input_kolmogorov = len(zlib.compress(input_concat.encode('utf-8'), level=9))
        output_kolmogorov = len(zlib.compress(output_concat.encode('utf-8'), level=9))
        complexity_amplification = output_kolmogorov / max(input_kolmogorov, 1)
        
        # Entropy analysis
        input_entropy = self._calculate_shannon_entropy(input_concat)
        output_entropy = self._calculate_shannon_entropy(output_concat)
        entropy_production = output_entropy - input_entropy
        
        # Structural novelty
        input_trigrams = set()
        for inp in self.input_strings:
            input_trigrams.update(self._extract_ngrams(inp, 3))
            
        output_trigrams = set()
        for out in outputs:
            output_trigrams.update(self._extract_ngrams(out, 3))
            
        novel_trigrams = output_trigrams - input_trigrams
        
        # Hierarchical emergence
        hierarchical_emergence = {}
        for n in range(2, 8):
            all_ngrams = set()
            for text in outputs:
                all_ngrams.update(self._extract_ngrams(text, n))
            hierarchical_emergence[f"{n}-gram"] = len(all_ngrams)
        
        return {
            'method': 'baseline_stochastic',
            'combinatorial_amplification': combinatorial_amplification,
            'complexity_amplification': complexity_amplification,
            'entropy_production': entropy_production,
            'novel_structures': len(novel_trigrams),
            'hierarchical_emergence': hierarchical_emergence,
            'unique_outputs': len(unique_outputs),
            'total_outputs': len(outputs)
        }
    
    def _calculate_shannon_entropy(self, data: str) -> float:
        """Calculate Shannon entropy"""
        if not data:
            return 0.0
        counts = Counter(data)
        total = len(data)
        probs = [count/total for count in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)
    
    def _extract_ngrams(self, text: str, n: int) -> set:
        """Extract n-grams from text"""
        if len(text) < n:
            return set()
        return {text[i:i+n] for i in range(len(text) - n + 1)}
