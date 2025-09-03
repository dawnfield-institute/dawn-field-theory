"""
GAIA Language Learning System
============================

Clean implementation using GAIA's entropy field dynamics for language learning.
"""

import torch
import sys
import json
import random
import os
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Setup clean logging - debug to files, conversation to console
logging.basicConfig(
    level=logging.ERROR,  # Only show errors on console
    format='%(message)s'  # Clean console format
)

# File logger for all debug information
debug_logger = logging.getLogger('debug')
debug_handler = logging.FileHandler('gaia_debug.log', mode='w')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
debug_logger.addHandler(debug_handler)
debug_logger.setLevel(logging.DEBUG)
debug_logger.propagate = False

# Conversation logger for Ollama interactions (console + file)
conversation_logger = logging.getLogger('conversation')
conversation_handler = logging.StreamHandler()
conversation_handler.setLevel(logging.INFO)
conversation_handler.setFormatter(logging.Formatter('%(message)s'))
conversation_logger.addHandler(conversation_handler)
conversation_file_handler = logging.FileHandler('gaia_conversation.log', mode='w', encoding='utf-8')
conversation_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
conversation_logger.addHandler(conversation_file_handler)
conversation_logger.setLevel(logging.INFO)
conversation_logger.propagate = False

# System logger for important system messages (console + file)
system_logger = logging.getLogger('system')
system_handler = logging.StreamHandler()
system_handler.setLevel(logging.INFO)
system_handler.setFormatter(logging.Formatter('%(message)s'))
system_logger.addHandler(system_handler)
system_file_handler = logging.FileHandler('gaia_system.log', mode='w', encoding='utf-8')
system_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
system_logger.addHandler(system_file_handler)
system_logger.setLevel(logging.INFO)
system_logger.propagate = False

# Careful import to avoid circular dependencies
sys.path.append('..')
from gaia_runtime import GAIARuntime, GAIAConfig

# Import GAIA adaptive controller for auto parameter tuning
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.core.adaptive_controller import GAIAAdaptiveController

print("GAIA Language Learning System")


# Thermodynamic Memory Management Functions
def landauer_energy_cost(entropy: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Calculate the energy cost of erasing information based on Landauer's Principle.
    
    Args:
        entropy: Information entropy of the pattern
        temperature: System temperature for thermodynamic calculations
    
    Returns:
        Energy cost in normalized units
    """
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    # Normalize for neural network scale
    return k_B * temperature * entropy / 1e-20  # Scale to reasonable range


def compute_pattern_entropy(pattern: torch.Tensor) -> torch.Tensor:
    """Compute information entropy of a pattern tensor with safe handling"""
    try:
        # Flatten and normalize to probability distribution
        flat_pattern = pattern.flatten()
        
        # Ensure we have valid values
        if torch.isnan(flat_pattern).any() or torch.isinf(flat_pattern).any():
            return torch.tensor(1.0)  # Default entropy
        
        # Shift to positive values if needed
        min_val = torch.min(flat_pattern)
        if min_val < 0:
            flat_pattern = flat_pattern - min_val + 1e-8
        
        # Normalize to probability distribution
        total = torch.sum(flat_pattern)
        if total <= 0:
            return torch.tensor(1.0)  # Default entropy
        
        probabilities = flat_pattern / total + 1e-8
        
        # Calculate Shannon entropy safely
        log_probs = torch.log2(probabilities)
        entropy = -torch.sum(probabilities * log_probs)
        
        # Ensure finite result
        if torch.isnan(entropy) or torch.isinf(entropy):
            return torch.tensor(1.0)
        
        return entropy
        
    except Exception as e:
        print(f"    ENTROPY: Error computing entropy: {e}")
        return torch.tensor(1.0)  # Safe fallback


def thermodynamic_pattern_score(pattern: torch.Tensor, usage_count: int, 
                               temperature: float, recency: float) -> float:
    """
    Score a fractal pattern for pruning/growth decisions using thermodynamic principles.
    
    Args:
        pattern: The fractal geometric pattern
        usage_count: How many times this pattern has been used
        temperature: Current system temperature
        recency: How recently the pattern was used (0-1, 1 = most recent)
    
    Returns:
        Score for keeping the pattern (higher = more valuable)
    """
    # Calculate entropy of the pattern
    pattern_entropy = compute_pattern_entropy(pattern)
    
    # Calculate Landauer energy cost of erasing this pattern
    erasure_cost = landauer_energy_cost(pattern_entropy, temperature)
    
    # Usage frequency contributes to retention value
    usage_value = torch.log1p(torch.tensor(float(usage_count)))
    
    # Recency decay (recent patterns are more valuable)
    recency_value = torch.exp(-2.0 * (1.0 - torch.tensor(recency)))
    
    # Information content value (more structured patterns are more valuable)
    info_value = pattern_entropy / torch.log2(torch.tensor(float(pattern.numel())))
    
    # Combine all factors with thermodynamic weighting
    total_score = (
        0.4 * usage_value +
        0.3 * recency_value + 
        0.2 * info_value +
        0.1 * torch.log1p(erasure_cost)  # Higher erasure cost = more valuable to keep
    )
    
    return total_score.item()


class OllamaTeacher:
    """
    Ollama integration for dynamic conversational learning.
    Replaces static dataset with real-time conversation generation.
    """
    
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.conversation_history = []
        self.teaching_phase = True
        self.assessment_phase = False
        
        # Test connection
        if self._test_connection():
            debug_logger.info(f"Connected to Ollama model: {model_name}")
        else:
            debug_logger.error(f"❌ Failed to connect to Ollama at {base_url}")
            raise ConnectionError("Cannot connect to Ollama")
    
    def _test_connection(self) -> bool:
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            debug_logger.error(f"Ollama connection test failed: {e}")
            return False
    
    def _send_prompt(self, prompt: str) -> str:
        """Send a prompt to Ollama and get response"""
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                debug_logger.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            debug_logger.error(f"Error communicating with Ollama: {e}")
            return ""
    
    def get_teaching_sentence(self, learning_progress: dict) -> str:
        """
        Phase 1: Generate teaching sentences based on GAIA's learning progress
        """
        if not self.teaching_phase:
            return ""
        
        # Create adaptive curriculum based on progress
        current_vocab_size = learning_progress.get('vocab_size', 0)
        current_patterns = learning_progress.get('pattern_count', 0)
        coherence_score = learning_progress.get('coherence', 0.0)
        
        if current_vocab_size < 20:
            # Basic vocabulary building
            prompt = f"""You are teaching a neural language model its first words. 
Generate ONE simple sentence using basic English words like: I, am, is, the, a, time, where, what, hello, learning.
Keep it under 8 words. Make it slightly different from previous attempts.
Previous vocabulary size: {current_vocab_size}
Sentence:"""
        
        elif current_vocab_size < 50:
            # Expanding vocabulary
            prompt = f"""You are teaching a neural language model expanding vocabulary.
Generate ONE sentence using common English words, focusing on verbs, nouns, and simple concepts.
Keep it under 12 words. Include some new words the model hasn't seen much.
Current vocabulary: {current_vocab_size} words
Sentence:"""
        
        else:
            # Advanced concepts and structure
            prompt = f"""You are teaching a neural language model advanced language structure.
Generate ONE sentence with more complex grammar, but still clear and educational.
Include concepts like causality, time, relationships, or abstract thinking.
Current progress: {current_vocab_size} words, {current_patterns} patterns, coherence: {coherence_score:.2f}
Sentence:"""
        
        response = self._send_prompt(prompt)
        
        # Clean up response to extract just the sentence
        if response:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.lower().startswith(('generate', 'you are', 'current', 'previous', 'sentence:')):
                    # Remove common prefixes
                    for prefix in ['Sentence:', 'Example:', 'Here:', 'Teaching:']:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                    if line:
                        debug_logger.info(f"Ollama teaching: '{line}'")
                        return line
        
        # Fallback if parsing fails
        return "I am learning to understand language."
    
    def assess_learning(self, gaia_output: str, learning_stats: dict) -> dict:
        """
        Phase 2: Assess if GAIA is ready for conversation
        Returns assessment with readiness decision
        """
        if not self.assessment_phase:
            return {"ready": False, "continue_teaching": True}
        
        prompt = f"""You are assessing a neural language model's English learning progress.

GAIA's Recent Output: "{gaia_output}"
Learning Statistics:
- Vocabulary size: {learning_stats.get('vocab_size', 0)}
- Pattern count: {learning_stats.get('pattern_count', 0)}
- Coherence score: {learning_stats.get('coherence', 0.0)}
- Learning iterations: {learning_stats.get('iterations', 0)}

Assess if GAIA is ready for basic conversation. Consider:
1. Can it form coherent word sequences?
2. Does it understand basic language structure?
3. Can it generate meaningful responses?

Respond with ONLY:
READY: YES/NO
REASONING: [brief explanation]"""

        response = self._send_prompt(prompt)
        
        # Parse assessment
        ready = False
        reasoning = "Assessment failed"
        
        if response:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.upper().startswith('READY:'):
                    ready = 'YES' in line.upper()
                elif line.upper().startswith('REASONING:'):
                    reasoning = line[len('REASONING:'):].strip()
        
        debug_logger.info(f"Ollama assessment - Ready: {ready}, Reasoning: {reasoning}")
        
        return {
            "ready": ready,
            "reasoning": reasoning,
            "continue_teaching": not ready
        }
    
    def start_conversation_phase(self):
        """Switch to conversation/assessment phase"""
        self.teaching_phase = False
        self.assessment_phase = True
        debug_logger.info("Switching to conversation/assessment phase")
    
    def generate_conversation_prompt(self, context: str = "") -> str:
        """Generate a conversation prompt for GAIA to respond to"""
        prompt = f"""Generate a simple question or statement that a language learning AI could respond to.
Keep it basic and encourage the AI to demonstrate its English learning.
Context: {context}
Make it conversational and friendly.

Question:"""
        
        response = self._send_prompt(prompt)
        if response:
            # Clean up response
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.lower().startswith(('generate', 'question:', 'context')):
                    if line.startswith('Question:'):
                        line = line[len('Question:'):].strip()
                    if line:
                        return line
        
        return "Hello! How are you learning English today?"


class GAIALanguageLearner:
    def __init__(self, use_ollama: bool = True):
        system_logger.info("=== Initializing GAIA Language Learner ===")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Match GAIA runtime device
        debug_logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        debug_logger.info(f"Output directory: {self.output_dir}")
        
        # Initialize Ollama teacher for dynamic learning
        self.use_ollama = use_ollama
        self.ollama_teacher = None
        if use_ollama:
            try:
                self.ollama_teacher = OllamaTeacher()
                system_logger.info("Ollama teacher initialized - Dynamic conversational learning enabled")
            except Exception as e:
                system_logger.warning(f"Failed to initialize Ollama teacher: {e}")
                system_logger.info("📚 Falling back to static dataset learning")
                self.use_ollama = False
        
        # Learning state
        self.vocabulary = {}
        self.word_list = []
        self.patterns = {}
        self.stats = {'sentences': 0, 'tokens': 0, 'patterns': 0}
        self.learning_log = []
        self.generation_results = []
        
        # Initialize GAIA Adaptive Controller for auto parameter tuning
        try:
            self.adaptive_controller = GAIAAdaptiveController(
                base_collapse_threshold=0.0003,
                adaptation_window=20,
                balance_momentum=0.8,
                qbe_sensitivity=0.1
            )
            debug_logger.info("GAIA Adaptive Controller initialized")
        except:
            debug_logger.warning("! Adaptive Controller import failed, using manual parameters")
            self.adaptive_controller = None
        
        # Enhanced QBE regulation parameters
        self.qbe_lambda = 0.3  # QBE regulation strength
        self.qbe_target_ratio = 1.0  # Target E/I balance ratio
        self.qbe_stability_threshold = 0.1  # Stability tolerance
        self.pressure_explosion_limit = 1000.0  # Prevent infinite pressure
        
        # Advanced thermodynamic memory parameters
        self.thermodynamic_temperature = 300.0  # Room temperature (K)
        self.landauer_constant = 1.38e-23  # Boltzmann constant for Landauer's Principle
        self.memory_efficiency_target = 0.85  # Target memory efficiency
        self.fractal_reuse_bonus = 0.2  # Bonus for reusing crystallized patterns
        
        # Initialize with smaller field and disabled geometric guidance for easier crystallization
        config = GAIAConfig(
            field_shape=(32, 32),  # Smaller field = more concentrated pressure
            adaptive_tuning=True,
            geometric_guidance=False,  # Disable for easier crystallization
            scbf_enabled=False,  # Disable SCBF for simpler testing
            device="cpu"  # Force CPU usage
        )
        self.gaia = GAIARuntime(config)
        debug_logger.info("GAIA Runtime ready")
        
        # Simple encoders
        self.vocab_size = 500
        self.emb_dim = 32
        self.embeddings = torch.randn(self.vocab_size, self.emb_dim, device=self.device) * 0.1
        self.to_field = torch.randn(self.emb_dim, 64*64, device=self.device) * 0.1
        
        # Initialize accumulated memory for field operations
        self.accumulated_memory = torch.zeros(32, 32, device=self.device)
        
        debug_logger.info("Language Learner initialized")
    
    def load_dataset(self, path: str) -> Dict:
        print(f"\n=== Loading Dataset: {path} ===")
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            sentences = data.get('sentences', [])
            print(f"✓ Loaded {len(sentences)} sentences")
            return data
        except Exception as e:
            print(f"✗ Loading failed: {e}")
            return {}
    
    def build_vocabulary(self, sentences: List[Dict]):
        debug_logger.info("=== Building Vocabulary ===")
        
        # Count words
        word_counts = {}
        for sent in sentences:
            for token in sent.get('tokens', []):
                word_counts[token] = word_counts.get(token, 0) + 1
        
        # Take most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, (word, count) in enumerate(sorted_words[:self.vocab_size]):
            self.vocabulary[word] = i
            self.word_list.append(word)
        
        print(f"✓ Vocabulary: {len(self.vocabulary)} words")
        print(f"Sample: {self.word_list[:10]}")
    
    def recursive_balance_field(self, energy_field: torch.Tensor, info_field: torch.Tensor, memory_field: torch.Tensor) -> torch.Tensor:
        """Compute recursive balance field for dynamic scaling with detailed logging"""
        try:
            # Enhanced Balance potential: B(x,t) = λ * [(E-I)/(1+αM)] * Φ(x) with stability controls
            
            # Energy-Information differential with gradient smoothing
            ei_diff = energy_field - info_field
            
            # NaN protection for E-I differential
            if torch.isnan(ei_diff).any() or torch.isinf(ei_diff).any():
                # Instead of zeros, use small random values to maintain dynamics
                ei_diff = torch.randn_like(ei_diff) * 0.001
            
            # Apply spatial smoothing to reduce sharp gradients
            kernel = torch.ones(3, 3, device=self.device) / 9.0
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            ei_diff_smooth = torch.nn.functional.conv2d(
                ei_diff.unsqueeze(0).unsqueeze(0), 
                kernel, 
                padding=1
            ).squeeze()
            
            # NaN protection for smoothed differential
            if torch.isnan(ei_diff_smooth).any() or torch.isinf(ei_diff_smooth).any():
                ei_diff_smooth = torch.zeros_like(ei_diff_smooth)
            
            print(f"    RBF: E-I differential stats - mean={torch.mean(ei_diff_smooth).item():.6f}, std={torch.std(ei_diff_smooth).item():.6f}")
            
            # Enhanced memory modulation with adaptive damping
            base_alpha = 0.05  # Reduced base damping for more stability
            field_magnitude = torch.mean(torch.abs(energy_field) + torch.abs(info_field))
            adaptive_alpha = base_alpha * (1.0 + torch.tanh(field_magnitude / 10.0))  # Increase damping for large fields
            
            # NaN protection for adaptive_alpha
            if torch.isnan(adaptive_alpha) or torch.isinf(adaptive_alpha):
                adaptive_alpha = base_alpha
            
            memory_modulation = 1.0 + adaptive_alpha * torch.clamp(memory_field, 0, 8)  # Tighter memory clamp
            memory_modulation = torch.clamp(memory_modulation, 0.2, 15.0)  # More conservative bounds
            
            # Check for NaN in memory_modulation
            if torch.isnan(memory_modulation).any() or torch.isinf(memory_modulation).any():
                memory_modulation = torch.ones_like(memory_modulation)
                
            print(f"    RBF: Memory modulation stats - mean={torch.mean(memory_modulation).item():.6f}, min={torch.min(memory_modulation).item():.6f}")
            
            # Enhanced harmonic modulation with multiple frequencies for better stability
            x = torch.arange(32, device=self.device).float()
            y = torch.arange(32, device=self.device).float()
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Multi-frequency harmonic pattern for smoother fields
            phi1 = torch.sin(X/6) * torch.cos(Y/6)  # Primary frequency
            phi2 = 0.3 * torch.sin(X/12 + Y/12)     # Lower frequency component
            phi3 = 0.1 * torch.sin(X/3) * torch.cos(Y/4)  # Higher frequency detail
            phi = phi1 + phi2 + phi3 + 0.1  # Small offset to prevent zeros
            
            # Enhanced balance field computation with adaptive lambda
            base_lambda = 0.3  # Reduced base strength for stability
            field_variance = torch.var(ei_diff_smooth)
            adaptive_lambda = base_lambda * torch.exp(-field_variance / 5.0)  # Reduce lambda for high variance
            
            # NaN protection for adaptive_lambda
            if torch.isnan(adaptive_lambda) or torch.isinf(adaptive_lambda):
                adaptive_lambda = base_lambda
            
            balance_field = adaptive_lambda * (ei_diff_smooth / memory_modulation) * phi
            
            # Check for NaN in balance_field before clamping
            if torch.isnan(balance_field).any() or torch.isinf(balance_field).any():
                balance_field = torch.zeros_like(balance_field)
            
            # Progressive clamping based on field statistics
            field_std = torch.std(balance_field)
            if torch.isnan(field_std) or torch.isinf(field_std) or field_std == 0:
                clamp_range = 2.0
            else:
                clamp_range = max(2.0, min(8.0, 3.0 * field_std))  # Adaptive clamping
            balance_field = torch.clamp(balance_field, -clamp_range, clamp_range)
            
            print(f"    RBF: Balance field stats - mean={torch.mean(balance_field).item():.6f}, std={torch.std(balance_field).item():.6f}")
            print(f"    RBF: Adaptive params - alpha={adaptive_alpha:.6f}, lambda={adaptive_lambda:.6f}, clamp_range={clamp_range:.2f}")
            
            return balance_field
            
        except Exception as e:
            print(f"    RBF ERROR: {e}")
            # Return safe fallback
            return torch.zeros_like(energy_field)
    
    def adaptive_scaling(self, field: torch.Tensor, balance_field: torch.Tensor, memory_field: torch.Tensor) -> torch.Tensor:
        """Apply optimized adaptive scaling based on quantum balance equations"""
        try:
            print(f"    SCALING: Input field stats - mean={torch.mean(field).item():.6f}, std={torch.std(field).item():.6f}")
            
            # Optimized gradient computation using built-in operations
            with torch.no_grad():  # Disable gradients for efficiency
                # Use Sobel-like operators for faster gradient computation
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device, dtype=torch.float32)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device, dtype=torch.float32)
                
                # Reshape for conv2d
                sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
                sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)
                
                # Compute gradients efficiently
                field_unsq = field.unsqueeze(0).unsqueeze(0)
                grad_x = torch.nn.functional.conv2d(field_unsq, sobel_x, padding=1).squeeze()
                grad_y = torch.nn.functional.conv2d(field_unsq, sobel_y, padding=1).squeeze()
                
                gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
                print(f"    SCALING: Gradient magnitude - mean={torch.mean(gradient_magnitude).item():.6f}")
                
                # Optimized balance stability computation
                balance_unsq = balance_field.unsqueeze(0).unsqueeze(0)
                balance_grad_x = torch.nn.functional.conv2d(balance_unsq, sobel_x, padding=1).squeeze()
                balance_grad_y = torch.nn.functional.conv2d(balance_unsq, sobel_y, padding=1).squeeze()
                
                balance_gradient_mag = torch.sqrt(balance_grad_x**2 + balance_grad_y**2 + 1e-8)
                balance_stability = torch.exp(-torch.clamp(balance_gradient_mag, 0, 8))  # Reduced clamp range
                print(f"    SCALING: Balance stability - mean={torch.mean(balance_stability).item():.6f}")
                
                # Optimized scaling computation with pre-computed constants
                base_scale = 1.8  # Slightly reduced for more stability
                
                # Use fast approximations for exponentials where possible
                stability_modulation = 1.0 + 1.5 * balance_stability  # Reduced amplification
                
                # Optimized instability dampening with tanh approximation
                normalized_grad = torch.clamp(gradient_magnitude / 2.5, 0, 4)
                instability_dampening = 1.0 / (1.0 + normalized_grad**2)  # Faster than exp
                
                # Vectorized adaptive scaling
                adaptive_scale = base_scale * stability_modulation * instability_dampening
                adaptive_scale = torch.clamp(adaptive_scale, 0.2, 8.0)  # Tighter bounds for stability
                print(f"    SCALING: Adaptive scale - mean={torch.mean(adaptive_scale).item():.6f}")
            
                print(f"    SCALING: Adaptive scale - mean={torch.mean(adaptive_scale).item():.6f}")
                
                # Optimized memory-based scaling with efficient computation
                memory_clamped = torch.clamp(memory_field, 0, 4)  # Tighter memory bounds
                # Use polynomial approximation instead of division for speed
                memory_scale = 1.0 + 0.4 * memory_clamped * (1.0 - 0.15 * memory_clamped)  # Quadratic falloff
                print(f"    SCALING: Memory scale - mean={torch.mean(memory_scale).item():.6f}")
                
                # Optimized final scaling with in-place operations where possible
                result = field * adaptive_scale * memory_scale
                
                # Apply final stability check with efficient operations
                result_mean = torch.mean(torch.abs(result))
                if result_mean > 100:  # Detect potential instability
                    stabilization_factor = 50.0 / result_mean
                    result = result * stabilization_factor
                    print(f"    SCALING: Applied stabilization factor {stabilization_factor:.4f}")
                
                print(f"    SCALING: Output field stats - mean={torch.mean(result).item():.6f}, std={torch.std(result).item():.6f}")
                return result
                
        except Exception as e:
            print(f"    SCALING ERROR: {e}")
            # Return safe fallback
            return field

    def encode_to_field(self, tokens: List[str]) -> torch.Tensor:
        """Convert tokens to GAIA field with recursive balance field dynamics"""
        if not tokens:
            return torch.zeros(32, 32, device=self.device)
        
        # Create base semantic field with small random initialization to prevent zeros
        energy_field = torch.randn(32, 32, device=self.device) * 0.001
        info_field = torch.randn(32, 32, device=self.device) * 0.001
        
        # Initialize neuromorphic fractal memory system
        if not hasattr(self, 'fractal_memory_bank'):
            self.fractal_memory_bank = {}  # Store crystallized geometric patterns
            self.memory_usage_stats = {}   # Track pattern usage frequency
            self.memory_bank_size = 50     # Maximum number of stored patterns
            
            # Thermodynamic parameters for pruning/growth
            self.system_temperature = 300.0  # Base temperature in Kelvin
            self.entropy_history = []        # Track system entropy over time
            self.growth_threshold = 0.7      # When to grow new patterns
            self.prune_threshold = 0.3       # When to prune old patterns
            
        # Initialize accumulated memory field (accumulated from previous learning)
        if not hasattr(self, 'accumulated_memory'):
            self.accumulated_memory = torch.zeros(32, 32, device=self.device)
        
        for i, token in enumerate(tokens):
            # Dynamically add unknown words to vocabulary
            if token not in self.vocabulary:
                if len(self.vocabulary) < self.vocab_size:
                    word_id = len(self.vocabulary)
                    self.vocabulary[token] = word_id
                    self.word_list.append(token)
                    print(f"  Added '{token}' to vocabulary (ID: {word_id})")
                else:
                    # Use a hash-based fallback for unknown words
                    word_id = hash(token) % self.vocab_size
            else:
                word_id = self.vocabulary[token]
                
            # Map word semantics to field positions
            word_hash = hash(token) % (32 * 32)
            row, col = word_hash // 32, word_hash % 32
            
            # Add word embedding influence to energy field (optimized vectorized version)
            if word_id < self.vocab_size:
                    emb = self.embeddings[word_id]
                    
                    # Optimized spatial influence computation using meshgrid
                    dr_range = torch.arange(-2, 3, device=self.device)
                    dc_range = torch.arange(-2, 3, device=self.device)
                    DR, DC = torch.meshgrid(dr_range, dc_range, indexing='ij')
                    
                    # Vectorized distance computation
                    distances = torch.sqrt(DR.float()**2 + DC.float()**2)
                    influence_mask = distances < 3
                    
                    # Compute all positions at once
                    r_positions = (row + DR) % 32
                    c_positions = (col + DC) % 32
                    
                    # Vectorized influence computation
                    emb_indices = torch.clamp(DR + 2, 0, self.emb_dim - 1)
                    influences = emb[emb_indices] * torch.exp(-distances / 1.5)
                    influences = influences * (1.0 + i * 0.1) * influence_mask
                    
                    # Add influences to energy field efficiently
                    for dr_idx in range(5):
                        for dc_idx in range(5):
                            if influence_mask[dr_idx, dc_idx]:
                                r, c = r_positions[dr_idx, dc_idx], c_positions[dr_idx, dc_idx]
                                energy_field[r, c] += influences[dr_idx, dc_idx].item()
                                
                                # Info field computation
                                info_influence = (1.0 / (1.0 + distances[dr_idx, dc_idx])) * (i + 1) / len(tokens)
                                info_field[r, c] += info_influence.item()
        
        # Apply semantic gradients for linguistic structure
        if len(tokens) > 1:
            grad_x = torch.gradient(energy_field, dim=0)[0] * 0.2
            grad_y = torch.gradient(energy_field, dim=1)[0] * 0.2
            energy_field = energy_field + grad_x + grad_y
            
            info_grad_x = torch.gradient(info_field, dim=0)[0] * 0.3
            info_grad_y = torch.gradient(info_field, dim=1)[0] * 0.3
            info_field = info_field + info_grad_x + info_grad_y
        
        # Compute recursive balance field
        balance_field = self.recursive_balance_field(energy_field, info_field, self.accumulated_memory)
        
        # Start with energy field as base
        field = energy_field.clone()
        
        # Apply adaptive scaling instead of uniform scaling
        field = self.adaptive_scaling(field, balance_field, self.accumulated_memory)
        
        # Update accumulated memory using QBE-balanced fractal crystallization
        field_pressure = self.calculate_field_pressure(field, balance_field)
        crystallized_geometry = self.crystallize_to_fractal_geometry(field, balance_field)
        
        # Quantum Balance Equation: dE/dt + dI/dt = λ·QPL(t)
        # Energy component: Field pressure drives crystallization
        # Information component: Fractal patterns encode structured information
        energy_rate = field_pressure  # dE/dt
        info_rate = self.calculate_information_rate(crystallized_geometry)  # dI/dt
        qpl_regulation = self.compute_qpl_regulation(energy_rate, info_rate)
        
        # Balance energy and information according to QBE
        balanced_energy, balanced_info = self.apply_qbe_balance(
            energy_rate, info_rate, qpl_regulation
        )
        
        # Store balanced fractal pattern
        self.store_fractal_pattern(crystallized_geometry, tokens, 
                                  energy_component=balanced_energy,
                                  info_component=balanced_info)
        
        # Track entropy for thermodynamic control
        self.update_entropy_tracking(field)
        
        # Use QBE-regulated memory reconstruction with NaN protection
        memory_decay = 0.9
        memory_integration = 0.1
        
        # Get memory components with NaN protection
        reconstructed_memory = self.reconstruct_from_fractals()
        if torch.isnan(reconstructed_memory).any() or torch.isinf(reconstructed_memory).any():
            reconstructed_memory = torch.zeros_like(self.accumulated_memory)
        
        new_memory_component = (crystallized_geometry[:32, :32] * balanced_info) * memory_integration
        if torch.isnan(new_memory_component).any() or torch.isinf(new_memory_component).any():
            new_memory_component = torch.zeros_like(self.accumulated_memory)
        
        self.accumulated_memory = reconstructed_memory * memory_decay + new_memory_component
        
        # Final NaN check for accumulated memory
        if torch.isnan(self.accumulated_memory).any() or torch.isinf(self.accumulated_memory).any():
            self.accumulated_memory = torch.zeros_like(self.accumulated_memory)
        
        # Log QBE balance for monitoring with adaptive controller status
        if self.adaptive_controller:
            qbe_status = self.adaptive_controller.get_qbe_status()
            pattern_type = self.adaptive_controller.detect_field_pattern_type()
            thresholds = self.adaptive_controller.get_adaptive_thresholds()
            print(f"    QBE: E_rate={energy_rate:.3f}, I_rate={info_rate:.3f}, QPL={qpl_regulation:.3f}")
            print(f"    Adaptive: {qbe_status}, Pattern={pattern_type}, Threshold={thresholds.collapse_threshold:.6f}")
        else:
            print(f"    QBE: E_rate={energy_rate:.3f}, I_rate={info_rate:.3f}, QPL={qpl_regulation:.3f}")
        
        return field
    
    def calculate_field_pressure(self, field: torch.Tensor, balance_field: torch.Tensor) -> float:
        """Calculate energetic pressure in the field system"""
        try:
            # Pressure from field gradients (energy accumulation)
            field_gradient = torch.gradient(field, dim=[0, 1])
            gradient_magnitude = torch.sqrt(field_gradient[0]**2 + field_gradient[1]**2)
            
            # Balance field contribution to pressure
            balance_gradient = torch.gradient(balance_field, dim=[0, 1])
            balance_magnitude = torch.sqrt(balance_gradient[0]**2 + balance_gradient[1]**2)
            
            # Total pressure as mean gradient energy
            pressure = torch.mean(gradient_magnitude + 0.5 * balance_magnitude)
            
            return pressure.item()
            
        except Exception as e:
            print(f"    QBE: Error calculating pressure: {e}")
            return 0.0
    
    def calculate_information_rate(self, fractal_geometry: torch.Tensor) -> float:
        """Calculate rate of information structuring (dI/dt) from fractal patterns"""
        try:
            # Information content from pattern entropy
            pattern_entropy = compute_pattern_entropy(fractal_geometry)
            
            # Information rate as negative entropy change (negentropy creation)
            # Higher structured patterns = more negative entropy = higher info rate
            max_entropy = torch.log2(torch.tensor(float(fractal_geometry.numel())))
            normalized_entropy = pattern_entropy / max_entropy
            
            # Information rate: 1 - normalized_entropy (more structure = higher rate)
            info_rate = 1.0 - normalized_entropy.item()
            
            return max(0.0, info_rate)  # Ensure non-negative
            
        except Exception as e:
            print(f"    QBE: Error calculating info rate: {e}")
            return 0.0
    
    def compute_qpl_regulation(self, energy_rate: float, info_rate: float) -> float:
        """Compute Quantum Potential Layer regulation term QPL(t)"""
        # QPL acts as a stabilizing function that prevents runaway growth
        # Higher combined rates = stronger regulation needed
        
        # Base regulation proportional to total rate
        total_rate = energy_rate + info_rate
        
        # Adaptive regulation based on system temperature
        temperature_factor = self.system_temperature / 300.0  # Normalized to room temp
        
        # QPL regulation balances the system
        # High rates need stronger regulation, high temperature increases regulation
        qpl_regulation = total_rate * (1.0 + 0.5 * temperature_factor)
        
        return qpl_regulation
    
    def apply_qbe_balance(self, energy_rate: float, info_rate: float, 
                         qpl_regulation: float) -> tuple[float, float]:
        """Enhanced QBE with pressure explosion prevention and adaptive tuning"""
        
        # Prevent infinite/NaN values that cause pressure explosion
        if not (torch.isfinite(torch.tensor(energy_rate)) and torch.isfinite(torch.tensor(info_rate))):
            print(f"  QBE: Detected infinite rates E={energy_rate}, I={info_rate} - applying emergency scaling")
            energy_rate = max(-1000.0, min(1000.0, energy_rate if torch.isfinite(torch.tensor(energy_rate)) else 0.0))
            info_rate = max(-1000.0, min(1000.0, info_rate if torch.isfinite(torch.tensor(info_rate)) else 0.0))
        
        # Enhanced QBE: dE/dt + dI/dt = λ·QPL(t) with adaptive regulation
        lambda_factor = self.qbe_lambda
        
        # Use adaptive controller if available
        if self.adaptive_controller:
            thresholds = self.adaptive_controller.get_adaptive_thresholds()
            lambda_factor *= thresholds.field_sensitivity
        
        target_balance = lambda_factor * qpl_regulation
        current_total = energy_rate + info_rate
        
        # Pressure explosion prevention
        if abs(current_total) > self.pressure_explosion_limit:
            explosion_scale = self.pressure_explosion_limit / abs(current_total)
            energy_rate *= explosion_scale
            info_rate *= explosion_scale
            current_total = energy_rate + info_rate
            print(f"  QBE: Pressure explosion prevented - scaled by {explosion_scale:.3f}")
        
        # Apply QBE balance regulation
        if current_total > target_balance and current_total > 0:
            # Scale down both rates proportionally to maintain balance
            scale_factor = target_balance / current_total
            balanced_energy = energy_rate * scale_factor
            balanced_info = info_rate * scale_factor
            
            # Update adaptive controller with balance metrics
            if self.adaptive_controller:
                balance_ratio = abs(energy_rate) / (abs(info_rate) + 1e-8)
                field_balance = 1.0 / (1.0 + abs(balance_ratio - self.qbe_target_ratio))
                self.adaptive_controller.update_field_metrics(
                    field_pressure=current_total,
                    field_variance=abs(energy_rate - info_rate),
                    collapse_count=1 if scale_factor < 0.9 else 0,
                    field_balance=field_balance
                )
        else:
            # Already balanced or rates are very low
            balanced_energy = energy_rate
            balanced_info = info_rate
        
        return balanced_energy, balanced_info
    
    def crystallize_to_fractal_geometry(self, field: torch.Tensor, balance_field: torch.Tensor) -> torch.Tensor:
        """Convert field state to fractal geometric pattern for efficient storage"""
        try:
            # Extract geometric features that capture learning patterns
            with torch.no_grad():
                # 1. Dominant frequency analysis
                field_fft = torch.fft.fft2(field)
                power_spectrum = torch.abs(field_fft)**2
                
                # 2. Extract key geometric features
                # Center mass distribution
                x_coords = torch.arange(32, device=self.device).float()
                y_coords = torch.arange(32, device=self.device).float()
                X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
                
                total_mass = torch.sum(torch.abs(field)) + 1e-8
                center_x = torch.sum(X * torch.abs(field)) / total_mass
                center_y = torch.sum(Y * torch.abs(field)) / total_mass
                
                # 3. Fractal self-similarity pattern
                fractal_pattern = torch.zeros(8, 8, device=self.device)  # Compressed representation
                for i in range(8):
                    for j in range(8):
                        # Multi-scale sampling
                        region_4x4 = field[i*4:(i+1)*4, j*4:(j+1)*4]
                        region_2x2 = field[i*2:(i+1)*2, j*2:(j+1)*2] if i < 16 and j < 16 else torch.zeros(2, 2, device=self.device)
                        
                        # Combine scales with balance field influence
                        balance_influence = torch.mean(balance_field[i*4:(i+1)*4, j*4:(j+1)*4])
                        scale_mix = torch.mean(region_4x4) + 0.3 * balance_influence
                        fractal_pattern[i, j] = scale_mix
                
                # 4. Add geometric invariants
                # Rotational symmetry measure
                rotation_symmetry = self.compute_rotational_symmetry(fractal_pattern)
                
                # Scale invariance measure  
                scale_invariance = self.compute_scale_invariance(field)
                
                # 5. Create compact geometric signature (32x32 to match accumulated_memory)
                geometric_signature = torch.zeros(32, 32, device=self.device)
                # Place fractal pattern in center
                geometric_signature[12:20, 12:20] = fractal_pattern
                # Add metadata around edges
                geometric_signature[0, 0] = center_x / 32.0  # Normalized center
                geometric_signature[0, 1] = center_y / 32.0
                geometric_signature[31, 30] = rotation_symmetry
                geometric_signature[31, 31] = scale_invariance
                
                return geometric_signature
                
        except Exception as e:
            print(f"    FRACTAL: Crystallization error {e}, using fallback")
            return torch.zeros(32, 32, device=self.device)
    
    def compute_rotational_symmetry(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute rotational symmetry measure of pattern"""
        rotated_90 = torch.rot90(pattern, 1)
        rotated_180 = torch.rot90(pattern, 2)
        rotated_270 = torch.rot90(pattern, 3)
        
        symmetry = torch.mean(torch.abs(pattern - rotated_180))  # 180-degree symmetry
        return torch.clamp(1.0 - symmetry, 0, 1)
    
    def compute_scale_invariance(self, field: torch.Tensor) -> torch.Tensor:
        """Compute scale invariance measure"""
        # Compare field with its downsampled version
        downsampled = torch.nn.functional.avg_pool2d(
            field.unsqueeze(0).unsqueeze(0), 
            kernel_size=2, 
            stride=2
        ).squeeze()
        
        upsampled = torch.nn.functional.interpolate(
            downsampled.unsqueeze(0).unsqueeze(0),
            size=(32, 32),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        scale_similarity = 1.0 - torch.mean(torch.abs(field - upsampled))
        return torch.clamp(scale_similarity, 0, 1)
    
    def store_fractal_pattern(self, geometric_signature: torch.Tensor, tokens: List[str],
                             energy_component: float = 1.0, info_component: float = 1.0):
        """Store fractal pattern with QBE-balanced energy and information components"""
        pattern_key = self.compute_pattern_key(geometric_signature, tokens)
        
        # Update usage statistics
        if pattern_key in self.memory_usage_stats:
            self.memory_usage_stats[pattern_key] += 1
        else:
            self.memory_usage_stats[pattern_key] = 1
        
        # Calculate pattern entropy for thermodynamic assessment
        pattern_entropy = compute_pattern_entropy(geometric_signature)
        
        # QBE-regulated pattern value: balance energy and information
        qbe_value = energy_component + info_component
        
        # Store the pattern with QBE balance information
        self.fractal_memory_bank[pattern_key] = {
            'geometry': geometric_signature.clone(),
            'tokens': tokens,
            'usage_count': self.memory_usage_stats[pattern_key],
            'timestamp': len(self.memory_usage_stats),
            'entropy': pattern_entropy.item(),
            'energy_component': energy_component,
            'info_component': info_component,
            'qbe_value': qbe_value
        }
        
        # QBE-based thermodynamic decisions for memory management
        bank_utilization = len(self.fractal_memory_bank) / self.memory_bank_size
        
        if bank_utilization > 0.9:  # Bank is nearly full
            # Use QBE value instead of just entropy for quality assessment
            avg_qbe_value = sum(data['qbe_value'] for data in self.fractal_memory_bank.values()) / len(self.fractal_memory_bank)
            
            if qbe_value > avg_qbe_value * 1.2:  # High QBE value pattern
                # This pattern has good energy-information balance, consider growing
                if self.memory_bank_size < 100:  # Cap at reasonable size
                    growth_factor = min(1.1, 1.0 + qbe_value / 10.0)
                    self.memory_bank_size = int(self.memory_bank_size * growth_factor)
                    print(f"    QBE: Growing memory bank to {self.memory_bank_size} for high-QBE pattern (QBE={qbe_value:.3f})")
                else:
                    # Bank at max size, prune to make room
                    self.prune_memory_bank()
            else:
                # Normal or low-quality pattern, prune if needed
                self.prune_memory_bank()
        
        elif bank_utilization < 0.3 and self.memory_bank_size > 20:  # Bank underutilized
            # Consider shrinking if patterns have consistently low QBE values
            recent_qbe_values = [data['qbe_value'] for data in list(self.fractal_memory_bank.values())[-10:]]
            if recent_qbe_values and sum(recent_qbe_values) / len(recent_qbe_values) < 1.0:  # Low average QBE
                self.memory_bank_size = max(20, int(self.memory_bank_size * 0.95))
                print(f"    QBE: Shrinking underutilized memory bank to {self.memory_bank_size}")
        
        # Standard pruning if over capacity
        elif len(self.fractal_memory_bank) > self.memory_bank_size:
            self.prune_memory_bank()
    
    def compute_pattern_key(self, geometric_signature: torch.Tensor, tokens: List[str]) -> str:
        """Generate a key for the geometric pattern"""
        # Create hash from geometry and semantic content
        geom_hash = str(hash(geometric_signature.flatten().cpu().numpy().tobytes()))[:8]
        token_hash = str(hash(tuple(tokens)))[:8]
        return f"g{geom_hash}_t{token_hash}"
    
    def prune_memory_bank(self):
        """Intelligently prune memory bank using thermodynamic principles and Landauer's Principle"""
        if len(self.fractal_memory_bank) <= 2:  # Keep minimum patterns
            return
            
        # Update system temperature based on entropy variance
        if len(self.entropy_history) > 0:
            entropy_variance = torch.var(torch.tensor(self.entropy_history))
            # Adaptive temperature: higher variance = higher temperature = more aggressive pruning
            self.system_temperature = 300.0 * (1.0 + 0.5 * entropy_variance.item())
        
        # Calculate QBE-based thermodynamic scores for all patterns
        pattern_scores = {}
        current_time = len(self.memory_usage_stats)
        
        for key, data in self.fractal_memory_bank.items():
            usage_count = data['usage_count']
            timestamp = data['timestamp']
            pattern = data['geometry']
            qbe_value = data.get('qbe_value', 1.0)  # Default for older patterns
            energy_comp = data.get('energy_component', 0.5)
            info_comp = data.get('info_component', 0.5)
            
            # Calculate recency (0-1, 1 = most recent)
            recency = 1.0 - (current_time - timestamp) / max(current_time, 1)
            
            # Enhanced thermodynamic scoring with QBE balance
            thermo_score = thermodynamic_pattern_score(
                pattern, usage_count, self.system_temperature, recency
            )
            
            # Weight by QBE balance: patterns with good energy-info balance are more valuable
            qbe_weight = 1.0 + 0.5 * qbe_value  # Boost for well-balanced patterns
            energy_info_balance = min(energy_comp, info_comp) / max(energy_comp + 1e-6, info_comp + 1e-6)  # Prefer balanced
            
            combined_score = thermo_score * qbe_weight * (1.0 + energy_info_balance)
            pattern_scores[key] = combined_score
        
        # Sort by thermodynamic score (higher = more valuable to keep)
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Determine how many to keep based on thermodynamic efficiency
        total_patterns = len(sorted_patterns)
        
        # Calculate system-wide efficiency metrics
        avg_score = sum(score for _, score in sorted_patterns) / total_patterns
        score_variance = torch.var(torch.tensor([score for _, score in sorted_patterns]))
        
        # Adaptive retention rate based on score distribution
        if score_variance > 0.5:  # High variance = clear winners and losers
            retention_rate = 0.6  # Keep fewer, higher quality patterns
        else:  # Low variance = patterns are similar quality
            retention_rate = 0.8  # Keep more patterns
        
        patterns_to_keep_count = max(5, int(total_patterns * retention_rate))
        patterns_to_keep = sorted_patterns[:patterns_to_keep_count]
        
        # Thermodynamic growth decision: create new pattern slots if needed
        if avg_score > self.growth_threshold and len(patterns_to_keep) < self.memory_bank_size:
            # System is performing well, allow more pattern storage
            self.memory_bank_size = min(100, int(self.memory_bank_size * 1.1))
            print(f"    THERMO: Growing memory bank to {self.memory_bank_size} slots (avg_score={avg_score:.3f})")
        
        # Rebuild memory bank with thermodynamically selected patterns
        new_bank = {}
        new_stats = {}
        total_erasure_cost = 0.0
        
        for key, score in patterns_to_keep:
            new_bank[key] = self.fractal_memory_bank[key]
            new_stats[key] = self.memory_usage_stats[key]
        
        # Calculate energy cost of erased patterns (for logging)
        for key, score in sorted_patterns[patterns_to_keep_count:]:
            pattern = self.fractal_memory_bank[key]['geometry']
            pattern_entropy = compute_pattern_entropy(pattern)
            erasure_cost = landauer_energy_cost(pattern_entropy, self.system_temperature)
            total_erasure_cost += erasure_cost.item()
        
        self.fractal_memory_bank = new_bank
        self.memory_usage_stats = new_stats
        
        print(f"    THERMO: Pruned {total_patterns - len(new_bank)} patterns "
              f"(T={self.system_temperature:.1f}K, erasure_cost={total_erasure_cost:.3f})")
        print(f"    THERMO: Retained {len(new_bank)} patterns with avg_score={avg_score:.3f}")
    
    def update_entropy_tracking(self, field: torch.Tensor):
        """Track system entropy for thermodynamic temperature control"""
        field_entropy = compute_pattern_entropy(field)
        self.entropy_history.append(field_entropy.item())
        
        # Keep only recent history to avoid unbounded growth
        if len(self.entropy_history) > 100:
            self.entropy_history = self.entropy_history[-50:]  # Keep last 50 measurements
    
    def reconstruct_from_fractals(self) -> torch.Tensor:
        """Reconstruct memory field from stored fractal patterns using QBE weighting"""
        if not self.fractal_memory_bank:
            return torch.zeros(32, 32, device=self.device)
        
        # Weight patterns by QBE balance and usage frequency
        memory_field = torch.zeros(32, 32, device=self.device)
        total_weight = 0
        
        for pattern_data in self.fractal_memory_bank.values():
            geometry = pattern_data['geometry']
            usage_weight = min(pattern_data['usage_count'], 5) / 5.0  # Cap influence
            
            # QBE-based weighting: prefer balanced energy-information patterns
            qbe_value = pattern_data.get('qbe_value', 1.0)
            energy_comp = pattern_data.get('energy_component', 0.5)
            info_comp = pattern_data.get('info_component', 0.5)
            
            # Balance factor: reward patterns with good energy-information balance
            balance_factor = 1.0 + 0.3 * min(energy_comp, info_comp) / max(energy_comp + 1e-6, info_comp + 1e-6)
            
            # Combined QBE weight
            qbe_weight = usage_weight * qbe_value * balance_factor
            
            # Use geometry directly since it's already 32x32
            memory_field += geometry * qbe_weight
            total_weight += qbe_weight
        
        if total_weight > 0:
            memory_field /= total_weight
        
        return memory_field
        
        # Safe normalization with balance field guidance
        field_std = torch.std(field)
        field_mean = torch.mean(field)
        
        if field_std > 1e-4:
            # Use balance field to guide normalization strength
            balance_strength = torch.mean(torch.abs(balance_field))
            norm_factor = max(0.5, min(2.0, balance_strength))  # Adaptive normalization
            field = norm_factor * (field - field_mean) / field_std
        elif torch.abs(field_mean) > 1e-6:
            field = field - field_mean
        else:
            # Guided noise injection using balance field pattern
            field = field + 0.1 * balance_field + 0.05 * torch.randn_like(field, device=self.device)
            field_std = torch.std(field)
            if field_std > 1e-6:
                field = (field - torch.mean(field)) / field_std
        
        # Final clipping with balance-aware bounds
        max_val = 10.0 + 5.0 * torch.mean(torch.abs(balance_field))  # Dynamic clipping
        field = torch.clamp(field, -max_val, max_val)
        
        return field
        
        return field
    
    def learn_sentence(self, sentence_data: Dict):
        """Learn from one sentence with robust error handling and detailed logging"""
        text = sentence_data.get('text', '')
        tokens = sentence_data.get('tokens', [])
        pattern = sentence_data.get('grammar_pattern', 'unknown')
        difficulty = sentence_data.get('difficulty', 'beginner')
        
        print(f"Learning: '{text}' [{pattern}]")
        print(f"  DEBUG: Tokens={tokens}, Length={len(tokens)}")
        
        learning_result = {
            'text': text,
            'tokens': tokens,
            'pattern': pattern,
            'difficulty': difficulty,
            'timestamp': datetime.now().isoformat(),
            'injection_result': None,
            'evolution_result': None,
            'success': False,
            'debug_info': {},
            'error_trace': None
        }
        
        try:
            # Step 1: Field Encoding with detailed logging
            print(f"  DEBUG: Step 1 - Encoding to field...")
            field = self.encode_to_field(tokens)
            field_stats = {
                'shape': field.shape,
                'mean': torch.mean(field).item(),
                'std': torch.std(field).item(),
                'min': torch.min(field).item(),
                'max': torch.max(field).item(),
                'nan_count': torch.isnan(field).sum().item(),
                'inf_count': torch.isinf(field).sum().item()
            }
            print(f"  DEBUG: Field encoded - {field_stats}")
            learning_result['debug_info']['field_stats'] = field_stats
            
            # Ensure field has valid values before proceeding
            if torch.isnan(field).any() or torch.isinf(field).any():
                print(f"  ⚠️ Invalid field values detected: NaN={field_stats['nan_count']}, Inf={field_stats['inf_count']}")
                learning_result['error'] = f"Invalid field values: NaN={field_stats['nan_count']}, Inf={field_stats['inf_count']}"
                self.learning_log.append(learning_result)
                return learning_result
            
            # Step 2: Pattern tensor creation and fractal geometry preparation
            print(f"  DEBUG: Step 2 - Creating pattern tensor...")
            pattern_tensor = field.flatten()
            pattern_stats = {
                'size': pattern_tensor.shape[0],
                'mean': torch.mean(pattern_tensor).item(),
                'std': torch.std(pattern_tensor).item(),
                'min': torch.min(pattern_tensor).item(),
                'max': torch.max(pattern_tensor).item()
            }
            print(f"  DEBUG: Pattern tensor - {pattern_stats}")
            learning_result['debug_info']['pattern_stats'] = pattern_stats
            
            # Create crystallized geometry for fractal reuse analysis
            balance_field = self.recursive_balance_field(
                self.gaia.field_engine.get_field_state().energy_field, 
                self.gaia.field_engine.get_field_state().information_field, 
                self.accumulated_memory
            )
            crystallized_geometry = self.crystallize_to_fractal_geometry(field, balance_field)
            
            # Step 3: Amplification and injection with logging
            print(f"  DEBUG: Step 3 - Amplifying and injecting...")
            # MASSIVELY amplify pattern to reach crystallization pressure (>0.3)
            pattern_tensor = pattern_tensor * 50.0  # Increased from 15.0 for crystallization
            amp_stats = {
                'amplified_mean': torch.mean(pattern_tensor).item(),
                'amplified_std': torch.std(pattern_tensor).item()
            }
            print(f"  DEBUG: Amplified pattern - {amp_stats}")
            learning_result['debug_info']['amplified_stats'] = amp_stats
            
            # Don't reset engines - let field pressure accumulate across sentences
            # Only reset every 10 sentences to allow pressure buildup
            if self.stats['sentences'] % 10 == 0:
                print(f"  DEBUG: Resetting engines (every 10 sentences)")
                self.gaia.reset_engines()
            
            # Step 4: Energy field injection with logging
            print(f"  DEBUG: Step 4 - Energy field injection...")
            try:
                self.gaia.field_engine.inject_stimulus(pattern_tensor, stimulus_type="energy")
                print(f"  DEBUG: Energy injection successful")
            except Exception as e:
                print(f"  ERROR: Energy injection failed: {e}")
                raise
            
            # Step 5: Enhanced Information field injection with fractal reuse optimization
            print(f"  DEBUG: Step 5 - Information field injection...")
            # Create strong information field injection based on grammar patterns and fractal reuse
            info_pattern = torch.zeros_like(pattern_tensor)
            
            # Check for existing fractal pattern reuse opportunity
            fractal_reuse_bonus = 0.0
            if pattern in self.patterns:
                # Use pattern frequency to create strong information field structure
                base_strength = self.patterns[pattern]['count'] * 5.0  # Increased
                
                # Check if we have crystallized fractal geometry for this pattern
                geometric_key = self.compute_pattern_key(crystallized_geometry, tokens)
                if geometric_key in self.fractal_memory_bank:
                    # Reuse crystallized fractal pattern for efficiency
                    reuse_data = self.fractal_memory_bank[geometric_key]
                    reuse_geometry = reuse_data['geometry']
                    
                    # Apply fractal reuse bonus
                    fractal_reuse_bonus = self.fractal_reuse_bonus * reuse_data.get('qbe_value', 1.0)
                    enhanced_strength = base_strength * (1.0 + fractal_reuse_bonus)
                    
                    # Blend pattern tensor with reused fractal geometry
                    reuse_weight = min(0.7, reuse_data['usage_count'] / 10.0)  # More reuse = more trust
                    # Flatten reuse_geometry to match pattern_tensor shape
                    reuse_geometry_flat = reuse_geometry[:32, :32].flatten()
                    blended_pattern = (1.0 - reuse_weight) * pattern_tensor + reuse_weight * reuse_geometry_flat
                    info_pattern = blended_pattern * enhanced_strength * 0.5
                    
                    # Update usage count for reused pattern
                    self.fractal_memory_bank[geometric_key]['usage_count'] += 1
                    
                    print(f"  DEBUG: Reusing fractal pattern (bonus={fractal_reuse_bonus:.3f}, strength={enhanced_strength:.1f})")
                else:
                    # Standard existing pattern without fractal reuse
                    info_pattern = pattern_tensor * base_strength * 0.5
                    print(f"  DEBUG: Using existing pattern strength: {base_strength}")
            else:
                # Even new patterns get some information field activation
                info_pattern = pattern_tensor * 0.3
                print(f"  DEBUG: New pattern, using base strength: 0.3")
            
            try:
                self.gaia.field_engine.inject_stimulus(info_pattern, stimulus_type="information")
                print(f"  DEBUG: Information injection successful")
            except Exception as e:
                print(f"  ERROR: Information injection failed: {e}")
                raise
            
            # Step 6: Field state analysis with logging
            print(f"  DEBUG: Step 6 - Analyzing field state...")
            field_state = self.gaia.field_engine.get_field_state()
            energy_var = torch.var(field_state.energy_field).item()
            info_var = torch.var(field_state.information_field).item()
            result = (energy_var, info_var)
            
            # NaN protection for field state analysis
            energy_mean = torch.mean(field_state.energy_field).item()
            energy_std = torch.std(field_state.energy_field).item()
            info_mean = torch.mean(field_state.information_field).item()
            info_std = torch.std(field_state.information_field).item()
            entropy_mean = torch.mean(field_state.entropy_tensor).item()
            
            # Replace NaN values with small random values to maintain dynamics
            if torch.isnan(torch.tensor(energy_mean)) or torch.isinf(torch.tensor(energy_mean)):
                energy_mean = torch.randn(1).item() * 0.001
            if torch.isnan(torch.tensor(energy_std)) or torch.isinf(torch.tensor(energy_std)):
                energy_std = abs(torch.randn(1).item() * 0.001)
            if torch.isnan(torch.tensor(info_mean)) or torch.isinf(torch.tensor(info_mean)):
                info_mean = torch.randn(1).item() * 0.001
            if torch.isnan(torch.tensor(info_std)) or torch.isinf(torch.tensor(info_std)):
                info_std = abs(torch.randn(1).item() * 0.001)
            if torch.isnan(torch.tensor(entropy_mean)) or torch.isinf(torch.tensor(entropy_mean)):
                entropy_mean = abs(torch.randn(1).item() * 0.001)
            
            field_state_stats = {
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'info_mean': info_mean,
                'info_std': info_std,
                'entropy_mean': entropy_mean,
                'field_pressure': field_state.field_pressure
            }
            print(f"  DEBUG: Field state - {field_state_stats}")
            learning_result['debug_info']['field_state_stats'] = field_state_stats
            learning_result['injection_result'] = result
            print(f"  Injected: {result}")
            
            # Step 7: Evolution with detailed logging and error tracking
            print(f"  DEBUG: Step 7 - Starting evolution (50 steps)...")
            evolution = []
            collapse_attempts = 0
            
            # Initialize QBE tracking
            energy_history = []
            info_history = []
            balance_violation_count = 0
            
            for step in range(50):
                # Check field state before step
                field_state = self.gaia.field_engine.get_field_state()
                
                # Safe field pressure calculation with NaN protection
                entropy_tensor = self.gaia.field_engine.entropy_tensor
                if torch.isnan(entropy_tensor).any() or torch.isinf(entropy_tensor).any():
                    field_pressure = 0.0  # Reset to safe value
                    # Reset entropy tensor to prevent NaN propagation
                    self.gaia.field_engine.entropy_tensor = torch.zeros_like(entropy_tensor)
                else:
                    field_pressure = torch.mean(entropy_tensor).item()
                
                # Quantum Balance Equation monitoring with NaN protection
                current_energy = torch.mean(field_state.energy_field).item()
                current_info = torch.mean(field_state.information_field).item()
                
                # Check for NaN values in field states using torch functions
                if torch.isnan(torch.tensor(current_energy)) or torch.isinf(torch.tensor(current_energy)):
                    current_energy = 0.0
                if torch.isnan(torch.tensor(current_info)) or torch.isinf(torch.tensor(current_info)):
                    current_info = 0.0
                
                energy_history.append(current_energy)
                info_history.append(current_info)
                
                # Compute balance ratio and stability with NaN protection
                if len(energy_history) > 1:
                    energy_change = energy_history[-1] - energy_history[-2]
                    info_change = info_history[-1] - info_history[-2]
                    
                    # Check for NaN in changes
                    if torch.isnan(torch.tensor(energy_change)) or torch.isinf(torch.tensor(energy_change)):
                        energy_change = 0.0
                    if torch.isnan(torch.tensor(info_change)) or torch.isinf(torch.tensor(info_change)):
                        info_change = 0.0
                    
                    # QBE: dE/dt + dI/dt should be regulated by QPL(t)
                    balance_sum = abs(energy_change) + abs(info_change)
                    
                    # Additional NaN check for balance_sum
                    if torch.isnan(torch.tensor(balance_sum)) or torch.isinf(torch.tensor(balance_sum)):
                        balance_sum = 0.0
                    
                    # Adaptive step scaling based on balance state
                    if balance_sum > 0.1:  # High imbalance - reduce step intensity
                        step_scaling = 0.5
                        balance_violation_count += 1
                    elif balance_sum < 0.01:  # Low activity - boost to encourage evolution
                        step_scaling = 1.5
                    else:  # Good balance - normal operation
                        step_scaling = 1.0
                        
                    # Apply QPL feedback to field if balance violated
                    if balance_violation_count > 3:  # Persistent imbalance
                        # Apply quantum potential layer correction with NaN protection
                        qpl_correction = 0.1 * (current_info - current_energy)
                        if torch.isnan(torch.tensor(qpl_correction)) or torch.isinf(torch.tensor(qpl_correction)):
                            qpl_correction = 0.0
                        correction_field = qpl_correction * torch.ones_like(field_state.energy_field)
                        self.gaia.field_engine.inject_stimulus(correction_field, stimulus_type="energy")
                        balance_violation_count = 0  # Reset counter
                else:
                    step_scaling = 1.0
                
                # Run field evolution step with adaptive scaling
                if step_scaling != 1.0:
                    # Modify field evolution parameters temporarily
                    original_params = getattr(self.gaia.field_engine, '_step_params', {})
                    # Apply scaling through field amplitude modulation
                    field_state = self.gaia.field_engine.get_field_state()
                    scaled_energy = field_state.energy_field * step_scaling
                    self.gaia.field_engine.inject_stimulus(scaled_energy * 0.1, stimulus_type="energy")
                
                collapse_event = self.gaia.field_engine.step()
                collapse_attempts += 1
                
                # If collapse occurred, process it
                if collapse_event:
                    field_state = self.gaia.field_engine.get_field_state()
                    try:
                        structure = self.gaia.collapse_core.process_collapse_event(collapse_event, field_state)
                        if structure:
                            evolution.append(structure)
                    except Exception as e:
                        # Skip this collapse if processing fails
                        pass
                        
                # Debug output every 10 steps with crystallization and balance details
                if step % 10 == 0 and step < 30:  # Only show first few for brevity
                    entropy_delta = collapse_event.entropy_delta if collapse_event else 0.0
                    crystallize_check = ""
                    balance_info = ""
                    
                    if collapse_event:
                        # Enhanced crystallization conditions with adaptive thresholds
                        # Get adaptive thresholds from controller
                        if self.adaptive_controller:
                            thresholds = self.adaptive_controller.get_adaptive_thresholds()
                            entropy_threshold = 0.1 * thresholds.field_sensitivity
                            pressure_threshold = thresholds.collapse_threshold * 1000  # Scale for display
                        else:
                            # Fallback to fixed thresholds
                            entropy_threshold = 0.1
                            pressure_threshold = 0.3
                        
                        entropy_ok = entropy_delta >= entropy_threshold
                        pressure_ok = field_pressure >= pressure_threshold
                        crystallize_check = f" | Crystallize: ΔS={entropy_delta:.3f}{'✓' if entropy_ok else '✗'} P={field_pressure:.3f}{'✓' if pressure_ok else '✗'}"
                        
                        # Update adaptive controller with crystallization results
                        if self.adaptive_controller:
                            crystallization_quality = (entropy_delta / entropy_threshold) * (field_pressure / pressure_threshold)
                            pattern_type = self.adaptive_controller.detect_field_pattern_type()
                            if pattern_type in ["convergence", "chaotic"]:
                                self.adaptive_controller.adapt_for_pattern(pattern_type)
                    
                    # Add balance monitoring info
                    if len(energy_history) > 1:
                        energy_change = abs(energy_history[-1] - energy_history[-2])
                        info_change = abs(info_history[-1] - info_history[-2])
                        balance_sum = energy_change + info_change
                        balance_status = "stable" if balance_sum < 0.05 else "active" if balance_sum < 0.15 else "unstable"
                        balance_info = f" | QBE: Σ|Δ|={balance_sum:.4f}({balance_status}) Viol:{balance_violation_count}"
                    
                    print(f"    Step {step}: pressure={field_pressure:.6f}, collapse={'YES' if collapse_event else 'NO'}{crystallize_check}{balance_info}")
            
            # Summary with balance analysis
            learning_result['evolution_result'] = len(evolution)
            evolution_summary = f"{len(evolution)} structures"
            if evolution:
                # Show some details about evolved structures
                structure_types = [str(type(s).__name__) for s in evolution[:3]]
                evolution_summary += f" ({', '.join(structure_types)})"
            
            # Add balance summary
            final_balance = abs(energy_history[-1] - info_history[-1]) if len(energy_history) > 0 else 0
            balance_summary = f" | Final QBE balance: {final_balance:.4f}, violations: {balance_violation_count}"
            print(f"  Evolved: {evolution_summary}{balance_summary}")
            
            # Update stats
            self.stats['sentences'] += 1
            self.stats['tokens'] += len(tokens)
            
            # Store pattern with more details
            if pattern not in self.patterns:
                self.patterns[pattern] = {'count': 0, 'examples': []}
            self.patterns[pattern]['count'] += 1
            self.patterns[pattern]['examples'].append({
                'text': text,
                'tokens': tokens,
                'evolution_count': len(evolution),
                'injection_strength': result if isinstance(result, (int, float)) else str(result)
            })
            self.stats['patterns'] = len(self.patterns)
            
            learning_result['success'] = True
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"  ✗ Error: {e}")
            print(f"  ✗ Full traceback:")
            print(error_trace)
            
            learning_result['error'] = str(e)
            learning_result['error_trace'] = error_trace
            learning_result['debug_info']['error_location'] = {
                'sentence': text,
                'tokens': tokens,
                'step': 'unknown'
            }
        
        self.learning_log.append(learning_result)
        return learning_result
    
    def learn_progressively(self, sentences: List[Dict], max_count: int = 30):
        """Learn from dataset progressively"""
        print(f"\n=== Progressive Learning ({max_count} sentences) ===")
        
        # Sort by complexity
        sentences.sort(key=lambda x: x.get('complexity_score', 0))
        
        for i, sent in enumerate(sentences[:max_count]):
            print(f"\n--- Session {i+1}/{max_count} ---")
            self.learn_sentence(sent)
            
            if (i + 1) % 10 == 0:
                print(f"\nProgress: {self.stats}")
        
        print(f"\n=== Learning Complete ===")
        print(f"Final stats: {self.stats}")
        print(f"Patterns learned: {list(self.patterns.keys())[:5]}")
    
    def generate_text(self, prompt: str = "", length: int = 6) -> str:
        """Generate text using GAIA with robust error handling"""
        print(f"\n=== Generating from: '{prompt}' ===")
        
        generation_record = {
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'target_length': length,
            'generated_text': '',
            'generation_steps': [],
            'success': False
        }
        
        try:
            if prompt:
                words = prompt.lower().split()
            else:
                # Start with common word
                starters = ['the', 'i', 'hello', 'what', 'this']
                available = [w for w in starters if w in self.vocabulary]
                words = [random.choice(available) if available else 'hello']
            
            generation_record['initial_words'] = words.copy()
        
            # Generate more words
            for step in range(length - len(words)):
                try:
                    # Use last few words as context
                    context = words[-3:]
                    field = self.encode_to_field(context)
                    
                    # Inject field data into GAIA for evolution with higher energy
                    pattern_tensor = field.flatten()
                    # Significantly amplify pattern for better field activation
                    pattern_tensor = pattern_tensor * 20.0  # Higher amplification for generation
                    
                    # Don't reset engines during generation - preserve accumulated field state
                    # Only inject additional energy on top of learned patterns
                    self.gaia.field_engine.inject_stimulus(pattern_tensor, stimulus_type="energy")
                    
                    # Add information field activation based on context
                    info_tensor = pattern_tensor * 0.5  # Information follows energy pattern
                    self.gaia.field_engine.inject_stimulus(info_tensor, stimulus_type="information")
                    
                    # Run evolution
                    evolution = []
                    for step in range(30):
                        # Run field evolution step
                        collapse_event = self.gaia.field_engine.step()
                        
                        # If collapse occurred, process it
                        if collapse_event:
                            field_state = self.gaia.field_engine.get_field_state()
                            try:
                                structure = self.gaia.collapse_core.process_collapse_event(collapse_event, field_state)
                                if structure:
                                    evolution.append(structure)
                            except Exception as e:
                                # Skip this collapse if processing fails
                                pass
                    
                    # Get current field state for word prediction
                    field_state = self.gaia.field_engine.get_field_state()
                    
                    # Calculate injection results for tracking
                    energy_var = torch.var(field_state.energy_field).item()
                    info_var = torch.var(field_state.information_field).item()
                    injection_result = (energy_var, info_var)
                    
                    # Use field state to predict next word
                    next_word = self._predict_next(words[-1] if words else "", field_state)
                    
                    step_record = {
                        'step': step,
                        'context': context,
                        'predicted_word': next_word,
                        'evolution_count': len(evolution),
                        'injection_result': injection_result,
                        'field_has_state': field_state is not None and hasattr(field_state, 'energy_field')
                    }
                    generation_record['generation_steps'].append(step_record)
                    
                    if next_word and next_word in self.vocabulary:
                        words.append(next_word)
                    else:
                        break
                        
                except Exception as e:
                    print(f"Generation error: {e}")
                    generation_record['error'] = str(e)
                    break
            
            result = ' '.join(words)
            generation_record['generated_text'] = result
            generation_record['final_length'] = len(words)
            generation_record['success'] = True
            
            self.generation_results.append(generation_record)
            
            print(f"Generated: '{result}'")
            return result
            
        except Exception as e:
            # Handle any errors during generation gracefully
            print(f"Generation error: {e}")
            result = ' '.join(words) if words else 'error'
            generation_record['generated_text'] = result
            generation_record['error'] = str(e)
            generation_record['success'] = False
            self.generation_results.append(generation_record)
            return result
    
    def _predict_next(self, last_word: str, field_state=None) -> Optional[str]:
        """Predict next word using GAIA field state analysis"""
        if not self.vocabulary:
            return None
        
        # If we have field state, use it for prediction
        if field_state is not None and hasattr(field_state, 'field') and field_state.field is not None:
            field = field_state.field
            
            # Find the most active regions in the field
            activation_threshold = field.mean() + field.std()
            active_mask = field > activation_threshold
            
            if active_mask.sum() > 0:
                # Get active positions
                active_positions = torch.nonzero(active_mask)
                
                # Find candidate words based on field activity
                candidates = []
                for pos in active_positions:
                    row, col = int(pos[0]), int(pos[1])
                    
                    # Map position back to word space
                    for word, word_id in self.vocabulary.items():
                        word_hash = hash(word) % (64 * 64)
                        word_row, word_col = word_hash // 64, word_hash % 64
                        
                        # Check if word position is near active region
                        distance = ((row - word_row)**2 + (col - word_col)**2)**0.5
                        if distance < 5:  # Within activation radius
                            activation_strength = float(field[row, col])
                            candidates.append((word, activation_strength))
                
                # Sort by activation strength and select
                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    # Add some randomness but bias toward high activation
                    if len(candidates) > 1:
                        # Pick from top 3 candidates with weighted randomness
                        top_candidates = candidates[:min(3, len(candidates))]
                        weights = [c[1] + 1 for c in top_candidates]  # Add 1 to avoid negative weights
                        total_weight = sum(weights)
                        
                        if total_weight > 0:
                            rand_val = random.random() * total_weight
                            current_sum = 0
                            for i, (word, strength) in enumerate(top_candidates):
                                current_sum += weights[i]
                                if rand_val <= current_sum:
                                    return word
                    
                    return candidates[0][0]  # Return highest activation word
        
        # Fallback to pattern-based prediction (but make it smarter)
        if last_word:
            # Enhanced continuation patterns using actual vocabulary
            continuations = {}
            
            # Build continuations from learned patterns
            for pattern_type in self.patterns.keys():
                if 'verb' in pattern_type and last_word in ['i', 'you', 'we', 'they']:
                    continuations[last_word] = [w for w in ['am', 'are', 'see', 'think', 'understand', 'feel', 'like'] if w in self.vocabulary]
                elif 'noun' in pattern_type and last_word in ['the', 'a']:
                    continuations[last_word] = [w for w in ['cat', 'dog', 'sun', 'moon', 'time', 'water', 'energy', 'consciousness'] if w in self.vocabulary]
                elif 'question' in pattern_type and last_word in ['what', 'how', 'why', 'when']:
                    continuations[last_word] = [w for w in ['is', 'are', 'do', 'does', 'happens', 'emerges'] if w in self.vocabulary]
            
            # Use learned continuations
            if last_word in continuations and continuations[last_word]:
                return random.choice(continuations[last_word])
        
        # Final fallback - select from most common words in vocabulary
        common_words = list(self.vocabulary.keys())[:50]
        return random.choice(common_words) if common_words else None
    
    def save_results(self):
        """Save all learning and generation results to output folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n=== Saving Results to {self.output_dir} ===")
        
        # Save learning log
        learning_file = os.path.join(self.output_dir, f"learning_log_{timestamp}.json")
        with open(learning_file, 'w') as f:
            json.dump(self.learning_log, f, indent=2)
        print(f"✓ Learning log saved: {learning_file}")
        
        # Save generation results
        generation_file = os.path.join(self.output_dir, f"generation_results_{timestamp}.json")
        with open(generation_file, 'w') as f:
            json.dump(self.generation_results, f, indent=2)
        print(f"✓ Generation results saved: {generation_file}")
        
        # Save vocabulary
        vocab_file = os.path.join(self.output_dir, f"vocabulary_{timestamp}.json")
        vocab_data = {
            'vocabulary': self.vocabulary,
            'word_list': self.word_list,
            'size': len(self.vocabulary)
        }
        with open(vocab_file, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        print(f"✓ Vocabulary saved: {vocab_file}")
        
        # Save patterns
        patterns_file = os.path.join(self.output_dir, f"patterns_{timestamp}.json")
        with open(patterns_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
        print(f"✓ Patterns saved: {patterns_file}")
        
        # Save summary report
        summary_file = os.path.join(self.output_dir, f"summary_report_{timestamp}.md")
        self._generate_summary_report(summary_file)
        print(f"✓ Summary report saved: {summary_file}")
        
        # Save statistics
        stats_file = os.path.join(self.output_dir, f"statistics_{timestamp}.json")
        full_stats = {
            'basic_stats': self.stats,
            'vocabulary_stats': {
                'total_words': len(self.vocabulary),
                'most_common': list(self.patterns.items())[:10]
            },
            'learning_stats': {
                'successful_sentences': len([x for x in self.learning_log if x.get('success', False)]),
                'failed_sentences': len([x for x in self.learning_log if not x.get('success', True)])
            },
            'generation_stats': {
                'total_generations': len(self.generation_results),
                'successful_generations': len([x for x in self.generation_results if x.get('success', False)])
            }
        }
        with open(stats_file, 'w') as f:
            json.dump(full_stats, f, indent=2)
        print(f"✓ Statistics saved: {stats_file}")
    
    def _generate_summary_report(self, filename: str):
        """Generate a human-readable summary report"""
        with open(filename, 'w') as f:
            f.write("# GAIA Language Learning Summary Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Learning Statistics\n")
            f.write(f"- Sentences processed: {self.stats['sentences']}\n")
            f.write(f"- Tokens processed: {self.stats['tokens']}\n")
            f.write(f"- Grammar patterns discovered: {self.stats['patterns']}\n")
            f.write(f"- Vocabulary size: {len(self.vocabulary)}\n\n")
            
            f.write("## Grammar Patterns Learned\n")
            pattern_items = []
            for pattern, data in self.patterns.items():
                if isinstance(data, dict):
                    count = data.get('count', 0)
                    pattern_items.append((pattern, count))
                else:
                    pattern_items.append((pattern, data))
            
            for pattern, count in sorted(pattern_items, key=lambda x: x[1], reverse=True):
                f.write(f"- {pattern}: {count} examples\n")
            f.write("\n")
            
            f.write("## Sample Vocabulary\n")
            sample_words = self.word_list[:20]
            f.write(f"First 20 words: {', '.join(sample_words)}\n\n")
            
            f.write("## Generation Examples\n")
            for i, gen in enumerate(self.generation_results):
                f.write(f"**Generation {i+1}:**\n")
                f.write(f"- Prompt: '{gen['prompt']}'\n")
                f.write(f"- Generated: '{gen['generated_text']}'\n")
                f.write(f"- Length: {gen['final_length']} words\n")
                f.write(f"- Success: {gen['success']}\n\n")
            
            f.write("## Learning Progress\n")
            successful = len([x for x in self.learning_log if x.get('success', False)])
            total = len(self.learning_log)
            f.write(f"Learning success rate: {successful}/{total} ({100*successful/total:.1f}%)\n\n")
            
            f.write("## Sample Learning Entries\n")
            for i, entry in enumerate(self.learning_log[:5]):
                f.write(f"**Entry {i+1}:**\n")
                f.write(f"- Text: '{entry['text']}'\n")
                f.write(f"- Pattern: {entry['pattern']}\n")
                f.write(f"- Difficulty: {entry['difficulty']}\n")
                f.write(f"- Success: {entry['success']}\n\n")
    
    def learn_with_ollama(self, max_iterations: int = 100):
        """
        Dynamic learning using Ollama as teacher
        Phase 1: Teaching phase with adaptive curriculum
        Phase 2: Assessment and conversation phase
        """
        if not self.use_ollama or not self.ollama_teacher:
            system_logger.error("❌ Ollama not available - falling back to static learning")
            return
        
        conversation_logger.info("[START] === DYNAMIC LEARNING WITH OLLAMA ===")
        conversation_logger.info("Phase 1: Teaching Phase")
        conversation_logger.info("Clean console mode - detailed logs in debug files")
        conversation_logger.info("-" * 50)
        
        iteration = 0
        assessment_attempts = 0
        max_assessment_attempts = 5
        
        while iteration < max_iterations:
            # Get current learning progress
            learning_progress = {
                'vocab_size': len(self.vocabulary),
                'pattern_count': len(self.patterns),
                'coherence': self._calculate_coherence_score(),
                'iterations': iteration
            }
            
            # Phase 1: Teaching
            if self.ollama_teacher.teaching_phase:
                # Get adaptive teaching sentence from Ollama
                teaching_sentence = self.ollama_teacher.get_teaching_sentence(learning_progress)
                
                if teaching_sentence:
                    conversation_logger.info(f"[TEACHER] {teaching_sentence}")
                    
                    # Create sentence data structure for learning
                    sentence_data = {
                        'text': teaching_sentence,
                        'tokens': teaching_sentence.lower().split(),  # Add tokenization
                        'complexity_score': self._estimate_complexity(teaching_sentence),
                        'grammar_pattern': 'dynamic',  # Pattern from Ollama teaching
                        'difficulty': 'adaptive'      # Difficulty from Ollama assessment
                    }
                    
                    # Learn from the sentence  (debug output to file only)
                    self.learn_sentence(sentence_data)
                    
                    # Generate response to show learning
                    if iteration % 5 == 0:  # Every 5 iterations
                        generated = self.generate_text("", length=6)
                        conversation_logger.info(f"[GAIA] {generated}")
                        debug_logger.info(f"GAIA generation: '{generated}'")
                    
                    # Check if ready to move to assessment phase
                    if learning_progress['vocab_size'] > 25 and iteration > 20:
                        self.ollama_teacher.start_conversation_phase()
                        conversation_logger.info("[SYSTEM] Moving to assessment phase...")
                
                else:
                    debug_logger.warning("Failed to get teaching sentence from Ollama")
            
            # Phase 2: Assessment
            elif self.ollama_teacher.assessment_phase:
                # Generate GAIA's current output for assessment
                test_prompt = self.ollama_teacher.generate_conversation_prompt()
                gaia_response = self.generate_text(test_prompt, length=8)
                
                conversation_logger.info(f"[QUESTION] {test_prompt}")
                conversation_logger.info(f"[GAIA] {gaia_response}")
                
                # Get Ollama's assessment
                assessment = self.ollama_teacher.assess_learning(gaia_response, learning_progress)
                
                if assessment['ready']:
                    conversation_logger.info(f"[ASSESSMENT] READY! {assessment['reasoning']}")
                    conversation_logger.info("[SUCCESS] GAIA is ready for basic conversation!")
                    break
                else:
                    conversation_logger.info(f"[ASSESSMENT] Continue learning - {assessment['reasoning']}")
                    assessment_attempts += 1
                    
                    if assessment_attempts >= max_assessment_attempts:
                        conversation_logger.info("[SYSTEM] Max assessment attempts reached - switching back to teaching")
                        self.ollama_teacher.teaching_phase = True
                        self.ollama_teacher.assessment_phase = False
                        assessment_attempts = 0
            
            iteration += 1
            
            # Progress update (log to debug, show simplified progress in conversation)
            if iteration % 10 == 0:
                debug_logger.info(f"Progress: Iteration {iteration}, Vocab: {len(self.vocabulary)}, Patterns: {len(self.patterns)}")
                conversation_logger.info(f"[PROGRESS] {iteration} iterations, {len(self.vocabulary)} words learned")
        
        conversation_logger.info(f"[COMPLETE] Dynamic learning completed after {iteration} iterations")
        debug_logger.info(f"Final statistics: {self.stats}")
        
    def _calculate_coherence_score(self) -> float:
        """Calculate a simple coherence score based on vocabulary and patterns"""
        if not self.vocabulary:
            return 0.0
        
        # Simple heuristic: more vocabulary + more patterns = better coherence
        vocab_score = min(len(self.vocabulary) / 50.0, 1.0)  # Cap at 50 words
        pattern_score = min(len(self.patterns) / 20.0, 1.0)  # Cap at 20 patterns
        
        return (vocab_score + pattern_score) / 2.0
    
    def _estimate_complexity(self, sentence: str) -> float:
        """Estimate sentence complexity for learning progression"""
        words = sentence.lower().split()
        
        # Basic complexity factors
        length_factor = min(len(words) / 15.0, 1.0)  # Longer = more complex
        unique_words = len(set(words))
        uniqueness_factor = unique_words / len(words) if words else 0
        
        # Check for complex patterns
        complexity_bonus = 0
        if any(word in sentence.lower() for word in ['because', 'however', 'although', 'therefore']):
            complexity_bonus += 0.2
        if '?' in sentence or '!' in sentence:
            complexity_bonus += 0.1
        
        return min(length_factor + uniqueness_factor + complexity_bonus, 1.0)


def main():
    """Original main function - now calls dynamic learning"""
    main_dynamic()


def main_dynamic():
    """Main function for dynamic Ollama-based learning"""
    system_logger.info("=" * 60)
    system_logger.info("GAIA Dynamic Language Learning with Ollama")
    system_logger.info("=" * 60)
    
    # Initialize learner with Ollama
    learner = GAIALanguageLearner(use_ollama=True)
    
    try:
        # Start dynamic learning
        learner.learn_with_ollama(max_iterations=50)
        
        # Final demonstration
        conversation_logger.info("🎯 === FINAL DEMONSTRATION ===")
        test_prompts = ["hello", "what is", "i am", "time"]
        
        for prompt in test_prompts:
            result = learner.generate_text(prompt, length=6)
            conversation_logger.info(f"'{prompt}' → '{result}'")
        
        # Save results
        learner.save_results()
        
    except KeyboardInterrupt:
        system_logger.info("Learning interrupted by user")
        learner.save_results()
    except Exception as e:
        system_logger.error(f"❌ Error during learning: {e}")
        debug_logger.error(f"Learning error: {e}")
        learner.save_results()


if __name__ == "__main__":
    main()
