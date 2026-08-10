# Signal Decomposer Module

## Overview

The Signal Decomposer is the theory module of the Field Decomposition system, responsible for breaking down complex input signals into constituent frequency bands and symbolic layers. It employs recursive bandpass filtering with adaptive thresholds to extract meaningful structure from high-entropy signals while preserving information for downstream analysis.

## Core Responsibilities

### Multi-Scale Decomposition
- Recursive frequency band separation using adaptive filters
- Wavelet and Fourier transform-based decomposition
- Symbolic sequence segmentation and layering
- Time-frequency analysis for non-stationary signals

### Adaptive Filtering
- Dynamic threshold adjustment based on signal characteristics
- Context-aware filter parameter optimization
- Real-time filter adaptation for streaming data
- Quality-preserving noise separation

### Layer Management
- Hierarchical layer organization and metadata tracking
- Cross-layer relationship modeling
- Layer quality assessment and validation
- Efficient memory management for large signals

### Stream Processing
- Real-time signal decomposition for live data streams
- Buffering and windowing for continuous processing
- Parallel processing for multiple signal channels
- Backpressure handling and flow control
- Handle various signal types (time series, symbolic sequences, etc.)
- Apply appropriate windowing techniques
- Detect and handle anomalies and edge cases

### Recursive Bandpass Filtering
- Apply multi-scale bandpass filters to isolate frequency components
- Recursively decompose each band for finer-grained analysis
- Maintain phase information during decomposition
- Ensure energy conservation across decomposition levels

### Layer Extraction
- Extract meaningful layers from decomposition process
- Identify base symbolic structures (low frequency)
- Isolate mid-layer oscillations (medium frequency)
- Separate high-frequency turbulence (entropy proxies)
- Isolate residual noise (random, structureless entropy)

### Component Management
- Track relationships between decomposed components
- Maintain decomposition hierarchy and metadata
- Provide access to individual components and relationships
- Support recomposition for verification and analysis

## Architecture

### Input Processing Layer
```python
class SignalPreprocessor:
    def normalize_signal(self, signal: np.ndarray) -> np.ndarray
    def detect_anomalies(self, signal: np.ndarray) -> List[Anomaly]
    def apply_window(self, signal: np.ndarray, window_type: str) -> np.ndarray
    def prepare_for_decomposition(self, signal: np.ndarray) -> PreparedSignal
```

### Decomposition Engine
```python
class RecursiveBandpassFilter:
    def decompose(self, signal: PreparedSignal, levels: int) -> DecompositionResult
    def apply_filter(self, signal: np.ndarray, band: Tuple[float, float]) -> np.ndarray
    def recursive_decomposition(self, signal: np.ndarray, depth: int) -> LayerHierarchy
    def verify_conservation(self, original: np.ndarray, decomposed: List[np.ndarray]) -> bool
```

### Layer Management
```python
class LayerExtractor:
    def extract_base_layer(self, decomposition: DecompositionResult) -> SignalLayer
    def extract_mid_layers(self, decomposition: DecompositionResult) -> List[SignalLayer]
    def extract_high_frequency(self, decomposition: DecompositionResult) -> SignalLayer
    def extract_noise(self, decomposition: DecompositionResult) -> SignalLayer
    def create_layer_hierarchy(self, layers: List[SignalLayer]) -> LayerHierarchy
```

### Component Integration
```python
class ComponentManager:
    def store_components(self, layers: LayerHierarchy) -> ComponentRegistry
    def retrieve_component(self, component_id: str) -> SignalLayer
    def recompose_signal(self, components: List[SignalLayer]) -> np.ndarray
    def compare_original_recomposed(self, original: np.ndarray, recomposed: np.ndarray) -> float
```

## Data Structures

### Signal Representation
```python
@dataclass
class PreparedSignal:
    data: np.ndarray
    metadata: Dict[str, Any]
    sampling_rate: float
    dimensions: Tuple[int, ...]
    signal_type: SignalType
```

### Decomposition Result
```python
@dataclass
class DecompositionResult:
    original_signal: PreparedSignal
    decomposition_levels: int
    layers: List[SignalLayer]
    residual: np.ndarray
    conservation_metric: float
```

### Signal Layer
```python
@dataclass
class SignalLayer:
    id: str
    data: np.ndarray
    frequency_band: Tuple[float, float]
    layer_type: LayerType  # BASE, MID, HIGH, NOISE
    parent_id: Optional[str]
    children_ids: List[str]
    metadata: Dict[str, Any]
```

### Layer Hierarchy
```python
@dataclass
class LayerHierarchy:
    root_layer: SignalLayer
    all_layers: Dict[str, SignalLayer]
    max_depth: int
    total_layers: int
```

## Algorithms

### Recursive Bandpass Filtering Algorithm
```python
def recursive_bandpass(signal: np.ndarray, max_depth: int = 5, 
                      freq_divisions: List[Tuple[float, float]] = None) -> List[np.ndarray]:
    """
    Recursively apply bandpass filtering to decompose signal
    
    1. Apply initial frequency band divisions
    2. For each frequency band, check complexity/entropy
    3. If complexity > threshold, recursively decompose that band
    4. Continue until max_depth reached or complexity below threshold
    5. Return hierarchical structure of decomposed signals
    """
```

### Layer Extraction Algorithm
```python
def extract_layers(decomposition_result: List[np.ndarray], 
                  thresholds: Dict[LayerType, float]) -> Dict[LayerType, List[np.ndarray]]:
    """
    Extract meaningful layers from decomposition results
    
    1. Analyze frequency content of each decomposed band
    2. Classify based on frequency characteristics and entropy
    3. Group similar bands into coherent layers
    4. Identify residual/noise components
    5. Return categorized layers with metadata
    """
```

### Signal Recomposition Algorithm
```python
def recompose_signal(layers: Dict[LayerType, List[np.ndarray]],
                    weights: Dict[LayerType, float] = None) -> np.ndarray:
    """
    Recompose original signal from decomposed layers
    
    1. Apply appropriate weights to each layer (if specified)
    2. Sum all layers to reconstruct signal
    3. Verify reconstruction quality
    4. Apply phase corrections if necessary
    5. Return reconstructed signal with quality metrics
    """
```

## Integration Points

### Entropy Analyzer Integration
- Provide decomposed layers for entropy calculation
- Receive entropy feedback for adaptive decomposition
- Support iterative decomposition refinement
- Exchange metadata and analysis context

### SEC Classifier Integration
- Provide layer characteristics for classification
- Receive classification feedback for decomposition tuning
- Support hierarchical classification schemas
- Exchange classification metadata

### Field Reconstructor Integration
- Provide decomposition hierarchy for reconstruction
- Support selective layer inclusion/exclusion
- Enable annotated reconstruction
- Exchange reconstruction quality metrics

## Quality Metrics

### Decomposition Quality
- **Conservation Metric**: Energy/information conservation during decomposition
- **Layer Separation**: Distinctness between extracted layers
- **Hierarchy Quality**: Logical structure of decomposition hierarchy
- **Noise Isolation**: Effectiveness of noise separation

### Signal Processing Quality
- **Frequency Response**: Accuracy of bandpass filtering
- **Phase Preservation**: Phase distortion minimization
- **Edge Effect Management**: Handling of boundary conditions
- **Aliasing Prevention**: Prevention of sampling artifacts

### Reconstruction Quality
- **Fidelity Metric**: Similarity between original and reconstructed signals
- **Component Contribution**: Relative importance of each component
- **Information Preservation**: Critical information retention
- **Perceptual Quality**: Qualitative assessment of reconstruction

## Error Handling

### Signal Anomalies
- Invalid signal format detection and handling
- Non-stationary signal adaptation
- Extreme value management
- Missing data interpolation

### Decomposition Failures
- Filter instability detection
- Decomposition limit warnings
- Conservation violation alerts
- Computational resource management

## Future Enhancements

### Advanced Decomposition Techniques
- Wavelet-based decomposition options
- Empirical Mode Decomposition (EMD) integration
- Adaptive filter parameter optimization
- ML-enhanced decomposition strategies

### Signal Type Extensions
- Multidimensional signal support (images, volumes)
- Symbolic sequence specialized processing
- Non-linear signal handling improvements
- Quantum state decomposition techniques

### Performance Optimization
- GPU-accelerated decomposition
- Distributed processing for large signals
- Incremental/streaming decomposition
- Adaptive precision control

## Benchmarks and Validation

### Lorenz Attractor Tests
- Standard decomposition benchmarks using Lorenz attractors
- Component separation validation
- Reconstruction quality verification
- Processing efficiency measurement

### Synthetic Signal Tests
- Known component mixture decomposition tests
- Noise resilience evaluation
- Edge case handling assessment
- Scaling behavior analysis

### Real-world Signal Validation
- AI model activation decomposition tests
- Symbolic field analysis validation
- Integration test cases with SCBF and GAIA
- Practical application benchmarks
