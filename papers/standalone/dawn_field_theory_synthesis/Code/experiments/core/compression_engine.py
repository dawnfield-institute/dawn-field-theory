"""
Core Compression Engine for Information Amplification Measurement

Provides compression algorithms and measurement utilities without interpretation.
"""

import json
import zlib
import lzma
import gzip
from typing import Dict, Any


class CompressionEngine:
    """Core compression measurement engine."""
    
    def __init__(self):
        self.algorithms = ['lzma', 'gzip', 'zlib']
    
    def measure_compression(self, data: str) -> Dict[str, Any]:
        """
        Measure information content via multiple compression algorithms.
        Returns raw measurements without interpretation.
        """
        data_bytes = data.encode('utf-8')
        raw_size = len(data_bytes)
        
        compressions = {}
        
        # Try LZMA
        try:
            lzma_compressed = lzma.compress(data_bytes, preset=9)
            compressions['lzma'] = len(lzma_compressed)
        except Exception:
            compressions['lzma'] = None
        
        # Try Gzip
        try:
            gzip_compressed = gzip.compress(data_bytes, compresslevel=9)
            compressions['gzip'] = len(gzip_compressed)
        except Exception:
            compressions['gzip'] = None
        
        # Try Zlib
        try:
            zlib_compressed = zlib.compress(data_bytes, level=9)
            compressions['zlib'] = len(zlib_compressed)
        except Exception:
            compressions['zlib'] = None
        
        # Select optimal compression
        valid_compressions = {k: v for k, v in compressions.items() if v is not None}
        
        if valid_compressions:
            best_algorithm = min(valid_compressions, key=valid_compressions.get)
            compressed_size = valid_compressions[best_algorithm]
        else:
            # Fallback estimation
            unique_chars = len(set(data))
            compressed_size = int(raw_size * (unique_chars / 256))
            best_algorithm = 'estimated'
        
        compression_ratio = compressed_size / raw_size if raw_size > 0 else 0
        
        return {
            'raw_size': raw_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'best_algorithm': best_algorithm,
            'all_compressions': compressions
        }
    
    def measure_object(self, obj: Any) -> Dict[str, Any]:
        """Measure compression of Python object via JSON serialization."""
        try:
            data_str = json.dumps(obj, sort_keys=True)
            return self.measure_compression(data_str)
        except Exception as e:
            return {
                'error': str(e),
                'raw_size': 0,
                'compressed_size': 0,
                'compression_ratio': 0,
                'best_algorithm': 'error'
            }
