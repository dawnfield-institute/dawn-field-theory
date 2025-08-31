"""
CIP-SC Protocol Implementation

This module provides the protocol-level implementation for CIP-SC,
including serialization, validation, and format handling.
"""

import json
import struct
import zlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
import hashlib
import time
from enum import Enum

from .compression_engine import (
    CompressedData, CompressionHeader, SemanticPayload, ValidationData,
    SemanticInstruction, CompressionType
)

# ============================================================================
# Protocol Constants
# ============================================================================

CIPSC_MAGIC = b'CIPSC'
PROTOCOL_VERSION = "1.0"
SCHEMA_VERSION = "1.0"
MAX_SYMBOL_COUNT = 10000
MAX_PAYLOAD_SIZE = 100 * 1024 * 1024  # 100MB

# ============================================================================
# Protocol Serialization
# ============================================================================

class CIPSCProtocol:
    """CIP-SC protocol implementation for serialization and validation."""
    
    @staticmethod
    def serialize(compressed_data: CompressedData) -> bytes:
        """
        Serialize CompressedData to CIP-SC binary format.
        
        Format:
        - Magic bytes (5): 'CIPSC'
        - Version (1): Protocol version byte
        - Header length (4): Length of JSON header
        - Header (variable): JSON-encoded header
        - Payload length (4): Length of payload section
        - Payload (variable): Binary payload data
        - Validation length (4): Length of validation section
        - Validation (variable): JSON-encoded validation
        - Checksum (32): SHA256 of entire data
        """
        
        # Serialize components
        header_json = json.dumps(asdict(compressed_data.header), sort_keys=True).encode('utf-8')
        
        # Serialize payload
        payload_data = CIPSCProtocol._serialize_payload(compressed_data.payload)
        
        # Serialize validation
        validation_json = json.dumps(asdict(compressed_data.validation), sort_keys=True).encode('utf-8')
        
        # Build binary format
        buffer = bytearray()
        
        # Magic and version
        buffer.extend(CIPSC_MAGIC)
        buffer.append(0x10)  # Version 1.0
        
        # Header
        buffer.extend(struct.pack('<I', len(header_json)))
        buffer.extend(header_json)
        
        # Payload
        buffer.extend(struct.pack('<I', len(payload_data)))
        buffer.extend(payload_data)
        
        # Validation
        buffer.extend(struct.pack('<I', len(validation_json)))
        buffer.extend(validation_json)
        
        # Checksum
        checksum = hashlib.sha256(buffer).digest()
        buffer.extend(checksum)
        
        return bytes(buffer)
    
    @staticmethod
    def deserialize(data: bytes) -> CompressedData:
        """
        Deserialize CIP-SC binary format to CompressedData.
        
        Raises:
            ValueError: If data format is invalid
            RuntimeError: If checksum validation fails
        """
        
        if len(data) < 50:  # Minimum viable size
            raise ValueError("Data too short for CIP-SC format")
        
        offset = 0
        
        # Check magic
        magic = data[offset:offset+5]
        if magic != CIPSC_MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")
        offset += 5
        
        # Check version
        version = data[offset]
        if version != 0x10:
            raise ValueError(f"Unsupported version: {version}")
        offset += 1
        
        # Verify checksum first
        checksum_start = len(data) - 32
        stored_checksum = data[checksum_start:]
        calculated_checksum = hashlib.sha256(data[:checksum_start]).digest()
        
        if stored_checksum != calculated_checksum:
            raise RuntimeError("Checksum validation failed")
        
        # Parse header
        header_length = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        header_json = data[offset:offset+header_length].decode('utf-8')
        header_dict = json.loads(header_json)
        header = CompressionHeader(**header_dict)
        offset += header_length
        
        # Parse payload
        payload_length = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        payload_data = data[offset:offset+payload_length]
        payload = CIPSCProtocol._deserialize_payload(payload_data)
        offset += payload_length
        
        # Parse validation
        validation_length = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        validation_json = data[offset:offset+validation_length].decode('utf-8')
        validation_dict = json.loads(validation_json)
        validation = ValidationData(**validation_dict)
        
        return CompressedData(header, payload, validation)
    
    @staticmethod
    def _serialize_payload(payload: SemanticPayload) -> bytes:
        """Serialize semantic payload to binary format."""
        buffer = bytearray()
        
        # Entropy seed
        seed_length = len(payload.entropy_seed)
        buffer.extend(struct.pack('<I', seed_length))
        buffer.extend(payload.entropy_seed)
        
        # Symbols
        symbols_json = json.dumps([asdict(s) for s in payload.symbols], sort_keys=True).encode('utf-8')
        # Compress symbols for efficiency
        compressed_symbols = zlib.compress(symbols_json, level=9)
        buffer.extend(struct.pack('<I', len(compressed_symbols)))
        buffer.extend(compressed_symbols)
        
        # Auxiliary data
        aux_count = len(payload.auxiliary_data)
        buffer.extend(struct.pack('<I', aux_count))
        
        for key, value in payload.auxiliary_data.items():
            key_bytes = key.encode('utf-8')
            buffer.extend(struct.pack('<I', len(key_bytes)))
            buffer.extend(key_bytes)
            buffer.extend(struct.pack('<I', len(value)))
            buffer.extend(value)
        
        return bytes(buffer)
    
    @staticmethod
    def _deserialize_payload(data: bytes) -> SemanticPayload:
        """Deserialize semantic payload from binary format."""
        offset = 0
        
        # Entropy seed
        seed_length = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        entropy_seed = data[offset:offset+seed_length]
        offset += seed_length
        
        # Symbols
        symbols_length = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        compressed_symbols = data[offset:offset+symbols_length]
        symbols_json = zlib.decompress(compressed_symbols).decode('utf-8')
        symbols_list = json.loads(symbols_json)
        symbols = [SemanticInstruction(**s) for s in symbols_list]
        offset += symbols_length
        
        # Auxiliary data
        aux_count = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        auxiliary_data = {}
        for _ in range(aux_count):
            key_length = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            key = data[offset:offset+key_length].decode('utf-8')
            offset += key_length
            
            value_length = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            value = data[offset:offset+value_length]
            offset += value_length
            
            auxiliary_data[key] = value
        
        return SemanticPayload(entropy_seed, symbols, auxiliary_data)

# ============================================================================
# Protocol Validation
# ============================================================================

class ProtocolValidator:
    """Validates CIP-SC protocol compliance."""
    
    @staticmethod
    def validate_compressed_data(compressed: CompressedData) -> Tuple[bool, List[str]]:
        """
        Validate compressed data for protocol compliance.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate header
        if not compressed.header.protocol_version:
            errors.append("Missing protocol version")
        
        if compressed.header.protocol_version != PROTOCOL_VERSION:
            errors.append(f"Unsupported protocol version: {compressed.header.protocol_version}")
        
        if compressed.header.original_size <= 0:
            errors.append("Invalid original size")
        
        if not compressed.header.compressor_id:
            errors.append("Missing compressor ID")
        
        # Validate payload
        if not compressed.payload.entropy_seed:
            errors.append("Missing entropy seed")
        
        if len(compressed.payload.entropy_seed) < 16:
            errors.append("Entropy seed too short (minimum 16 bytes)")
        
        if not compressed.payload.symbols:
            errors.append("No semantic symbols found")
        
        if len(compressed.payload.symbols) > MAX_SYMBOL_COUNT:
            errors.append(f"Too many symbols: {len(compressed.payload.symbols)} > {MAX_SYMBOL_COUNT}")
        
        # Validate symbols
        for i, symbol in enumerate(compressed.payload.symbols):
            if not symbol.type:
                errors.append(f"Symbol {i}: Missing type")
            
            if not symbol.parameters:
                errors.append(f"Symbol {i}: Missing parameters")
            
            if not (0.0 <= symbol.confidence <= 1.0):
                errors.append(f"Symbol {i}: Invalid confidence: {symbol.confidence}")
        
        # Validate validation data
        if not compressed.validation.content_hash:
            errors.append("Missing content hash")
        
        if len(compressed.validation.content_hash) != 128:  # SHA3-512 hex
            errors.append("Invalid content hash length")
        
        if not compressed.validation.symbol_hash:
            errors.append("Missing symbol hash")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_protocol_version(version: str) -> bool:
        """Check if protocol version is supported."""
        supported_versions = ["1.0"]
        return version in supported_versions
    
    @staticmethod
    def validate_compression_ratio(compressed: CompressedData, minimum_ratio: float = 10.0) -> bool:
        """Validate that compression ratio meets minimum requirements."""
        ratio = compressed.get_compression_ratio()
        return ratio >= minimum_ratio

# ============================================================================
# File Format Handlers
# ============================================================================

class CIPSCFile:
    """Handle CIP-SC file operations."""
    
    @staticmethod
    def save(compressed_data: CompressedData, filepath: str) -> None:
        """Save compressed data to file."""
        serialized = CIPSCProtocol.serialize(compressed_data)
        
        with open(filepath, 'wb') as f:
            f.write(serialized)
    
    @staticmethod
    def load(filepath: str) -> CompressedData:
        """Load compressed data from file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        
        return CIPSCProtocol.deserialize(data)
    
    @staticmethod
    def get_file_info(filepath: str) -> Dict[str, Any]:
        """Get information about CIP-SC file without full loading."""
        with open(filepath, 'rb') as f:
            # Read enough data to get header
            header_data = f.read(1024)  # Should be enough for most headers
        
        if len(header_data) < 10:
            raise ValueError("File too short")
        
        # Check magic and version
        magic = header_data[:5]
        if magic != CIPSC_MAGIC:
            raise ValueError("Not a CIP-SC file")
        
        version = header_data[5]
        if version != 0x10:
            raise ValueError(f"Unsupported version: {version}")
        
        # Parse header length and extract header
        header_length = struct.unpack('<I', header_data[6:10])[0]
        
        if len(header_data) < 10 + header_length:
            # Need to read more
            with open(filepath, 'rb') as f:
                f.seek(10)
                header_json_data = f.read(header_length)
        else:
            header_json_data = header_data[10:10+header_length]
        
        header_dict = json.loads(header_json_data.decode('utf-8'))
        
        return {
            'protocol_version': header_dict.get('protocol_version'),
            'compressor_id': header_dict.get('compressor_id'),
            'compression_type': header_dict.get('compression_type'),
            'original_size': header_dict.get('original_size'),
            'compressed_size': header_dict.get('compressed_size'),
            'timestamp': header_dict.get('timestamp'),
            'compression_ratio': header_dict.get('original_size', 0) / max(header_dict.get('compressed_size', 1), 1)
        }

# ============================================================================
# Protocol Utilities
# ============================================================================

class ProtocolUtils:
    """Utility functions for CIP-SC protocol operations."""
    
    @staticmethod
    def estimate_compressed_size(symbols: List[SemanticInstruction], 
                               auxiliary_data: Dict[str, bytes] = None) -> int:
        """Estimate compressed size without full serialization."""
        # Base overhead
        overhead = 100  # Magic, version, lengths, checksums
        
        # Header (estimated)
        header_size = 200
        
        # Payload
        symbols_json = json.dumps([asdict(s) for s in symbols], sort_keys=True)
        compressed_symbols_size = len(zlib.compress(symbols_json.encode('utf-8'), level=9))
        
        seed_size = 32  # Standard seed size
        
        aux_size = 0
        if auxiliary_data:
            aux_size = sum(len(k.encode('utf-8')) + len(v) + 8 for k, v in auxiliary_data.items())
        
        payload_size = 4 + seed_size + 4 + compressed_symbols_size + 4 + aux_size
        
        # Validation
        validation_size = 200
        
        return overhead + header_size + payload_size + validation_size
    
    @staticmethod
    def convert_to_json(compressed_data: CompressedData) -> str:
        """Convert compressed data to JSON format (for debugging/inspection)."""
        data_dict = {
            'header': asdict(compressed_data.header),
            'payload': {
                'entropy_seed': compressed_data.payload.entropy_seed.hex(),
                'symbols': [asdict(s) for s in compressed_data.payload.symbols],
                'auxiliary_data': {k: v.hex() for k, v in compressed_data.payload.auxiliary_data.items()}
            },
            'validation': asdict(compressed_data.validation)
        }
        
        return json.dumps(data_dict, indent=2, sort_keys=True)
    
    @staticmethod
    def from_json(json_str: str) -> CompressedData:
        """Create compressed data from JSON format."""
        data_dict = json.loads(json_str)
        
        # Reconstruct header
        header = CompressionHeader(**data_dict['header'])
        
        # Reconstruct payload
        payload_data = data_dict['payload']
        entropy_seed = bytes.fromhex(payload_data['entropy_seed'])
        symbols = [SemanticInstruction(**s) for s in payload_data['symbols']]
        auxiliary_data = {k: bytes.fromhex(v) for k, v in payload_data['auxiliary_data'].items()}
        
        payload = SemanticPayload(entropy_seed, symbols, auxiliary_data)
        
        # Reconstruct validation
        validation = ValidationData(**data_dict['validation'])
        
        return CompressedData(header, payload, validation)

# ============================================================================
# Example Usage
# ============================================================================

def demo_protocol_operations():
    """Demonstrate protocol operations."""
    print("🔧 CIP-SC Protocol Demo")
    print("=" * 30)
    
    # This would typically come from the compression engine
    from .compression_engine import generate_test_image, CompressionEngine
    
    # Generate and compress test data
    test_data = generate_test_image(64)  # Smaller for demo
    engine = CompressionEngine()
    
    print(f"📊 Original data size: {len(test_data):,} bytes")
    
    # Compress
    result = engine.compress(test_data, 'geometric')
    if not result.success:
        print(f"❌ Compression failed: {result.error_message}")
        return
    
    compressed = result.compressed_data
    print(f"✅ Compressed with {len(compressed.payload.symbols)} symbols")
    
    # Validate protocol compliance
    is_valid, errors = ProtocolValidator.validate_compressed_data(compressed)
    print(f"🔍 Protocol validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
    if errors:
        for error in errors:
            print(f"   ⚠️  {error}")
    
    # Serialize to binary format
    print(f"\n💾 Serializing to binary format...")
    binary_data = CIPSCProtocol.serialize(compressed)
    print(f"   Binary size: {len(binary_data):,} bytes")
    
    # Deserialize back
    print(f"📤 Deserializing from binary...")
    reconstructed_compressed = CIPSCProtocol.deserialize(binary_data)
    print(f"   ✅ Successfully reconstructed metadata")
    
    # Verify integrity
    original_hash = compressed.validation.content_hash
    reconstructed_hash = reconstructed_compressed.validation.content_hash
    integrity_ok = original_hash == reconstructed_hash
    print(f"🔐 Integrity check: {'✅ PASS' if integrity_ok else '❌ FAIL'}")
    
    # Show file operations
    print(f"\n📁 File operations demo...")
    test_file = "test_compression.cipsc"
    
    try:
        CIPSCFile.save(compressed, test_file)
        print(f"   💾 Saved to {test_file}")
        
        file_info = CIPSCFile.get_file_info(test_file)
        print(f"   📋 File info: {file_info['compression_ratio']:.1f}:1 ratio")
        
        loaded_compressed = CIPSCFile.load(test_file)
        print(f"   📤 Loaded successfully")
        
        # Cleanup
        import os
        os.remove(test_file)
        print(f"   🗑️  Cleaned up test file")
        
    except Exception as e:
        print(f"   ❌ File operations failed: {e}")
    
    print(f"\n🎉 Protocol demo complete!")

if __name__ == "__main__":
    demo_protocol_operations()
