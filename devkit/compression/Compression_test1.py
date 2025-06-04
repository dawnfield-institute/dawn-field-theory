import numpy as np
import zlib
import bz2
import lzma
import time
import threading

def shannon_entropy(data):
    """Calculate Shannon entropy for a given data chunk."""
    if not data:
        return 0
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = byte_counts / len(data)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def chunkwise_entropy_analysis(data, chunk_size=1024):
    """Splits data into chunks and computes entropy for each chunk."""
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    entropy_values = [shannon_entropy(chunk) for chunk in chunks]
    return chunks, entropy_values

def optimized_entropic_compression(data, entropy_threshold=3.5, chunk_size=1024):
    """Applies entropic compression: retains high-entropy chunks, compresses low-entropy chunks."""
    chunks, entropy_values = chunkwise_entropy_analysis(data, chunk_size)
    high_entropy_chunks = []
    compressed_low_entropy_chunks = []
    metadata = []

    for i, (chunk, entropy) in enumerate(zip(chunks, entropy_values)):
        if entropy >= entropy_threshold:
            high_entropy_chunks.append(chunk)
            metadata.append((i, "RAW"))
        else:
            compressed_chunk = zlib.compress(chunk)
            compressed_low_entropy_chunks.append(compressed_chunk)
            metadata.append((i, "COMPRESSED"))
    
    return high_entropy_chunks, compressed_low_entropy_chunks, metadata

def optimized_entropic_decompression(high_entropy_chunks, compressed_low_entropy_chunks, metadata):
    """Reconstructs data from entropically compressed format."""
    reconstructed_data = bytearray()
    compressed_index, high_entropy_index = 0, 0

    for i, chunk_type in metadata:
        if chunk_type == "RAW":
            reconstructed_data.extend(high_entropy_chunks[high_entropy_index])
            high_entropy_index += 1
        else:
            decompressed_chunk = zlib.decompress(compressed_low_entropy_chunks[compressed_index])
            reconstructed_data.extend(decompressed_chunk)
            compressed_index += 1
    
    return bytes(reconstructed_data)

def compute_on_compressed_data_detailed(compressed_data_list, decompression_func, iterations=1000):
    """Measures time taken to process (decompress) compressed data with dynamic iterations."""
    start_time = time.perf_counter()
    for _ in range(iterations):
        for compressed_data in compressed_data_list:
            _ = decompression_func(compressed_data)  # Simulating a computation step
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time / iterations  # Average time per operation

# Generate sample data (reduced size for efficiency)
data_high_entropy = np.random.bytes(16384)  # Simulating high-entropy (random) data
data_low_entropy = b'A' * 16384  # Simulating low-entropy (repetitive) data
test_data = data_high_entropy + data_low_entropy

# Apply entropic compression
high_entropy_chunks, compressed_low_entropy_chunks, metadata = optimized_entropic_compression(test_data)

# Verify decompression
reconstructed_data = optimized_entropic_decompression(high_entropy_chunks, compressed_low_entropy_chunks, metadata)
decompression_valid = reconstructed_data == test_data

def calculate_data_loss(original, reconstructed):
    """Computes the difference between original and reconstructed data."""
    return sum(1 for a, b in zip(original, reconstructed) if a != b) / len(original)

data_loss = calculate_data_loss(test_data, reconstructed_data)

# Run compression timing in a separate thread to prevent blocking
def run_computation_tests():
    global gzip_time, bz2_time, lzma_time, entropic_time
    gzip_time = compute_on_compressed_data_detailed([zlib.compress(test_data)], zlib.decompress)
    bz2_time = compute_on_compressed_data_detailed([bz2.compress(test_data)], bz2.decompress)
    lzma_time = compute_on_compressed_data_detailed([lzma.compress(test_data)], lzma.decompress)
    entropic_time = compute_on_compressed_data_detailed(compressed_low_entropy_chunks, zlib.decompress)

thread = threading.Thread(target=run_computation_tests)
thread.start()
thread.join()

# Print results with 25 decimal places
print(f"Original Size: {len(test_data)} bytes")
print(f"Entropic Compression Size: {sum(len(chunk) for chunk in high_entropy_chunks) + sum(len(chunk) for chunk in compressed_low_entropy_chunks)} bytes")
print(f"GZIP Compression Size: {len(zlib.compress(test_data))} bytes")
print(f"BZ2 Compression Size: {len(bz2.compress(test_data))} bytes")
print(f"LZMA Compression Size: {len(lzma.compress(test_data))} bytes")
print(f"Decompression Valid: {decompression_valid}")
print(f"Data Loss Percentage: {data_loss:.25f}")
print(f"Computation Time (Avg per Operation, s) - GZIP: {gzip_time:.25f}")
print(f"Computation Time (Avg per Operation, s) - BZ2: {bz2_time:.25f}")
print(f"Computation Time (Avg per Operation, s) - LZMA: {lzma_time:.25f}")
print(f"Computation Time (Avg per Operation, s) - Entropic: {entropic_time:.25f}")