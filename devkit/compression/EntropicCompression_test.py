import numpy as np
import zlib
import bz2
import lzma
import time
import struct

# ---------- Utility Functions ----------
def compute_entropy(data):
    if not data:
        return 0
    vec = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(vec, minlength=256).astype(np.float64)
    probs = counts / len(vec)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def balance_entropy(data, key):
    np.random.seed(key)
    perm = np.random.permutation(256)
    return bytes(perm[b] for b in data)

def unbalance_entropy(data, key):
    np.random.seed(key)
    perm = np.random.permutation(256)
    reverse_perm = np.argsort(perm)
    return bytes(reverse_perm[b] for b in data)

def rle_encode_metadata(flags):
    encoded = bytearray()
    prev = flags[0]
    count = 1
    for current in flags[1:]:
        if current == prev and count < 255:
            count += 1
        else:
            encoded.extend([prev, count])
            prev = current
            count = 1
    encoded.extend([prev, count])
    return bytes(encoded)

def rle_decode_metadata(encoded):
    total = sum(encoded[i+1] for i in range(0, len(encoded), 2))
    result = np.empty(total, dtype=np.uint8)
    idx = 0
    for i in range(0, len(encoded), 2):
        value = encoded[i]
        count = encoded[i+1]
        result[idx:idx+count] = value
        idx += count
    return result.tolist()

def fixed_chunk_entropy_map(data, chunk_size=256):
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

def parallel_balance_chunk(chunk, key, entropy_threshold=7.5):
    entropy = compute_entropy(chunk)
    if entropy > entropy_threshold:
        return balance_entropy(chunk, key), 1
    return chunk, 0

# ---------- Selective Entropy Compression (Final Version) ----------
def selective_entropy_compress(data, key=42, chunk_size=256, entropy_threshold=7.5):
    chunks = fixed_chunk_entropy_map(data, chunk_size)
    flags = []
    transformed_chunks = []

    for chunk in chunks:
        result_chunk, flag = parallel_balance_chunk(chunk, key, entropy_threshold)
        flags.append(flag)
        transformed_chunks.append(result_chunk)

    chunk_lengths = [len(c) for c in transformed_chunks]
    encoded_flags = rle_encode_metadata(flags)

    meta = bytearray()
    meta.extend(struct.pack('>H', len(chunk_lengths)))
    meta.extend(struct.pack('>H', len(encoded_flags)))
    for length in chunk_lengths:
        meta.extend(struct.pack('>H', length))
    meta.extend(encoded_flags)

    compressed_data = zlib.compress(b''.join(transformed_chunks))
    return bytes(meta) + b'||' + compressed_data

def selective_entropy_decompress(packed_data, key=42):
    meta_split = packed_data.split(b'||', 1)
    metadata = meta_split[0]
    compressed_data = meta_split[1]

    size_count = struct.unpack('>H', metadata[:2])[0]
    flag_len = struct.unpack('>H', metadata[2:4])[0]
    sizes = [struct.unpack('>H', metadata[i:i+2])[0] for i in range(4, 4 + size_count * 2)]
    flags_start = 4 + size_count * 2
    flags_encoded = metadata[flags_start:flags_start + flag_len]
    flags = rle_decode_metadata(flags_encoded)

    if len(flags) > len(sizes):
        flags = flags[:len(sizes)]
    elif len(flags) < len(sizes):
        flags += [0] * (len(sizes) - len(flags))

    decompressed = zlib.decompress(compressed_data)
    chunks = []
    cursor = 0
    for length, flag in zip(sizes, flags):
        chunk = decompressed[cursor:cursor+length]
        if flag == 1:
            chunk = unbalance_entropy(chunk, key)
        chunks.append(chunk)
        cursor += length

    return b''.join(chunks)

# ---------- Benchmark Functions ----------
def benchmark_compression(data, compress_func, decompress_func, label, iterations=100):
    start = time.perf_counter()
    compressed = compress_func(data)
    compression_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        result = decompress_func(compressed)
    decompression_time = (time.perf_counter() - start) / iterations

    entropy_start = time.perf_counter()
    compressed_entropy = compute_entropy(compressed)
    entropy_time = time.perf_counter() - entropy_start

    assert result == data, f"Data integrity failed for {label}"

    return {
        "Label": label,
        "Compressed Size": len(compressed),
        "Compressed Entropy": round(compressed_entropy, 5),
        "Compression Time (s)": round(compression_time, 6),
        "Decompression Time (s)": round(decompression_time, 8),
        "Entropy Calc Time (s)": round(entropy_time, 6)
    }

# ---------- Built-in Compression Wrappers ----------
def gzip_compress(data): return zlib.compress(data)
def gzip_decompress(data): return zlib.decompress(data)

def bz2_compress(data): return bz2.compress(data)
def bz2_decompress(data): return bz2.decompress(data)

def lzma_compress(data): return lzma.compress(data)
def lzma_decompress(data): return lzma.decompress(data)

# ---------- Main Test Routine ----------
if __name__ == '__main__':
    test_data = np.random.bytes(10_000_000)
    key = 42

    print("Original Size:", len(test_data))
    print("Original Entropy:", round(compute_entropy(test_data), 5))

    results = [
        benchmark_compression(test_data, gzip_compress, gzip_decompress, "GZIP"),
        benchmark_compression(test_data, bz2_compress, bz2_decompress, "BZ2"),
        benchmark_compression(test_data, lzma_compress, lzma_decompress, "LZMA"),
        benchmark_compression(
            test_data,
            lambda d: selective_entropy_compress(d, key),
            lambda c: selective_entropy_decompress(c, key),
            "Selective Entropy Compression"
        )
    ]

    for r in results:
        print(f"\n--- {r['Label']} ---")
        for k, v in r.items():
            if k != "Label":
                print(f"{k}: {v}")
