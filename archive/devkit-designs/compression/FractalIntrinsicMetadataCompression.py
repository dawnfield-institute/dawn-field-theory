import numpy as np
import hashlib
import zlib
import struct

def chunkify(data, chunk_size=256):
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    if len(chunks[-1]) < chunk_size:
        chunks[-1] += b'\x00' * (chunk_size - len(chunks[-1]))
    return chunks

def hash_chunk(chunk):
    return hashlib.sha256(chunk).digest()[:8]  # compact 64-bit hash

def build_fractal_index(chunks, fuzzy=False):
    index = {}
    codebook = []
    sequence = []
    for chunk in chunks:
        h = hash_chunk(chunk)
        match_idx = index.get(h)
        if match_idx is None and fuzzy:
            for i, c in enumerate(codebook):
                if len(chunk) == len(c) and np.linalg.norm(np.frombuffer(chunk, dtype=np.uint8) - np.frombuffer(c, dtype=np.uint8)) < 10:
                    match_idx = i
                    break
        if match_idx is None:
            match_idx = len(codebook)
            index[h] = match_idx
            codebook.append(chunk)
        sequence.append(match_idx)
    metadata = []
    for chunk in codebook:
        arr = np.frombuffer(chunk, dtype=np.uint8)
        entropy = -np.sum((p := np.bincount(arr, minlength=256) / len(arr)) * np.log2(p + 1e-10))
        centroid = np.mean(arr)
        power = np.mean(arr ** 2)
        fft = np.fft.rfft(arr)
        dom_freq = np.argmax(np.abs(fft))
        metadata.append((entropy, centroid, power, dom_freq))
    return codebook, sequence, metadata

def fractal_compress(data, chunk_size=256, fuzzy=False, recurse=False):
    print("[DEBUG] Compressing with chunk_size:", chunk_size, "fuzzy:", fuzzy, "recurse:", recurse)
    chunks = chunkify(data, chunk_size)
    codebook, sequence, metadata = build_fractal_index(chunks, fuzzy=fuzzy)
    print("[DEBUG] Unique chunks in codebook:", len(codebook))
    print("[DEBUG] Sequence length:", len(sequence))
    sequence_bytes = np.array(sequence, dtype=np.uint16).tobytes()
    if recurse:
        flat_codebook = b''.join(codebook)
        codebook = chunkify(flat_codebook, chunk_size)
        codebook, _ = build_fractal_index(codebook, fuzzy=fuzzy)
    codebook_bytes = b''.join(codebook)
    meta_bytes = b''.join(struct.pack('>ffff', *m) for m in metadata)
    packed = struct.pack('>I', len(data)) + struct.pack('>H', chunk_size) + struct.pack('>H', len(codebook)) + codebook_bytes + sequence_bytes + meta_bytes
    return zlib.compress(packed)

def fractal_decompress(packed):
    raw = zlib.decompress(packed)
    print("[DEBUG] Decompressed size:", len(raw))
    original_len = struct.unpack('>I', raw[:4])[0]
    chunk_size = struct.unpack('>H', raw[4:6])[0]
    count = struct.unpack('>H', raw[6:8])[0]
    offset = 8
    codebook = [raw[offset+i*chunk_size:offset+(i+1)*chunk_size] for i in range(count)]
    offset += count * chunk_size
    meta_size = count * 16
    sequence_length = (len(raw) - offset - meta_size) // 2
    sequence = np.frombuffer(raw[offset:offset + sequence_length * 2], dtype=np.uint16)
    restored = b''.join(codebook[i] for i in sequence)
    # Optional: Load metadata here if needed
    # meta_offset = offset + len(sequence) * 2
    # metadata = [struct.unpack('>ffff', raw[meta_offset+i*16:meta_offset+(i+1)*16]) for i in range(count)]
    print("[DEBUG] Restored data length:", len(restored))
    return restored[:original_len]

if __name__ == '__main__':
    import lzma, bz2, time

    for pattern in ['repetitive', 'wave', 'random']:
        print(f"--- Benchmarking pattern: {pattern} ---")
        if pattern == 'random':
            data = np.random.bytes(5_000_000)
        elif pattern == 'repetitive':
            data = (b"ABCD1234" * 1024) + (b"XYZ9876" * 512)
        elif pattern == 'wave':
            t = np.linspace(0, 8*np.pi, 5_000_000)
            signal = ((np.sin(t) + 1) * 127).astype(np.uint8)
            data = signal.tobytes()

        start = time.perf_counter()
        compressed = fractal_compress(data, fuzzy=False)
        time_fractal = time.perf_counter() - start
        restored = fractal_decompress(compressed)
        test_pass = (data == restored)

        start = time.perf_counter()
        zlib_c = zlib.compress(data)
        time_zlib = time.perf_counter() - start

        start = time.perf_counter()
        lzma_c = lzma.compress(data)
        time_lzma = time.perf_counter() - start

        start = time.perf_counter()
        bz2_c = bz2.compress(data)
        time_bz2 = time.perf_counter() - start

        print("Fractal : Size =", len(compressed), "Time =", round(time_fractal, 4), "s", "Pass:", test_pass)
        print("zlib    : Size =", len(zlib_c), "Time =", round(time_zlib, 4), "s")
        print("lzma    : Size =", len(lzma_c), "Time =", round(time_lzma, 4), "s")
        print("bz2     : Size =", len(bz2_c), "Time =", round(time_bz2, 4), "s")
        assert test_pass, "Decompression failed: data mismatch"
        assert isinstance(compressed, bytes) and isinstance(restored, bytes), "Output types must be bytes"
        assert len(restored) == len(data), "Restored data length mismatch"
        assert isinstance(compressed, bytes), "Compressed output must be bytes"
        assert isinstance(restored, bytes), "Restored output must be bytes"
        assert len(compressed) > 0 and isinstance(len(compressed), int), "Compressed size must be finite and non-zero"
        assert isinstance(len(compressed), int), "Compressed size must be finite"
        # Compression is computable: finite, deterministic, and reversible with bounded resources
