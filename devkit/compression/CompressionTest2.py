import numpy as np
import zlib

def compute_entropy(data):
    """Calculate Shannon entropy for a given byte sequence."""
    if not data:
        return 0
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = byte_counts / len(data)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def balance_entropy(data, key):
    """Applies an entropy-balancing transformation using a key."""
    np.random.seed(key)  # Use key as seed for reproducibility
    perm = np.random.permutation(256)  # Generate a byte permutation
    return bytes(perm[b] for b in data)

def unbalance_entropy(data, key):
    """Reverses the entropy-balancing transformation using the same key."""
    np.random.seed(key)
    perm = np.random.permutation(256)
    reverse_perm = np.argsort(perm)  # Reverse mapping
    return bytes(reverse_perm[b] for b in data)

def entropy_balanced_compression(data, key):
    """Balances entropy, compresses the data, and returns transformed output."""
    balanced_data = balance_entropy(data, key)
    compressed_data = zlib.compress(balanced_data)
    return compressed_data

def entropy_balanced_decompression(compressed_data, key):
    """Decompresses and rebalances entropy to retrieve the original data."""
    decompressed_data = zlib.decompress(compressed_data)
    original_data = unbalance_entropy(decompressed_data, key)
    return original_data

# Generate test data
test_data = b"This is a test message for entropy-balanced encryption and compression." * 10
key = 42  # Example key for entropy transformation

# Apply entropy-balanced compression
compressed_data = entropy_balanced_compression(test_data, key)

# Apply entropy-balanced decompression
decompressed_data = entropy_balanced_decompression(compressed_data, key)

# Validate results
assert decompressed_data == test_data, "Decompressed data does not match original!"
print("original Size:", len(test_data))
# Print results
print("Original Entropy:", compute_entropy(test_data))
print("Compressed Size:", len(compressed_data))
print("Decompressed Entropy:", compute_entropy(decompressed_data))
print("Decompression Successful:", decompressed_data == test_data)
