import numpy as np
import hashlib
import os
import time
import matplotlib.pyplot as plt  # Added for visualization
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError  # For error handling
from Crypto.Cipher import AES, DES3, ChaCha20, Blowfish
from Crypto.Random import get_random_bytes

class SELID:
    def __init__(self, chunk_size=16):
        self.chunk_size = chunk_size
        self.visualize = True  # New flag for enabling/disabling visualization
        self.training_data = []
        self.model = LinearRegression()

    def shannon_entropy(self, data):
        """Calculate Shannon entropy for a given dataset."""
        if not data:
            return 0
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts / np.sum(byte_counts)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def approximate_entropy(self, data, m=2, r=0.2):
        """Calculate Approximate Entropy (ApEn) for a given dataset."""
        if len(data) < m + 1:
            return 0
        data = np.array(data, dtype=np.float64)
        def _phi(m):
            patterns = [data[i:i + m] for i in range(len(data) - m + 1)]
            counts = np.array([np.sum(np.all(np.abs(patterns - pattern) <= r, axis=1)) for pattern in patterns])
            return np.sum(np.log(counts / (len(data) - m + 1))) / (len(data) - m + 1)
        return abs(_phi(m) - _phi(m + 1))

    def compute_entropy_dispersion(self, data):
        if len(data) == 0:
            return 0
        num_chunks = max(1, len(data) // self.chunk_size)
        chunk_entropies = [self.shannon_entropy(data[i * self.chunk_size:(i + 1) * self.chunk_size]) for i in range(num_chunks)]
        entropy_std_dev = np.std(chunk_entropies)
        entropy_range = np.ptp(chunk_entropies)
        cluster_penalty = np.max(np.unique(chunk_entropies, return_counts=True)[1]) / len(chunk_entropies)
        
        # Fourier Transform Analysis (refined)
        fft_magnitude = np.abs(np.fft.fft(chunk_entropies))
        if np.sum(fft_magnitude) == 0:  # Handle division by zero
            fft_score = 0
        else:
            fft_score = np.sum(fft_magnitude[1:]) / np.sum(fft_magnitude)  # Ignore DC component
        
        # Additional statistical measures
        if np.std(chunk_entropies) == 0:  # Handle uniform chunk_entropies
            skewness = 0
            kurtosis = 0
        else:
            skewness = np.mean((chunk_entropies - np.mean(chunk_entropies))**3) / (np.std(chunk_entropies)**3)
            kurtosis = np.mean((chunk_entropies - np.mean(chunk_entropies))**4) / (np.std(chunk_entropies)**4) - 3
        
        return 1 / (1 + entropy_std_dev + entropy_range + cluster_penalty + (1 - fft_score) + abs(skewness) + abs(kurtosis))

    def train_model(self):
        if len(self.training_data) > 0:
            print("Training model with provided data...")
        else:
            print("No training data available. Using default scoring.")

    def predict_security_score(self, entropy, dispersion):
        try:
            return self.model.predict(np.array([[entropy, dispersion]]))[0]
        except NotFittedError:
            return self.compute_final_score(entropy, dispersion)

    def visualize_analysis(self, chunk_entropies, label):
        """Visualize entropy distribution and Fourier Transform results."""
        if not self.visualize:
            return
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(chunk_entropies, label="Chunk Entropies", color='blue')
        plt.title(f"Entropy Distribution - {label}")
        plt.xlabel("Chunk Index")
        plt.ylabel("Entropy")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        fft_magnitude = np.abs(np.fft.fft(chunk_entropies))
        plt.plot(fft_magnitude, label="FFT Magnitude", color='green')
        plt.title(f"Fourier Transform - {label}")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def analyze_data(self, data, label):
        try:
            start_time = time.perf_counter()
            entropy = self.shannon_entropy(data)
            dispersion = self.compute_entropy_dispersion(data)
            classification = self.classify_security(entropy, dispersion)
            elapsed_time = time.perf_counter() - start_time
            print(f"\n{label}:")
            print(f"  - Entropy: {entropy:.4f}")
            print(f"  - Dispersion Score: {dispersion:.4f}")
            print(f"  - Classification: {classification}")
            if "High Security" in classification:
                print("    Explanation: Data exhibits high randomness and low structural patterns.")
            elif "Moderate Security" in classification:
                print("    Explanation: Data shows moderate randomness with some structural patterns.")
            elif "Low Security" in classification:
                print("    Explanation: Data is predictable and may have weak security.")
            print(f"  - Time Taken: {elapsed_time:.10f} seconds\n")
            
            # Visualize analysis
            num_chunks = max(1, len(data) // self.chunk_size)
            chunk_entropies = [self.shannon_entropy(data[i * self.chunk_size:(i + 1) * self.chunk_size]) for i in range(num_chunks)]
            self.visualize_analysis(chunk_entropies, label)
        except Exception as e:
            print(f"Error analyzing data for {label}: {e}")

    def classify_security(self, entropy, dispersion):
        """Classify security based on entropy and dispersion."""
        final_score = self.compute_final_score(entropy, dispersion)
        if final_score >= 0.65:  # Adjusted threshold for High Security
            return "✅ High Security: Strong Encryption"
        elif final_score >= 0.5:
            return "⚠️ Moderate Security: Acceptable Structure"
        else:
            return "⚠️ Low Security: Possible Weakness"

    def compute_final_score(self, entropy, dispersion_score):
        """Compute the final security score with normalized entropy."""
        normalized_entropy = entropy / 8.0  # Normalize entropy to a range of 0 to 1 (assuming max entropy is 8)
        return (0.85 * normalized_entropy) + (0.15 * dispersion_score)

    def test_nist_vectors(self):
        """Run NIST-like validation tests using known weak and random data."""
        known_weak_data = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"  # Highly predictable data
        known_random_data = os.urandom(32)  # Cryptographically random data
        print("\nTesting NIST Known Weak Data:")
        self.analyze_data(known_weak_data, "NIST Weak Test")
        print("\nTesting NIST Known Random Data:")
        self.analyze_data(known_random_data, "NIST Random Test")

# Example usage
def main():
    selid = SELID()
    
    # Test Cases
    high_security_data = os.urandom(64)
    medium_security_data = os.urandom(32) + b"pattern"
    weak_security_data = b"123456abcdef" * 4
    structured_data = b"ABCD" * 16  # Repeating structured pattern
    encrypted_aes_data = AES.new(get_random_bytes(16), AES.MODE_ECB).encrypt(high_security_data * 2)
    encrypted_chacha20_data = ChaCha20.new(key=get_random_bytes(32)).encrypt(high_security_data)
    compressed_data_gzip = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + b"compressed" * 10
    compressed_data_bz2 = b"BZh91AY&SY" + b"compressed" * 10  # Simulated BZ2 header with repeated data

    # Analyze Test Cases
    selid.analyze_data(high_security_data, "High Security Test - Random Data")
    selid.analyze_data(medium_security_data, "Moderate Security Test - Partially Random Data")
    selid.analyze_data(weak_security_data, "Low Security Test - Predictable Data")
    selid.analyze_data(structured_data, "Low Security Test - Structured Pattern")
    selid.analyze_data(encrypted_aes_data, "High Security Test - AES Encrypted Data")
    selid.analyze_data(encrypted_chacha20_data, "High Security Test - ChaCha20 Encrypted Data")
    selid.analyze_data(compressed_data_gzip, "Moderate Security Test - GZIP Compressed Data")
    selid.analyze_data(compressed_data_bz2, "Moderate Security Test - BZ2 Compressed Data")
    
    # Run NIST validation tests
    selid.test_nist_vectors()

if __name__ == "__main__":
    main()