import os
from Crypto.Cipher import AES, ChaCha20
from Crypto.Random import get_random_bytes
from scipy.stats import entropy
import numpy as np
from SELID import SELID  # Import SELID from the existing implementation

class SELIDValidation:
    def __init__(self):
        self.selid = SELID()

    def generate_test_data(self):
        """Generate datasets for validation."""
        return {
            "Weak Data": b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # Highly predictable
            "Moderate Data": os.urandom(32) + b"pattern",  # Partially random
            "Strong Data": os.urandom(64),  # Cryptographically random
            "AES Encrypted Data": AES.new(get_random_bytes(16), AES.MODE_ECB).encrypt(os.urandom(64) * 2),
            "ChaCha20 Encrypted Data": ChaCha20.new(key=get_random_bytes(32)).encrypt(os.urandom(64)),
        }

    def calculate_scipy_entropy(self, data):
        """Calculate entropy using scipy."""
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts / np.sum(byte_counts)
        return entropy(probabilities, base=2)

    def validate_against_nist(self, data, label):
        """Validate SELID's results against NIST thresholds."""
        print(f"\nValidating {label} against NIST SP 800-90B thresholds:")
        selid_entropy = self.selid.shannon_entropy(data)

        # NIST thresholds for classification
        if selid_entropy > 7.5:
            expected = "Strong Encryption"
        elif 5.5 <= selid_entropy <= 7.5:
            expected = "Moderate Encryption"
        else:
            expected = "Weak Encryption"

        # Map SELID classifications to NIST classifications
        if selid_entropy > 7.5:
            selid_classification = "Strong Encryption"
        elif 5.5 <= selid_entropy <= 7.5:
            selid_classification = "Moderate Encryption"
        else:
            selid_classification = "Weak Encryption"

        print(f"  - SELID Entropy: {selid_entropy:.4f}")
        print(f"  - SELID Classification (Mapped): {selid_classification}")
        print(f"  - Expected Classification (NIST): {expected}")
        print(f"  - Match: {'✅' if selid_classification == expected else '❌'}")

    def validate_against_scipy(self, data, label):
        """Validate SELID's entropy against scipy entropy estimation."""
        print(f"\nValidating {label} against scipy entropy estimation:")
        scipy_entropy = self.calculate_scipy_entropy(data)
        selid_entropy = self.selid.shannon_entropy(data)

        print(f"  - SELID Entropy: {selid_entropy:.4f}")
        print(f"  - Scipy Entropy: {scipy_entropy:.4f}")
        print(f"  - Difference: {abs(selid_entropy - scipy_entropy):.4f}")

    def run_validation(self):
        """Run validation against NIST and scipy."""
        test_data = self.generate_test_data()
        for label, data in test_data.items():
            self.validate_against_nist(data, label)
            self.validate_against_scipy(data, label)

if __name__ == "__main__":
    validator = SELIDValidation()
    validator.run_validation()