import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import random_noise
from skimage import color
from skimage.transform import resize
from skimage.measure import shannon_entropy, regionprops, label as sk_label
from skimage.filters import threshold_otsu
import os

# -------------------------
# CORE METRICS & ANALYSIS
# -------------------------
def fractal_dimension(Z):
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])
    Z = (Z > np.mean(Z)).astype(int)
    sizes = 2 ** np.arange(int(np.log2(min(Z.shape))), 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts

def patch_entropy(img, patch_size):
    h, w = img.shape
    entropies = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.size > 0:
                entropies.append(shannon_entropy(patch))
    return np.mean(entropies)

def fourier_fractal_slope(img):
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = resize(img, (256, 256), mode='reflect', anti_aliasing=True)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    def radial_profile(data, center=None):
        y, x = np.indices((data.shape))
        if center is None:
            center = np.array([x.max() / 2, y.max() / 2])
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        r = r.astype(np.int32)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile
    radial_prof = radial_profile(magnitude_spectrum)
    x = np.arange(1, len(radial_prof))
    y = radial_prof[1:]
    log_x = np.log(x)
    log_y = np.log(y + 1e-8)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    return slope, log_x, log_y, radial_prof

# -------------------------
# SYMBOLIC TREE RECURSION
# -------------------------
def image_to_patch_tokens(img, patch_size=32, concept_bank=None):
    h, w = img.shape
    tokens = []
    vectors = []
    if concept_bank is None:
        concept_bank = [
            "energy", "entropy", "recursion", "balance", "collapse", "structure",
            "information", "gradient", "node", "field", "memory", "wave", "fractal"
        ]
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            mean_val = np.mean(patch)
            std_val = np.std(patch)
            idx = int((mean_val + std_val) / 2) % len(concept_bank)
            token = f"{concept_bank[idx]}_{int(mean_val)}"
            tokens.append(token)
            vec = np.zeros(len(concept_bank))
            vec[idx] = 1.0 + (std_val / 100.0)
            vectors.append(vec)
    return tokens, np.array(vectors)

def novelty_score(token):
    return int(token.split("_")[-1]) / 255

def recursive_image_tree(tokens, novelty_threshold=0.3):
    tree = {}
    pruned = 0
    prev = None
    for idx, token in enumerate(tokens):
        if novelty_score(token) < novelty_threshold:
            pruned += 1
            continue
        tree[idx] = {"token": token, "parent": prev}
        prev = idx
    return tree, pruned, len(tokens)

# -------------------------
# ADAPTIVE ENTROPY CHUNKING
# -------------------------
def adaptive_entropy_chunks(img_uint8, min_chunk=16, max_chunk=64, entropy_win=16, entropy_thresh=None):
    h, w = img_uint8.shape
    entropy_map = np.zeros((h // entropy_win, w // entropy_win))
    for i in range(0, h, entropy_win):
        for j in range(0, w, entropy_win):
            patch = img_uint8[i:i+entropy_win, j:j+entropy_win]
            if patch.size > 0:
                entropy_map[i//entropy_win, j//entropy_win] = shannon_entropy(patch)
    if entropy_thresh is None:
        entropy_thresh = threshold_otsu(entropy_map)
    high_entropy_mask = entropy_map > entropy_thresh
    labeled = sk_label(high_entropy_mask, connectivity=1)
    regions = regionprops(labeled, intensity_image=entropy_map)
    chunks = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        region_mean = np.mean(entropy_map[minr:maxr, minc:maxc])
        size = min_chunk if region_mean > entropy_thresh else max_chunk
        for i in range(minr*entropy_win, maxr*entropy_win, size):
            for j in range(minc*entropy_win, maxc*entropy_win, size):
                if i + size <= h and j + size <= w:
                    chunk = img_uint8[i:i+size, j:j+size]
                    if chunk.size > 0:
                        chunks.append(((i, j, size), chunk))
    return chunks, entropy_map, entropy_thresh

# -------------------------
# MAIN PIPELINE
# -------------------------
def run_full_provenance_pipeline(image_paths, labels=None):
    patch_sizes = [2, 4, 8, 16, 32, 64]
    all_stats = []
    for idx, path in enumerate(image_paths):
        label = labels[idx] if labels is not None else os.path.basename(path)
        img = imread(path)
        if img.ndim == 3:
            img = color.rgb2gray(img)
        img = resize(img, (256, 256), mode='reflect', anti_aliasing=True)
        img_uint8 = (img * 255).astype(np.uint8)
        # Global metrics
        fd, _, _ = fractal_dimension(img_uint8)
        fourier_slope, _, _, _ = fourier_fractal_slope(img)
        entropy_curve = [patch_entropy(img_uint8, s) for s in patch_sizes]
        # Adaptive chunking
        chunks, entropy_map, entropy_thresh = adaptive_entropy_chunks(img_uint8)
        fd_list = []
        tree_len_list = []
        for (i, j, size), chunk in chunks:
            fdc, _, _ = fractal_dimension(chunk)
            tokens, _ = image_to_patch_tokens(chunk, patch_size=max(size//2, 2))
            tree, pruned, total = recursive_image_tree(tokens, novelty_threshold=0.3)
            fd_list.append(fdc)
            tree_len_list.append(len(tree))
        all_stats.append({
            'label': label,
            'global_fractal_dim': fd,
            'global_fourier_slope': fourier_slope,
            'entropy_curve': entropy_curve,
            'n_chunks': len(chunks),
            'chunk_fractal_dim_mean': np.mean(fd_list),
            'chunk_fractal_dim_std': np.std(fd_list),
            'chunk_tree_len_mean': np.mean(tree_len_list),
            'chunk_tree_len_std': np.std(tree_len_list),
        })
        # Visualization (optional)
        plt.figure(figsize=(8,3))
        plt.subplot(1,2,1)
        plt.imshow(entropy_map, cmap='magma')
        plt.title(f"Entropy map: {label}")
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(img_uint8, cmap='gray')
        for (i, j, size), _ in chunks:
            plt.gca().add_patch(plt.Rectangle((j, i), size, size, edgecolor='red', facecolor='none', linewidth=0.7))
        plt.title(f"Adaptive Chunks: {label} ({len(chunks)})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        # Chunk metric distributions
        plt.figure(figsize=(6,3))
        plt.hist(fd_list, bins=8, alpha=0.6, label='Fractal Dim')
        plt.hist(tree_len_list, bins=8, alpha=0.6, label='Symbolic Tree Len')
        plt.title(f"Chunk Metrics: {label}")
        plt.xlabel("Value")
        plt.ylabel("Chunks")
        plt.legend()
        plt.show()
    return all_stats

# Example usage:
# image_paths = ["/path/to/organic1.png", "/path/to/ai1.png", ...]
# labels = ["organic1", "ai1", ...]
# results = run_full_provenance_pipeline(image_paths, labels=labels)
