# entropy_information_polarity_field.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Historical metrics
entropy_history = []
symbolic_history = []
lineage_history = []
coherence_history = []
ancestry_history = []
jaccard_ancestry_vs_symbolic = []
lineage_entropy_history = []


def compute_local_entropy(field_slice, kernel_size=5):
    pad = kernel_size // 2
    unfolded = torch.nn.functional.unfold(field_slice.unsqueeze(0).unsqueeze(0), kernel_size=(kernel_size, kernel_size), padding=pad)
    probs = unfolded / (unfolded.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-8), dim=1)
    entropy_map = entropy.view(field_slice.shape)
    return entropy_map

def initialize_field(resolution, seed=314159, device="cuda", mode="blackhole"):
    # Ensure asymmetric entropy seeding
    torch.manual_seed(seed)
    entropy_field = torch.rand(resolution, resolution, resolution, device=device)
    symbolic_field = torch.zeros_like(entropy_field)
    lineage_trace = torch.zeros_like(entropy_field)
    recursion_memory = torch.zeros_like(entropy_field)
    ancestry_field = torch.zeros_like(entropy_field, dtype=torch.int32)

    cx = resolution // 2
    r = 6

    if mode == "blackhole":
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx**2 + dy**2 + dz**2 <= r**2:
                        x, y, z = cx + dx, cx + dy, cx + dz
                        if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
                            entropy_field[x, y, z] = 0.0
                            symbolic_field[x, y, z] = 1.0
                            ancestry_field[x, y, z] = (x * resolution + y + z) % (resolution ** 2)

    elif mode == "whitehole":
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx**2 + dy**2 + dz**2 <= r**2:
                        x, y, z = cx + dx, cx + dy, cx + dz
                        if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
                            entropy_field[x, y, z] = 1.0
                            symbolic_field[x, y, z] = 0.1
                            ancestry_field[x, y, z] = (x * resolution + y + z) % (resolution ** 2)

    # Add asymmetric azimuthal perturbation to break radial symmetry
    x = torch.arange(resolution, device=device) - cx
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    phi = torch.atan2(Y.float(), X.float())
    R = torch.sqrt(X.float()**2 + Y.float()**2 + Z.float()**2)

    entropy_field += 0.1 * torch.sin(3 * phi)

    shell_mask = ((R > (resolution * 0.25)) & (R < (resolution * 0.4))).float()
    entropy_field += shell_mask * torch.rand_like(entropy_field) * 0.05
    entropy_field = entropy_field.clamp(0, 1)

    symbolic_field += 0.2 * torch.cos(2 * phi)
    symbolic_field += (entropy_field < 0.3).float() * torch.randn_like(symbolic_field) * 0.01
    symbolic_field = symbolic_field.clamp(0, 1)

    # After entropy initialization
    # Introduce asymmetry: azimuthal sinusoidal entropy bias
    x = torch.linspace(0, np.pi * 2, resolution)
    y = torch.linspace(0, np.pi * 2, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    asymmetry = 0.1 * torch.sin(X + Y)
    entropy_field[:, :, :] += asymmetry.unsqueeze(-1)

    return entropy_field, symbolic_field, lineage_trace, recursion_memory, ancestry_field


def compute_entropy_gradient(entropy_field):
    gradients = torch.gradient(entropy_field, dim=(0, 1, 2))
    return sum(torch.abs(g) for g in gradients), gradients

def collapse_decision(entropy_field, symbolic_field, lineage_trace):
    # Recursion gate: encourage collapse in symbolic-rich, lineage-prioritized zones
    recursive_gate = (symbolic_field + 1e-3) * (1 + lineage_trace.float())
    collapse = (entropy_field * recursive_gate) > 0.5
    return collapse, recursive_gate

def compute_curl(gradients):
    dx, dy, dz = gradients
    curl_x = torch.gradient(dz, dim=1)[0] - torch.gradient(dy, dim=2)[0]
    curl_y = torch.gradient(dx, dim=2)[0] - torch.gradient(dz, dim=0)[0]
    curl_z = torch.gradient(dy, dim=0)[0] - torch.gradient(dx, dim=1)[0]
    curl_mag = torch.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
    return curl_mag


def compute_curl(gradients):
    dx, dy, dz = gradients
    curl_x = torch.gradient(dz, dim=1)[0] - torch.gradient(dy, dim=2)[0]
    curl_y = torch.gradient(dx, dim=2)[0] - torch.gradient(dz, dim=0)[0]
    curl_z = torch.gradient(dy, dim=0)[0] - torch.gradient(dx, dim=1)[0]
    curl_mag = torch.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
    return curl_mag

def compute_symbolic_recursion(symbolic_field, recursion_memory):
    # Anisotropic/asymmetric kernel: bias vertical (Y) direction for wider propagation
    kernel = torch.tensor(
        [[[[[0.01, 0.1, 0.01],
            [0.1, 1.0, 0.1],
            [0.01, 0.1, 0.01]],
           [[0.1, 2.0, 0.1],   # <-- bias vertical (middle Y) direction
            [2.0, 8.0, 2.0],   # <-- stronger vertical propagation
            [0.1, 2.0, 0.1]],
           [[0.01, 0.1, 0.01],
            [0.1, 1.0, 0.1],
            [0.01, 0.1, 0.01]]]]], device=symbolic_field.device)
    kernel = kernel / kernel.sum()
    symbolic_field = symbolic_field.unsqueeze(0).unsqueeze(0)
    diffused = F.conv3d(symbolic_field, kernel, padding=1)
    return diffused.squeeze(), recursion_memory


def simulate_step(entropy_field, symbolic_field, lineage_trace, recursion_memory, ancestry_field, threshold, mode, curl_memory=None):
    resolution = entropy_field.shape[0]
    entropy_grad, gradients = compute_entropy_gradient(entropy_field)
    symbolic_force, recursion_memory = compute_symbolic_recursion(symbolic_field, recursion_memory)

    # Optional: accumulate curl memory for symbolic force
    dx = torch.gradient(symbolic_field, dim=0)[0]
    dy = torch.gradient(symbolic_field, dim=1)[0]
    dz = torch.gradient(symbolic_field, dim=2)[0]
    torque_bias = 0.025 * (dy - dx + dz)
    symbolic_force += torque_bias

    # Entropy-guided symbolic nudge
    entropy_soft_min = (entropy_field < 0.3).float()
    symbolic_force += entropy_soft_min * torch.rand_like(symbolic_force) * 0.05

    # Optional: add curl_memory to symbolic_force if provided
    curl = compute_curl(gradients)
    if curl_memory is not None:
        symbolic_force += 0.1 * curl_memory

    entropy_grad = entropy_grad + (symbolic_field > 0.9).float() * -0.3
    collapse, field_potential = collapse_decision(entropy_grad, symbolic_force, threshold)
    # 🧩 Add:
    collapse_density = collapse.mean().item()
    collapse_std = collapse.std().item()
    collapse_entropy = entropy_field[collapse.bool()].mean().item() if collapse.sum() > 0 else 0.0
    print(f"Step collapse metrics — Density: {collapse_density:.4f}, Std: {collapse_std:.4f}, Entropy@Collapse: {collapse_entropy:.4f}")

    if mode == "blackhole":
        symbolic_field = symbolic_field - collapse * 0.2
        entropy_field = entropy_field + collapse * 0.3
    elif mode == "whitehole":
        symbolic_field = symbolic_field + collapse * 0.2
        entropy_field = entropy_field - collapse * 0.2
    lineage_trace += collapse

    # Enhanced ancestry propagation
    kernel = torch.tensor([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,0,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]], device=entropy_field.device).unsqueeze(0).unsqueeze(0).float()
    ancestry_float = ancestry_field.float().unsqueeze(0).unsqueeze(0)
    neighbor_sum = torch.nn.functional.conv3d(ancestry_float, kernel, padding=1)[0,0]
    inherit_mask = (collapse > 0.5) | ((ancestry_field > 0) & (torch.rand_like(ancestry_field.float()) < 0.07))
    ancestry_field[inherit_mask] = (ancestry_field[inherit_mask] + neighbor_sum[inherit_mask]).round().int() % (resolution ** 2)

    # Decay stagnant ancestry to promote divergence
    decay_mask = (collapse < 0.1) & (torch.rand_like(ancestry_field.float()) < 0.01)
    ancestry_field[decay_mask] = 0

    # Symbolically biased ancestry branching
    branch_mask = (symbolic_field > 0.7) & (torch.rand_like(symbolic_field) < 0.03)
    ancestry_field[branch_mask] = (torch.randint_like(ancestry_field[branch_mask], 0, resolution**2)).int()

    # Smooth ancestry field after collapse step
    ancestry_field = smooth_ancestry_field(ancestry_field)

    # Enhanced metrics
    coherence = torch.mean(torch.abs(field_potential)) / torch.std(symbolic_field + 1e-6)
    ancestry_bin = (ancestry_field > 0).float()
    symbolic_bin = (symbolic_field > 0.1).float()
    intersection = (ancestry_bin * symbolic_bin).sum().item()
    union = ((ancestry_bin + symbolic_bin) > 0).float().sum().item()
    jaccard = intersection / union if union > 0 else 0.0

    lineage_mask = (lineage_trace > 0)
    lineage_entropy = entropy_field[lineage_mask].mean().item() if lineage_mask.any() else 0.0

    entropy_history.append(entropy_field.mean().item())
    symbolic_history.append(symbolic_field.mean().item())
    lineage_history.append(lineage_trace.sum().item())
    coherence_history.append(coherence.item())
    ancestry_history.append((ancestry_field > 0).sum().item())
    jaccard_ancestry_vs_symbolic.append(jaccard)
    lineage_entropy_history.append(lineage_entropy)

    # Return curl for curl_memory accumulation
    return entropy_field.clamp(0, 1), symbolic_field.clamp(0, 1), lineage_trace, recursion_memory, ancestry_field, curl, gradients, collapse

def smooth_ancestry_field(ancestry_field):
    ancestry_field = ancestry_field.float()
    kernel = torch.ones((1, 1, 3, 3, 3), device=ancestry_field.device) / 27.0
    ancestry_field = ancestry_field.unsqueeze(0).unsqueeze(0)
    smoothed = F.conv3d(ancestry_field, kernel, padding=1)
    return smoothed.squeeze()

def visualize_all(entropy_field, symbolic_field, lineage_trace, curl, ancestry_field, gradients, slice_idx=None):
    if slice_idx is None:
        slice_idx = entropy_field.shape[0] // 2

    grad_memory_components = torch.gradient(gradients[0], dim=(0, 1, 2))
    grad_memory = sum(torch.abs(g) for g in grad_memory_components)
    lineage_mask = (lineage_trace > 0)

    local_entropy_map = compute_local_entropy(entropy_field[slice_idx], kernel_size=7)
    lineage_entropy_slice = local_entropy_map * lineage_mask[slice_idx].float()
    lineage_entropy_slice = (lineage_entropy_slice - lineage_entropy_slice.min()) / (lineage_entropy_slice.max() - lineage_entropy_slice.min() + 1e-8)

    fig, axs = plt.subplots(2, 7, figsize=(42, 8))

    ancestry_smoothed = smooth_ancestry_field(ancestry_field)

    fields = [entropy_field, symbolic_field, lineage_trace, curl, ancestry_smoothed, grad_memory, lineage_entropy_slice]
    titles = ["Entropy Field", "Symbolic Field", "Lineage Trace", "Torque Field (Curl)", "Ancestry Field", "∇Memory", "Lineage Entropy"]

    for ax, tensor, title in zip(axs[0], fields, titles):
        tensor_cpu = tensor[slice_idx].detach().cpu().numpy() if tensor.ndim == 3 else tensor.detach().cpu().numpy()
        if tensor_cpu.ndim == 1:
            side = int(np.sqrt(tensor_cpu.shape[0]))
            if side * side == tensor_cpu.shape[0]:
                tensor_cpu = tensor_cpu.reshape((side, side))
            else:
                tensor_cpu = np.expand_dims(tensor_cpu, axis=0)
        im = ax.imshow(tensor_cpu, cmap='inferno')
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.axis('off')

    plots = [entropy_history, symbolic_history, lineage_history, coherence_history, ancestry_history, jaccard_ancestry_vs_symbolic, lineage_entropy_history]
    labels = ["Entropy Mean", "Symbolic Mean", "Lineage Sum", "Collapse Coherence", "Ancestry Active Count", "Jaccard Ancestry∩Symbolic", "Lineage Entropy"]
    plot_titles = ["Entropy Over Time", "Symbolic Over Time", "Lineage Over Time", "Collapse Coherence", "Ancestry Over Time", "Ancestry-Symbolic Jaccard", "Lineage Entropy"]

    for ax, data, label, title in zip(axs[1], plots, labels, plot_titles):
        ax.plot(data, label=label)
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.legend()

    plt.tight_layout()
    plt.show()

def run_simulation(steps=100, resolution=64, threshold=0.1, device="cuda", mode="blackhole"):
    entropy_history.clear()
    symbolic_history.clear()
    lineage_history.clear()
    coherence_history.clear()
    ancestry_history.clear()
    jaccard_ancestry_vs_symbolic.clear()
    lineage_entropy_history.clear()

    entropy_field, symbolic_field, lineage_trace, recursion_memory, ancestry_field = initialize_field(resolution, device=device, mode=mode)
    curl_memory = torch.zeros_like(entropy_field)
    for step in range(steps):
        entropy_field, symbolic_field, lineage_trace, recursion_memory, ancestry_field, curl, gradients, collapse = simulate_step(
            entropy_field, symbolic_field, lineage_trace, recursion_memory, ancestry_field, threshold, mode, curl_memory=curl_memory)
        # Accumulate curl memory for next step (optional enhancement)
        curl_memory = 0.95 * curl_memory + 0.05 * curl
        # In the metrics logging section (after collapse is computed)
        collapse_density = collapse.float().mean().item()
        collapse_std = collapse.float().std().item()
        collapse_entropy = entropy_field[collapse.bool()].mean().item() if collapse.sum() > 0 else 0.0
        print(f"Step {step}: Collapse density={collapse_density:.4f}, std={collapse_std:.4f}, collapse entropy={collapse_entropy:.4f}")
    return entropy_field, symbolic_field, lineage_trace, curl, ancestry_field, gradients


if __name__ == "__main__":
    for mode in ["blackhole", "whitehole"]:
        print(f"\nRunning mode: {mode}")
        entropy_field, symbolic_field, lineage_trace, curl, ancestry_field, gradients = run_simulation(mode=mode)
        visualize_all(entropy_field, symbolic_field, lineage_trace, curl, ancestry_field, gradients)
