# entropy_information_polarity_field.py

import torch
import numpy as np
import matplotlib.pyplot as plt

# Historical metrics
entropy_history = []
symbolic_history = []
lineage_history = []
coherence_history = []
ancestry_history = []
jaccard_ancestry_vs_symbolic = []
lineage_entropy_history = []


def initialize_field(resolution, seed=314159, device="cuda", mode="blackhole"):
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

    return entropy_field, symbolic_field, lineage_trace, recursion_memory, ancestry_field


def compute_entropy_gradient(entropy_field):
    gradients = torch.gradient(entropy_field, dim=(0, 1, 2))
    return sum(torch.abs(g) for g in gradients), gradients


def compute_symbolic_recursion(symbolic_field, recursion_memory):
    updated_memory = recursion_memory + symbolic_field
    return symbolic_field * 0.5 + updated_memory * 0.01, updated_memory


def collapse_decision(entropy_grad, symbolic_force, threshold=0.1):
    overload = (symbolic_force > 0.8).float()
    field_potential = -entropy_grad + symbolic_force + overload * 1.0
    return (field_potential > threshold).float(), field_potential


def compute_curl(gradients):
    dx, dy, dz = gradients
    curl_x = torch.gradient(dz, dim=1)[0] - torch.gradient(dy, dim=2)[0]
    curl_y = torch.gradient(dx, dim=2)[0] - torch.gradient(dz, dim=0)[0]
    curl_z = torch.gradient(dy, dim=0)[0] - torch.gradient(dx, dim=1)[0]
    curl_mag = torch.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
    return curl_mag


def simulate_step(entropy_field, symbolic_field, lineage_trace, recursion_memory, ancestry_field, threshold, mode):
    resolution = entropy_field.shape[0]
    entropy_grad, gradients = compute_entropy_gradient(entropy_field)
    symbolic_force, recursion_memory = compute_symbolic_recursion(symbolic_field, recursion_memory)
    entropy_grad = entropy_grad + (symbolic_field > 0.9).float() * -0.3
    collapse, field_potential = collapse_decision(entropy_grad, symbolic_force, threshold)

    if mode == "blackhole":
        symbolic_field = symbolic_field - collapse * 0.2
        entropy_field = entropy_field + collapse * 0.3
    elif mode == "whitehole":
        symbolic_field = symbolic_field + collapse * 0.2
        entropy_field = entropy_field - collapse * 0.2
    lineage_trace += collapse

    # Ancestry propagation enhancement
    kernel = torch.tensor([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,0,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]], device=entropy_field.device).unsqueeze(0).unsqueeze(0).float()
    ancestry_float = ancestry_field.float().unsqueeze(0).unsqueeze(0)
    neighbor_sum = torch.nn.functional.conv3d(ancestry_float, kernel, padding=1)[0,0]
    inherit_mask = (collapse > 0.5)
    ancestry_field[inherit_mask] = (ancestry_field[inherit_mask] + neighbor_sum[inherit_mask]).round().int() % (resolution ** 2)

    curl = compute_curl(gradients)
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

    return entropy_field.clamp(0, 1), symbolic_field.clamp(0, 1), lineage_trace, recursion_memory, ancestry_field, curl, gradients


def visualize_all(entropy_field, symbolic_field, lineage_trace, curl, ancestry_field, gradients, slice_idx=None):
    if slice_idx is None:
        slice_idx = entropy_field.shape[0] // 2

    grad_memory_components = torch.gradient(gradients[0], dim=(0, 1, 2))
    grad_memory = sum(torch.abs(g) for g in grad_memory_components)
    lineage_mask = (lineage_trace > 0)
    fig, axs = plt.subplots(2, 7, figsize=(42, 8))

    fields = [entropy_field, symbolic_field, lineage_trace, curl, ancestry_field, grad_memory, lineage_mask.float() * entropy_field]
    titles = ["Entropy Field", "Symbolic Field", "Lineage Trace", "Torque Field (Curl)", "Ancestry Field", "∇Memory", "Lineage Entropy"]

    for ax, tensor, title in zip(axs[0], fields, titles):
        tensor_cpu = tensor[slice_idx].detach().cpu().numpy()
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
    for step in range(steps):
        entropy_field, symbolic_field, lineage_trace, recursion_memory, ancestry_field, curl, gradients = simulate_step(
            entropy_field, symbolic_field, lineage_trace, recursion_memory, ancestry_field, threshold, mode)
    return entropy_field, symbolic_field, lineage_trace, curl, ancestry_field, gradients


if __name__ == "__main__":
    for mode in ["blackhole", "whitehole"]:
        print(f"\nRunning mode: {mode}")
        entropy_field, symbolic_field, lineage_trace, curl, ancestry_field, gradients = run_simulation(mode=mode)
        visualize_all(entropy_field, symbolic_field, lineage_trace, curl, ancestry_field, gradients)
