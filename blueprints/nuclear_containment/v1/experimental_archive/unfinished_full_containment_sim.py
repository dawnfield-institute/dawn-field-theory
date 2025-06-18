
import numpy as np
import matplotlib.pyplot as plt

def smooth_field_numpy(field):
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(field, 1, mode='edge')
    smoothed = np.zeros_like(field)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            region = padded[i:i+3, j:j+3]
            smoothed[i, j] = np.sum(region * kernel)
    return smoothed

def simulate(level='full'):
    np.random.seed(42)
    grid_size = 150
    dx = 0.01
    dt = 1e-6
    t_steps = 400
    c = 340.0
    damping = 0.002 if level != 'none' else 0.0
    gain = 2.5
    containment_radius_m = 0.25
    noise_level = 0.01 if level != 'none' else 0.05
    shell_radius_m = 0.5
    shell_reflectivity = 0.9 if level in ['full', 'shell_only'] else 0.0
    failure_threshold = 2e5
    max_output = 20000.0
    source_energy = 1e5
    source_duration = 20

    field = np.zeros((t_steps, grid_size, grid_size))
    center = grid_size // 2
    yy, xx = np.meshgrid(np.linspace(0, grid_size * dx, grid_size), np.linspace(0, grid_size * dx, grid_size), indexing='ij')
    dist = np.sqrt((xx - xx[center, center])**2 + (yy - yy[center, center])**2)
    containment_mask = dist <= containment_radius_m
    shell_mask = (dist >= shell_radius_m - 0.01) & (dist <= shell_radius_m + 0.01)
    outer_mask = dist > shell_radius_m

    print(f"\n=== Running: {level.upper()} containment ===")
    for t in range(2, t_steps):
        if t % 50 == 0:
            print(f"[{level}] Step {t}/{t_steps}")

        laplacian = (
            np.roll(field[t - 1], 1, axis=0) + np.roll(field[t - 1], -1, axis=0) +
            np.roll(field[t - 1], 1, axis=1) + np.roll(field[t - 1], -1, axis=1) -
            4 * field[t - 1]
        )
        wave = (
            2 * field[t - 1] - field[t - 2] +
            (c * dt / dx)**2 * laplacian -
            damping * field[t - 1]
        )

        if t < source_duration:
            field[t - 1, center, center] += source_energy
            field[t - 2, center, center] += source_energy

        n_energy = np.zeros_like(field[t])
        if level == 'full':
            sensed = field[t - 1]
            past = field[t - 2]
            velocity = sensed - past
            future_est = sensed + velocity
            suppression_field = smooth_field_numpy(future_est)

            grad_y, grad_x = np.gradient(np.abs(sensed))
            entropy_gradient = np.abs(grad_y) + np.abs(grad_x)
            entropy_modulation = 1 - entropy_gradient / np.max(entropy_gradient + 1e-5)
            envelope = suppression_field * entropy_modulation
            nullifier = -gain * envelope
            nullifier = np.clip(nullifier, -max_output, max_output)
            n_energy[containment_mask] = nullifier[containment_mask]

        if shell_reflectivity > 0:
            wave[shell_mask] *= shell_reflectivity
            shell_energy = np.abs(field[t - 1][shell_mask])
            if np.any(shell_energy > failure_threshold):
                wave[shell_mask] *= 0

        noise = np.random.normal(0, noise_level, size=(grid_size, grid_size))
        field[t] = wave + n_energy + noise

    final_energy = np.abs(field[-1])
    leakage = np.sum(final_energy[outer_mask])
    print(f"[{level}] Final leakage: {leakage:.2e} J/m³")

    plt.figure(figsize=(7, 6))
    im = plt.imshow(final_energy, cmap='hot', origin='lower', extent=[0, 1.5, 0, 1.5])
    plt.title(f"{level.capitalize()} Containment\nLeakage: {leakage:.2e} J/m³")
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.colorbar(im, label='Final Field Energy (J/m³)')
    plt.tight_layout()
    fname = f"{level}_containment_result.png"
    plt.savefig(fname)
    print(f"[{level}] Image saved to {fname}")
    plt.close()

if __name__ == "__main__":
    for mode in ['none', 'shell_only', 'full']:
        simulate(mode)
