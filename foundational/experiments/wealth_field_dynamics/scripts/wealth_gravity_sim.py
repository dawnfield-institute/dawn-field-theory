# Emergent Wealth Concentration via Informational Tangle
# Direct adaptation of recursive_gravity.py for economic domain
# Same mechanics, different semantics - testing universality of PAC/SEC

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# PARAMETERS (same structure as recursive_gravity.py)
# ==============================================================================

timesteps = 500
dt = 0.1
field_res = 200
field_extent = 10
memory_decay = 0.98  # Recursive memory decay

# Wealth "masses" - initial wealth concentrations
n_agents = 20
np.random.seed(42)

# Initial wealth follows rough power law (some start with more)
initial_wealth = np.random.pareto(2, n_agents) + 1
initial_wealth = initial_wealth / initial_wealth.sum() * 100  # Normalize to 100 total

# Initial positions (random in economic space)
positions = np.random.uniform(-field_extent * 0.8, field_extent * 0.8, (n_agents, 2))

# Initial velocities (small random)
velocities = np.random.randn(n_agents, 2) * 0.1

# Entropy/transaction field
entropy_field = np.zeros((field_res, field_res))

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def to_field_idx(pos):
    """Convert position to field index."""
    scale = field_res / (2 * field_extent)
    x_idx = int((pos[0] + field_extent) * scale)
    y_idx = int((pos[1] + field_extent) * scale)
    return np.clip(x_idx, 0, field_res - 1), np.clip(y_idx, 0, field_res - 1)

# ==============================================================================
# SIMULATION (same mechanics as recursive_gravity)
# ==============================================================================

traces = {i: [positions[i].copy()] for i in range(n_agents)}
wealth_history = [initial_wealth.copy()]
wealth = initial_wealth.copy()

print("=" * 60)
print("WEALTH FIELD DYNAMICS SIMULATION")
print("Testing PAC/SEC universality in economic domain")
print("=" * 60)
print(f"\nInitial wealth distribution:")
print(f"  Top 10%: {np.sum(np.sort(wealth)[-2:]):.1f}%")
print(f"  Bottom 50%: {np.sum(np.sort(wealth)[:10]):.1f}%")
print(f"  Gini coefficient: {1 - 2 * np.sum(np.cumsum(np.sort(wealth))) / (n_agents * np.sum(wealth)) + 1/n_agents:.3f}")

for t in range(timesteps):
    # Recursive memory decay
    entropy_field *= memory_decay
    
    # Calculate pairwise interactions (like gravity, but for wealth)
    for i in range(n_agents):
        force = np.zeros(2)
        
        for j in range(n_agents):
            if i == j:
                continue
                
            delta = positions[j] - positions[i]
            dist = np.linalg.norm(delta)
            direction = delta / (dist + 1e-9)
            
            # Tangle strength - same as recursive_gravity
            # Wealthier agents have stronger "pull"
            tangle_strength = np.exp(-dist) * wealth[j]
            
            # Force toward wealthier agents (like gravitational attraction)
            force += tangle_strength * direction
        
        # Update velocity and position
        velocities[i] += force * dt / (wealth[i] + 1)  # Heavier = slower response
        positions[i] += velocities[i] * dt
        
        # Boundary reflection
        for dim in range(2):
            if abs(positions[i, dim]) > field_extent:
                positions[i, dim] = np.clip(positions[i, dim], -field_extent, field_extent)
                velocities[i, dim] *= -0.5
        
        # Update entropy field
        x_idx, y_idx = to_field_idx(positions[i])
        entropy_field[y_idx, x_idx] += wealth[i] * 0.01
        
        traces[i].append(positions[i].copy())
    
    # Wealth transfer based on proximity (transactions)
    # Closer agents exchange more, net flow toward larger wealth
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 2.0:  # Interaction radius
                # Transaction probability
                transaction = 0.01 * np.exp(-dist) * min(wealth[i], wealth[j])
                
                # Net flow toward wealthier (SEC gradient)
                if wealth[i] > wealth[j]:
                    wealth[i] += transaction * 0.1
                    wealth[j] -= transaction * 0.1
                else:
                    wealth[j] += transaction * 0.1
                    wealth[i] -= transaction * 0.1
    
    # PAC conservation: normalize to maintain total = 100
    wealth = wealth * 100 / wealth.sum()
    wealth_history.append(wealth.copy())

# Convert to arrays
wealth_history = np.array(wealth_history)

# ==============================================================================
# RESULTS
# ==============================================================================

print(f"\nFinal wealth distribution (after {timesteps} steps):")
print(f"  Top 10%: {np.sum(np.sort(wealth)[-2:]):.1f}%")
print(f"  Bottom 50%: {np.sum(np.sort(wealth)[:10]):.1f}%")
print(f"  Gini coefficient: {1 - 2 * np.sum(np.cumsum(np.sort(wealth))) / (n_agents * np.sum(wealth)) + 1/n_agents:.3f}")

# Calculate concentration dynamics
initial_gini = 1 - 2 * np.sum(np.cumsum(np.sort(wealth_history[0]))) / (n_agents * np.sum(wealth_history[0])) + 1/n_agents
final_gini = 1 - 2 * np.sum(np.cumsum(np.sort(wealth_history[-1]))) / (n_agents * np.sum(wealth_history[-1])) + 1/n_agents

print(f"\nConcentration change:")
print(f"  Initial Gini: {initial_gini:.3f}")
print(f"  Final Gini: {final_gini:.3f}")
print(f"  Direction: {'CONCENTRATED (funnel up)' if final_gini > initial_gini else 'DISPERSED (trickle down)'}")

# Extract Ξ from dynamics
# Ξ should appear in the ratio of change rates
gini_series = []
for t in range(len(wealth_history)):
    w = wealth_history[t]
    gini = 1 - 2 * np.sum(np.cumsum(np.sort(w))) / (n_agents * np.sum(w)) + 1/n_agents
    gini_series.append(gini)
gini_series = np.array(gini_series)

# Look for balance point
d_gini = np.gradient(gini_series)
stable_mask = np.abs(d_gini) < 0.001
if np.any(stable_mask):
    stable_gini = np.mean(gini_series[stable_mask])
    print(f"\nStability analysis:")
    print(f"  Stable Gini value: {stable_gini:.3f}")
else:
    print(f"\nNo stable equilibrium reached in {timesteps} steps")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Agent trajectories with entropy field
ax1 = axes[0, 0]
extent = [-field_extent, field_extent, -field_extent, field_extent]
ax1.imshow(entropy_field, extent=extent, origin='lower', cmap='inferno', alpha=0.7)
for i in range(n_agents):
    trace = np.array(traces[i])
    color = plt.cm.viridis(wealth[i] / wealth.max())
    ax1.plot(trace[:, 0], trace[:, 1], color=color, alpha=0.6, linewidth=0.5)
    ax1.scatter(trace[-1, 0], trace[-1, 1], s=wealth[i] * 5, color=color, edgecolor='white')
ax1.set_facecolor("black")
ax1.set_aspect('equal')
ax1.set_title("Wealth Agent Trajectories (size = wealth)")
ax1.set_xlim(-field_extent, field_extent)
ax1.set_ylim(-field_extent, field_extent)

# Plot 2: Wealth distribution evolution
ax2 = axes[0, 1]
for i in range(n_agents):
    ax2.plot(wealth_history[:, i], alpha=0.5)
ax2.set_xlabel("Time")
ax2.set_ylabel("Wealth")
ax2.set_title("Individual Wealth Trajectories")

# Plot 3: Gini coefficient over time
ax3 = axes[1, 0]
ax3.plot(gini_series, 'b-', linewidth=2)
ax3.axhline(y=initial_gini, color='g', linestyle='--', label=f'Initial: {initial_gini:.3f}')
ax3.axhline(y=final_gini, color='r', linestyle='--', label=f'Final: {final_gini:.3f}')
ax3.set_xlabel("Time")
ax3.set_ylabel("Gini Coefficient")
ax3.set_title("Inequality Over Time")
ax3.legend()

# Plot 4: Final wealth distribution (log scale)
ax4 = axes[1, 1]
sorted_wealth = np.sort(wealth)[::-1]
ranks = np.arange(1, n_agents + 1)
ax4.loglog(ranks, sorted_wealth, 'bo-')
ax4.set_xlabel("Rank")
ax4.set_ylabel("Wealth")
ax4.set_title("Rank-Wealth Distribution (log-log)")

# Fit power law
log_ranks = np.log(ranks)
log_wealth = np.log(sorted_wealth)
slope, intercept = np.polyfit(log_ranks, log_wealth, 1)
ax4.loglog(ranks, np.exp(intercept) * ranks ** slope, 'r--', label=f'Power law: α = {-slope:.2f}')
ax4.legend()

plt.tight_layout()
plt.savefig("wealth_gravity_results.png", dpi=150, facecolor='white')
plt.show()

print(f"\nPower law exponent: α = {-slope:.2f}")
print(f"(Typical real-world: α ≈ 1.5-2.5)")
