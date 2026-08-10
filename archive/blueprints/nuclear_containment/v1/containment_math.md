# Nuclear Containment Math

## Field Definitions

Let the system be described over a 2D spatial grid $G$ with resolution $\Delta x$ and center $C$.

* $T(x, y, t)$: Thermal field (K)
* $I(x, y, t)$: Informational field (dimensionless)
* $M(x, y, t)$: Memory field (cumulative imbalance)
* $\Phi(x, y)$: Phase lattice coefficient (normalized between \[0.5, 1.5])
* $RBF(x, y, t)$: Recursive Balance Field strength

## Uncontained Thermal Expansion

For the baseline case with no containment:
$T_{t+1} = T_t + D \nabla^2 T_t$
where $D$ is a diffusion coefficient approximated via 2D Laplacian finite differences.

## Recursive Balance Field Dynamics

The RBF field is defined by the product of thermal and informational damping:
$RBF = e^{-\gamma_T T} \cdot e^{-\gamma_I I}$

With imbalances measured as:
$\Delta_{imb} = T - I$
$M_{t+1} = M_t + \alpha |\Delta_{imb}|$

The stabilization term applied to both fields:
$\text{stabilized} = \lambda_0 \cdot RBF \cdot \left( \frac{T - I}{1 + \alpha M} \right) \cdot \Phi$

### Informatic Update Rules

* Thermal field:
  $T_{t+1} = T_t \cdot \delta_T - \beta_T \cdot \text{stabilized}$

* Informatic field:
  $I_{t+1} = I_t \cdot \delta_I + \beta_I \cdot \text{stabilized}$

Where:

* $\delta_T, \delta_I$: decay factors
* $\beta_T, \beta_I$: stabilization scaling constants

## Priming Pulse and Phase Flip

Pre-primed containment injects a phase inversion and energy pulse:

* At $t = t_{pulse}$:
  $T(x, y, t) += P \cdot \chi_{r \leq r_0}$
  $\Phi(x, y) \leftarrow 1.0 - \Phi(x, y)$

Where $P$ is the pulse energy, and $\chi$ is the indicator function within radius $r_0$.

## Output Metrics

* **Blast Radius**:
  $R(t) = \sqrt{ \frac{A(T > T_{thresh})}{\pi} }$
* **Energy Contained**: $\int I(x, y, t_{end}) \, dx \, dy$
* **Thermal Residual**: $\int T(x, y, t_{end}) \, dx \, dy$
* **Informational Gain**: $I_{total}(t_{end}) - I_{total}(t=0)$

## Notes

This formulation relies on a semi-discrete, feedback-stabilized informational lattice tuned to thermal energy gradients. Recursive damping ensures containment scales with entropy influx.
