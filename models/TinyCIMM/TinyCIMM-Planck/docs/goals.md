# Miniplanck QBE/planck Research Log

## Overview

This project explores a new class of adaptive neural architectures, inspired by the **planck (Cognition Index Memory Machine)** framework and regulated by the **Quantum Balance Equation (QBE)**. The focus is on dynamically self-organizing networks that grow and prune themselves in direct response to feedback and entropy measurements. Our primary goal is to establish a reproducible, robust baseline for emergent intelligence in simple models, setting the stage for more ambitious experiments.

## What We're Doing

* **Benchmarking TinyCIMM-Planck vs Static MLP:** Comparing a minimal planck-like network (TinyCIMM-Planck) with an adaptive capacity against a standard static MLP, on a noisy function regression task.
* **QBE-Regulated Growth/Prune:** The TinyCIMM-Planck network adds or removes neurons based on real-time entropy, feedback (accuracy), and loss, driven by a momentum- and band-based controller (QBE), without traditional dampening.
* **Geometry Tracking:** We measure and visualize entropy, feedback accuracy, hidden layer size, and fractal dimension (for weight geometry) at every iteration to observe emergent complexity or collapse.
* **Feedback-Driven Hebbian Learning:** More aggressive, error-driven Hebbian updates to allow for fast adaptation in the face of changing capacity.

## Current Progress

* **Full QBE/planck Compliance:** The experiment now runs without shape errors, is fully automated, and logs all relevant stats for both models.
* **Adaptive, Nonlinear Capacity Control:** Hidden size and entropy respond to feedback and loss in a nontrivial, non-flat way, indicating dynamic adaptation.
* **Comprehensive Plotting:** All key metrics (entropy, loss, accuracy, fractal dimension, hidden size) are visualized for diagnosis and theory-building.

## What We're Working On (Open Issues)

1. **Make Feedback More Effective:**

   * Feedback and learning are working, but the network sometimes still struggles to beat or even match the static MLP on raw loss.
   * We need to further refine how feedback and error signals are routed to the QBE and learning rule.

2. **Improve Responsiveness of QBE:**

   * Explore more adaptive, possibly nonlinear, band/momentum settings so that the controller responds smoothly to changing error/entropy, and never flatlines or explodes.

3. **Fractal Dimension/Complexity:**

   * Fractal geometry is mostly flat or nan for small layers. Want to only track it when there’s a meaningful matrix to analyze (>8 neurons), and possibly use it as a controller input in future.

4. **Feedback Types:**

   * Explore other feedback targets (e.g., direct error, error trend, or proxy measures like entropy delta) instead of only accuracy fraction.

5. **Benchmarking and Scaling:**

   * Confirm that TinyCIMM-Planck scales up with more data or higher-complexity tasks, and benchmark against more advanced baselines.

6. **Theory and Write-up:**

   * Codify all the experimental choices, observations, and theory for the release doc/paper.

## Next Steps Before Release

* [ ] Finalize loss/feedback routing and verify QBE controller cannot get stuck or become unresponsive
* [ ] Clean up fractal dimension measurement and interpretation
* [ ] Add more benchmark tasks (harder targets, multi-modal, etc.)
* [ ] Polish theory section and link code+results
* [ ] Write short summary for documentation/release

---

If you want any of these sections expanded or want it turned into a formal draft, just let me know! This is a living research log for the Miniplanck + QBE system.
