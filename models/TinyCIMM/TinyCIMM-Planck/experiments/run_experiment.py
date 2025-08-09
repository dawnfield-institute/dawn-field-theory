import torch
import torch.nn as nn
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.decomposition import PCA
from tinycimm_planck import TinyCIMMPlanck, FractalQBEController, EntropyMonitorLite, compute_coherence
import matplotlib.pyplot as plt

IMG_DIR = "experiment_images"
os.makedirs(IMG_DIR, exist_ok=True)

# Placeholder for signal generator
# Replace with actual import if available
def get_signal(signal_type, steps, seed=42):
    torch.manual_seed(seed)
    x = torch.linspace(-2, 2, steps).unsqueeze(1)
    if signal_type == "chaotic_sine":
        freq = 2 + torch.sin(5 * x) + 0.5 * torch.randn_like(x)
        amp = 1 + 0.3 * torch.randn_like(x)
        y = amp * torch.sin(freq * x + 2 * torch.sin(3 * x))
    elif signal_type == "noisy_sine":
        y = torch.sin(2 * x) + 0.3 * torch.randn_like(x)
    elif signal_type == "freq_mod_sine":
        freq = 2 + 0.5 * torch.sin(3 * x)
        y = torch.sin(freq * x)
    elif signal_type == "amp_mod_sine":
        amp = 1 + 0.5 * torch.sin(2 * x)
        y = amp * torch.sin(2 * x)
    else:  # clean_sine
        y = torch.sin(2 * x)
    return x, y

def compute_loss(yhat, y_true):
    return torch.mean((yhat - y_true) ** 2)

def compute_entropy(model):
    return model.log_entropy()

def save_logs(logs, signal):
    df = pd.DataFrame(logs)
    os.makedirs("experiment_logs", exist_ok=True)
    df.to_csv(f"experiment_logs/tinycimm_planck_{signal}_log.csv", index=False)

def box_count(Z, k):
    S = Z.shape
    count = 0
    for i in range(0, S[0], k):
        for j in range(0, S[1], k):
            if torch.any(Z[i:i+k, j:j+k]):
                count += 1
    return count

def fractal_dim(weights):
    if isinstance(weights, torch.Tensor):
        W = weights.detach().abs() > 1e-5
    else:
        W = torch.tensor(weights).abs() > 1e-5
    if W.ndim != 2 or min(W.shape) < 4:
        return float('nan')
    if not torch.any(W):
        return float('nan')
    min_size = min(W.shape) // 2 + 1
    sizes = torch.arange(2, min_size)
    counts = []
    for size in sizes:
        bc = box_count(W, int(size.item()))
        if size < W.shape[0] and size < W.shape[1] and bc > 0:
            counts.append(bc)
    if len(counts) > 1:
        sizes_log = torch.log(sizes[:len(counts)].float())
        counts_log = torch.log(torch.tensor(counts, dtype=torch.float))
        coeffs = torch.linalg.lstsq(sizes_log.unsqueeze(1), counts_log).solution
        return -coeffs[0].item()
    else:
        return float('nan')

def conditional_smooth(pred, curvature, window=3, threshold=0.1):
    pred_flat = pred.detach().flatten()
    smooth_mask = (curvature.abs() < threshold)
    smoothed = pred_flat.clone()
    for i in range(window, len(pred_flat) - window):
        if smooth_mask[i]:
            smoothed[i] = pred_flat[i - window:i + window + 1].mean()
    return smoothed.unsqueeze(1).to(pred.device)

def run_experiment(model_cls, signal="chaotic_sine", steps=200, seed=42, **kwargs):
    x, y = get_signal(signal, steps, seed)
    device = x.device
    hidden_size = kwargs.pop('hidden_size', 8)  # Default to 8 if not provided
    model = model_cls(input_size=x.shape[1], hidden_size=hidden_size, output_size=1, device=device, **kwargs)
    controller = FractalQBEController()
    entropy_monitor = EntropyMonitorLite(momentum=0.9)
    model.set_entropy_monitor(entropy_monitor)  # Attach entropy monitor to model
    logs = []
    planck_entropies, planck_hsizes, planck_fractals, planck_feedbacks, planck_losses = [], [], [], [], []
    planck_raw_preds, planck_smoothed_preds = [], []
    prev_yhat = None

    # Create signal-specific subfolder
    signal_img_dir = os.path.join(IMG_DIR, signal)
    os.makedirs(signal_img_dir, exist_ok=True)

    for t in range(steps):
        yhat = model(x)
        loss = compute_loss(yhat, y)
        entropy = compute_entropy(model)
        entropy_monitor.update(yhat)
        # --- ONLINE ADAPTATION STEP: update weights ---
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        # Structure update (full logic)
        if hasattr(model, 'grow_and_prune'):
            avg_feedback = float(torch.abs(yhat - y).mean().item())
            model.grow_and_prune(avg_feedback, entropy, float(loss.item()), controller, entropy_monitor=entropy_monitor)
        logs.append({
            'step': t,
            'loss': loss.item(),
            'entropy': entropy,
            'neurons': model.hidden_dim,
        })
        planck_entropies.append(entropy)
        planck_hsizes.append(model.hidden_dim)
        planck_losses.append(loss.item())
        planck_feedbacks.append(float((torch.abs(yhat - y) < 0.3).float().mean().item()))

        # Analyze results using TinyCIMM-Planck's built-in method
        if t % 10 == 0:
            analysis_results = model.analyze_results()
            logs[-1].update(analysis_results)  # Add analysis results to logs

        # Fractal dimension logging
        if t % 10 == 0:
            fd = fractal_dim(model.W)
            planck_fractals.append(fd if not (torch.isnan(torch.tensor(fd)) or torch.isinf(torch.tensor(fd))) else float('nan'))
        # Save weights image
        if t % 50 == 0:
            plt.figure()
            plt.imshow(model.W.detach().cpu().numpy(), aspect='auto', cmap='bwr')
            plt.colorbar()
            plt.title(f'TinyCIMM-Planck Weights at step {t}')
            plt.tight_layout()
            plt.savefig(os.path.join(signal_img_dir, f'planck_weights_step_{t}.png'))
            plt.close()
        # Save fractal diagnostic
        if t % 10 == 0:
            fd = fractal_dim(model.W)
            if not torch.isnan(torch.tensor(fd)):
                plt.figure()
                plt.title(f"Fractal Dim (step {t}) = {fd:.2f}")
                plt.imshow((torch.abs(model.W.detach()) > 1e-5).cpu().numpy(), aspect='auto', cmap='gray_r')
                plt.savefig(os.path.join(signal_img_dir, f'fractal_dim_diag_{t}.png'))
                plt.close()
            plt.figure()
            plt.hist(model.W.detach().cpu().numpy().flatten(), bins=30)
            plt.title(f"Weight Histogram (step {t})")
            plt.savefig(os.path.join(signal_img_dir, f'weight_hist_{t}.png'))
            plt.close()
        # PCA of activations
        if t % 20 == 0 and hasattr(model, 'micro_memory') and len(model.micro_memory) >= 2:
            activations = torch.cat([torch.tensor(a) if not isinstance(a, torch.Tensor) else a for a in model.micro_memory], dim=0)
            if activations.shape[0] > 2 and activations.shape[1] > 1:
                pca = PCA(n_components=2)
                pca_proj = pca.fit_transform(activations.cpu().numpy())
                error_vals = torch.abs((yhat - y).detach().flatten()).cpu().numpy()
                n_points = pca_proj.shape[0]
                if len(error_vals) < n_points:
                    error_vals = list(error_vals) + [0.0] * (n_points - len(error_vals))
                elif len(error_vals) > n_points:
                    error_vals = error_vals[-n_points:]
                plt.figure()
                plt.scatter(pca_proj[:,0], pca_proj[:,1], c=error_vals, cmap='coolwarm', s=10)
                plt.colorbar(label='Error Magnitude')
                plt.title(f'Neuron Activation PCA (step {t}')
                plt.tight_layout()
                plt.savefig(os.path.join(signal_img_dir, f'feedback_geometry_pca_{t}.png'))
                plt.close()
        # Store predictions for diagnostics
        planck_raw_preds.append(yhat.detach().cpu().numpy().flatten())
        # Smoothing (optional, can be expanded)
        curvature = torch.gradient(torch.gradient(y.squeeze())[0])[0].detach()
        yhat_smooth = conditional_smooth(yhat, curvature)
        planck_smoothed_preds.append(yhat_smooth.detach().cpu().numpy().flatten())
    save_logs(logs, signal)
    # Final plots
    plt.figure()
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), label='Ground Truth')
    if len(planck_raw_preds) > 0:
        plt.plot(x.cpu().numpy(), torch.tensor(planck_raw_preds[-1]), label='TinyCIMM-Planck Raw Prediction', linestyle='dashed')
    if len(planck_smoothed_preds) > 0:
        plt.plot(x.cpu().numpy(), torch.tensor(planck_smoothed_preds[-1]), label='TinyCIMM-Planck Smoothed Prediction')
    plt.legend()
    plt.title('Predictions vs. Ground Truth')
    plt.tight_layout()
    plt.savefig(os.path.join(signal_img_dir, f'pred_vs_truth_{signal}.png'))
    plt.close()
    plt.figure()
    plt.plot(planck_entropies, label='TinyCIMM-Planck entropy')
    plt.xlabel('Iteration')
    plt.ylabel('Output entropy')
    plt.legend()
    plt.title('Entropy Evolution')
    plt.tight_layout()
    plt.savefig(os.path.join(signal_img_dir, f'entropy_evolution_{signal}.png'))
    plt.close()
    plt.figure()
    plt.plot(planck_hsizes, label='TinyCIMM-Planck hidden size')
    plt.xlabel('Iteration')
    plt.ylabel('Hidden Layer Size')
    plt.legend()
    plt.title('Hidden Layer Size Evolution')
    plt.tight_layout()
    plt.savefig(os.path.join(signal_img_dir, f'hidden_layer_size_evolution_{signal}.png'))
    plt.close()
    plt.figure()
    cfd_x = torch.arange(0, steps, 10)[:len(planck_fractals)]
    cfd_np = torch.tensor(planck_fractals)
    mask = ~torch.isnan(cfd_np)
    plt.plot(cfd_x[mask].cpu().numpy(), cfd_np[mask].cpu().numpy(), label='TinyCIMMPlanck fractal dim (W)')
    plt.xlabel('Iteration')
    plt.ylabel('Fractal Dimension')
    plt.legend()
    plt.title('Fractal Dimension Evolution')
    plt.tight_layout()
    plt.savefig(os.path.join(signal_img_dir, f'fractal_dim_evolution_{signal}.png'))
    plt.close()
    plt.figure()
    plt.plot(planck_feedbacks, label='TinyCIMM-Planck Feedback (Accuracy)')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy Fraction (<0.3 err)')
    plt.legend()
    plt.title('TinyCIMM-Planck Feedback/Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(signal_img_dir, f'feedback_accuracy_{signal}.png'))
    plt.close()
    plt.figure()
    plt.plot(planck_losses, label='TinyCIMM-Planck MSE Loss')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('TinyCIMM-Planck Raw Loss Evolution')
    plt.tight_layout()
    plt.savefig(os.path.join(signal_img_dir, f'loss_evolution_{signal}.png'))
    plt.close()
def run_all_experiments():
    test_cases = [
        ("clean_sine", {}),
        ("amp_mod_sine", {}),
        ("freq_mod_sine", {
            "hidden_size": 32,
            "micro_memory_size": 20,
            "symbolic_hold_steps": 5,
            "growth_cooldown": 1,
            "prune_margin": 1,
            "hysteresis_steps": 1,
            "min_interval": 1,
            "cooldown_after_adapt": 2
        }),
        ("noisy_sine", {}),
        ("chaotic_sine", {"hidden_size": 24, "micro_memory_size": 10, "symbolic_hold_steps": 20}),
    ]
    for test_name, model_kwargs in test_cases:
        print(f"\n=== Running experiment: {test_name} ===")
        run_experiment(TinyCIMMPlanck, signal=test_name, **model_kwargs)

if __name__ == "__main__":
    run_all_experiments()

