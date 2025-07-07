from sympy import im
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.decomposition import PCA
from tinycimm import TinyCIMM, FractalQBEController, EntropyMonitorLite, compute_coherence
import matplotlib.pyplot as plt

IMG_DIR = "experiment_images"
os.makedirs(IMG_DIR, exist_ok=True)

# Placeholder for signal generator
# Replace with actual import if available
def get_signal(signal_type, steps, seed=42):
    np.random.seed(seed)
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
    df.to_csv(f"experiment_logs/tinycimm_{signal}_log.csv", index=False)

def box_count(Z, k):
    S = Z.shape
    count = 0
    for i in range(0, S[0], k):
        for j in range(0, S[1], k):
            if np.any(Z[i:i+k, j:j+k]):
                count += 1
    return count

def fractal_dim(weights):
    if isinstance(weights, torch.Tensor):
        W = weights.data.cpu().numpy()
    else:
        W = weights
    if W.ndim != 2 or min(W.shape) < 4:
        return np.nan
    Z = np.abs(W) > 1e-5
    if not np.any(Z):
        return np.nan
    sizes = np.arange(2, min(Z.shape)//2+1)
    counts = [box_count(Z, size) for size in sizes if size < Z.shape[0] and size < Z.shape[1] and box_count(Z, size) > 0]
    if len(counts) > 1:
        coeffs = np.polyfit(np.log(sizes[:len(counts)]), np.log(counts), 1)
        return -coeffs[0]
    else:
        return np.nan

def conditional_smooth(pred, curvature, window=3, threshold=0.1):
    pred_np = pred.detach().cpu().numpy().flatten()
    smooth_mask = (np.abs(curvature) < threshold)
    smoothed = pred_np.copy()
    for i in range(window, len(pred_np) - window):
        if smooth_mask[i]:
            smoothed[i] = np.mean(pred_np[i - window:i + window + 1])
    return torch.tensor(smoothed).unsqueeze(1).to(pred.device)

def run_experiment(model_cls, signal="chaotic_sine", steps=200, seed=42):
    x, y = get_signal(signal, steps, seed)
    device = x.device
    model = model_cls(input_size=x.shape[1], hidden_size=8, output_size=1, device=device)
    controller = FractalQBEController()
    entropy_monitor = EntropyMonitorLite(momentum=0.9)
    logs = []
    cimm_entropies, cimm_hsizes, cimm_fractals, cimm_feedbacks, cimm_losses = [], [], [], [], []
    cimm_raw_preds, cimm_smoothed_preds = [], []
    prev_yhat = None
    for t in range(steps):
        yhat = model(x)
        loss = compute_loss(yhat, y)
        entropy = compute_entropy(model)
        entropy_monitor.update(yhat)
        # --- TRAINING STEP: update weights ---
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
        cimm_entropies.append(entropy)
        cimm_hsizes.append(model.hidden_dim)
        cimm_losses.append(loss.item())
        cimm_feedbacks.append(float((torch.abs(yhat - y) < 0.3).float().mean().item()))
        # Fractal dimension logging
        if t % 10 == 0:
            fd = fractal_dim(model.W)
            cimm_fractals.append(fd if not (np.isnan(fd) or np.isinf(fd)) else np.nan)
        # Save weights image
        if t % 50 == 0:
            plt.figure()
            plt.imshow(model.W.detach().cpu().numpy(), aspect='auto', cmap='bwr')
            plt.colorbar()
            plt.title(f'TinyCIMM Weights at step {t}')
            plt.tight_layout()
            plt.savefig(os.path.join(IMG_DIR, f'cimm_weights_step_{t}.png'))
            plt.close()
        # Save fractal diagnostic
        if t % 10 == 0:
            fd = fractal_dim(model.W)
            if not np.isnan(fd):
                plt.figure()
                plt.title(f"Fractal Dim (step {t}) = {fd:.2f}")
                plt.imshow(np.abs(model.W.detach().cpu().numpy()) > 1e-5, aspect='auto', cmap='gray_r')
                plt.savefig(os.path.join(IMG_DIR, f'fractal_dim_diag_{t}.png'))
                plt.close()
            plt.figure()
            plt.hist(model.W.detach().cpu().numpy().flatten(), bins=30)
            plt.title(f"Weight Histogram (step {t})")
            plt.savefig(os.path.join(IMG_DIR, f'weight_hist_{t}.png'))
            plt.close()
        # PCA of activations
        if t % 20 == 0 and hasattr(model, 'micro_memory') and len(model.micro_memory) >= 2:
            activations = np.concatenate(model.micro_memory, axis=0)
            if activations.shape[0] > 2 and activations.shape[1] > 1:
                pca = PCA(n_components=2)
                pca_proj = pca.fit_transform(activations)
                error_vals = np.abs((yhat - y).detach().cpu().numpy().flatten())
                n_points = pca_proj.shape[0]
                if len(error_vals) < n_points:
                    error_vals = np.pad(error_vals, (0, n_points - len(error_vals)), mode='constant')
                elif len(error_vals) > n_points:
                    error_vals = error_vals[-n_points:]
                plt.figure()
                plt.scatter(pca_proj[:,0], pca_proj[:,1], c=error_vals, cmap='coolwarm', s=10)
                plt.colorbar(label='Error Magnitude')
                plt.title(f'Neuron Activation PCA (step {t})')
                plt.tight_layout()
                plt.savefig(os.path.join(IMG_DIR, f'feedback_geometry_pca_{t}.png'))
                plt.close()
        # Store predictions for diagnostics
        cimm_raw_preds.append(yhat.detach().cpu().numpy().flatten())
        # Smoothing (optional, can be expanded)
        curvature = torch.gradient(torch.gradient(y.squeeze())[0])[0].detach().cpu().numpy()
        yhat_smooth = conditional_smooth(yhat, curvature)
        cimm_smoothed_preds.append(yhat_smooth.detach().cpu().numpy().flatten())
    save_logs(logs, signal)
    # Final plots
    plt.figure()
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), label='Ground Truth')
    if len(cimm_raw_preds) > 0:
        plt.plot(x.cpu().numpy(), np.array(cimm_raw_preds)[-1], label='TinyCIMM Raw Prediction', linestyle='dashed')
    if len(cimm_smoothed_preds) > 0:
        plt.plot(x.cpu().numpy(), np.array(cimm_smoothed_preds)[-1], label='TinyCIMM Smoothed Prediction')
    plt.legend()
    plt.title('Predictions vs. Ground Truth')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f'pred_vs_truth_{signal}.png'))
    plt.close()
    plt.figure()
    plt.plot(cimm_entropies, label='TinyCIMM entropy')
    plt.xlabel('Iteration')
    plt.ylabel('Output entropy')
    plt.legend()
    plt.title('Entropy Evolution')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f'entropy_evolution_{signal}.png'))
    plt.close()
    plt.figure()
    plt.plot(cimm_hsizes, label='TinyCIMM hidden size')
    plt.xlabel('Iteration')
    plt.ylabel('Hidden Layer Size')
    plt.legend()
    plt.title('Hidden Layer Size Evolution')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f'hidden_layer_size_evolution_{signal}.png'))
    plt.close()
    plt.figure()
    cfd_x = np.arange(0, steps, 10)[:len(cimm_fractals)]
    cfd_np = np.array(cimm_fractals)
    plt.plot(cfd_x[~np.isnan(cfd_np)], cfd_np[~np.isnan(cfd_np)], label='TinyCIMM fractal dim (W)')
    plt.xlabel('Iteration')
    plt.ylabel('Fractal Dimension')
    plt.legend()
    plt.title('Fractal Dimension Evolution')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f'fractal_dim_evolution_{signal}.png'))
    plt.close()
    plt.figure()
    plt.plot(cimm_feedbacks, label='TinyCIMM Feedback (Accuracy)')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy Fraction (<0.3 err)')
    plt.legend()
    plt.title('TinyCIMM Feedback/Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f'feedback_accuracy_{signal}.png'))
    plt.close()
    plt.figure()
    plt.plot(cimm_losses, label='TinyCIMM MSE Loss')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('TinyCIMM Raw Loss Evolution')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f'loss_evolution_{signal}.png'))
    plt.close()

def run_all_experiments():
    test_cases = [
        ("clean_sine", {}),
        ("amp_mod_sine", {}),
        ("freq_mod_sine", {}),
        ("noisy_sine", {}),
        ("chaotic_sine", {"hidden_size": 24, "micro_memory_size": 10, "symbolic_hold_steps": 20}),
    ]
    for test_name, model_kwargs in test_cases:
        print(f"\n=== Running experiment: {test_name} ===")
        img_dir = os.path.join("experiment_images", test_name)
        os.makedirs(img_dir, exist_ok=True)
        # Run TinyCIMM experiment (existing logic, but capture predictions and losses)
        x, y = get_signal(test_name, steps=200, seed=42)
        device = x.device
        if test_name == "clean_sine":
            model = TinyCIMM(input_size=x.shape[1], hidden_size=8, output_size=1, device=device)
        else:
            hidden_size = model_kwargs.get("hidden_size", 8)
            model_kwargs_no_hidden = {k: v for k, v in model_kwargs.items() if k != "hidden_size"}
            model = TinyCIMM(input_size=x.shape[1], hidden_size=hidden_size, output_size=1, device=device, **model_kwargs_no_hidden)
        controller = FractalQBEController()
        entropy_monitor = EntropyMonitorLite(momentum=0.9)
        logs = []
        cimm_entropies, cimm_hsizes, cimm_fractals, cimm_feedbacks, cimm_losses = [], [], [], [], []
        cimm_raw_preds, cimm_smoothed_preds = [], []
        for t in range(200):
            yhat = model(x)
            loss = compute_loss(yhat, y)
            entropy = compute_entropy(model)
            entropy_monitor.update(yhat)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            if hasattr(model, 'grow_and_prune'):
                avg_feedback = float(torch.abs(yhat - y).mean().item())
                model.grow_and_prune(avg_feedback, entropy, float(loss.item()), controller, entropy_monitor=entropy_monitor)
            logs.append({'step': t, 'loss': loss.item(), 'entropy': entropy, 'neurons': model.hidden_dim})
            cimm_entropies.append(entropy)
            cimm_hsizes.append(model.hidden_dim)
            cimm_losses.append(loss.item())
            cimm_feedbacks.append(float((torch.abs(yhat - y) < 0.3).float().mean().item()))
            if t % 10 == 0:
                fd = fractal_dim(model.W)
                cimm_fractals.append(fd if not (np.isnan(fd) or np.isinf(fd)) else np.nan)
            # Save weights image
            if t % 50 == 0:
                plt.figure()
                plt.imshow(model.W.detach().cpu().numpy(), aspect='auto', cmap='bwr')
                plt.colorbar()
                plt.title(f'TinyCIMM Weights at step {t}')
                plt.tight_layout()
                plt.savefig(os.path.join(img_dir, f'cimm_weights_step_{t}.png'))
                plt.close()
            # Save fractal diagnostic
            if t % 10 == 0:
                fd = fractal_dim(model.W)
                if not np.isnan(fd):
                    plt.figure()
                    plt.title(f"Fractal Dim (step {t}) = {fd:.2f}")
                    plt.imshow(np.abs(model.W.detach().cpu().numpy()) > 1e-5, aspect='auto', cmap='gray_r')
                    plt.savefig(os.path.join(img_dir, f'fractal_dim_diag_{t}.png'))
                    plt.close()
                plt.figure()
                plt.hist(model.W.detach().cpu().numpy().flatten(), bins=30)
                plt.title(f"Weight Histogram (step {t})")
                plt.savefig(os.path.join(img_dir, f'weight_hist_{t}.png'))
                plt.close()
            # PCA of activations
            if t % 20 == 0 and hasattr(model, 'micro_memory') and len(model.micro_memory) >= 2:
                activations = np.concatenate(model.micro_memory, axis=0)
                if activations.shape[0] > 2 and activations.shape[1] > 1:
                    pca = PCA(n_components=2)
                    pca_proj = pca.fit_transform(activations)
                    error_vals = np.abs((yhat - y).detach().cpu().numpy().flatten())
                    n_points = pca_proj.shape[0]
                    if len(error_vals) < n_points:
                        error_vals = np.pad(error_vals, (0, n_points - len(error_vals)), mode='constant')
                    elif len(error_vals) > n_points:
                        error_vals = error_vals[-n_points:]
                    plt.figure()
                    plt.scatter(pca_proj[:,0], pca_proj[:,1], c=error_vals, cmap='coolwarm', s=10)
                    plt.colorbar(label='Error Magnitude')
                    plt.title(f'Neuron Activation PCA (step {t})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(img_dir, f'feedback_geometry_pca_{t}.png'))
                    plt.close()
            # Store predictions for diagnostics
            cimm_raw_preds.append(yhat.detach().cpu().numpy().flatten())
            # Smoothing (optional, can be expanded)
            curvature = torch.gradient(torch.gradient(y.squeeze())[0])[0].detach().cpu().numpy()
            yhat_smooth = conditional_smooth(yhat, curvature)
            cimm_smoothed_preds.append(yhat_smooth.detach().cpu().numpy().flatten())
        save_logs(logs, test_name)
        # Run MLP benchmark and capture predictions/losses
        class SimpleMLP(nn.Module):
            def __init__(self, input_size, hidden_size=16, output_size=1):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size)
                )
            def forward(self, x):
                return self.net(x)
        mlp_model = SimpleMLP(input_size=x.shape[1], hidden_size=16, output_size=1).to(device)
        mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.01)
        mlp_losses = []
        mlp_preds = []
        for t in range(200):
            yhat = mlp_model(x)
            loss = compute_loss(yhat, y)
            mlp_optimizer.zero_grad()
            loss.backward()
            mlp_optimizer.step()
            mlp_losses.append(loss.item())
            mlp_preds.append(yhat.detach().cpu().numpy().flatten())
        save_logs([{'step': t, 'loss': l} for t, l in enumerate(mlp_losses)], f"mlp_{test_name}")
        # Overlayed plots
        # Predictions vs. Ground Truth
        plt.figure()
        plt.plot(x.cpu().numpy(), y.cpu().numpy(), label='Ground Truth')
        if len(cimm_raw_preds) > 0:
            plt.plot(x.cpu().numpy(), np.array(cimm_raw_preds)[-1], label='TinyCIMM', linestyle='dashed')
        plt.plot(x.cpu().numpy(), np.array(mlp_preds)[-1], label='MLP', linestyle='dotted')
        plt.legend()
        plt.title('Predictions vs. Ground Truth (TinyCIMM vs MLP)')
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, f'pred_vs_truth_overlay_{test_name}.png'))
        plt.close()
        # Loss Evolution
        plt.figure()
        plt.plot(cimm_losses, label='TinyCIMM')
        plt.plot(mlp_losses, label='MLP')
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title('Loss Evolution (TinyCIMM vs MLP)')
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, f'loss_evolution_overlay_{test_name}.png'))
        plt.close()
        # ...existing code for TinyCIMM-only plots (entropy, fractal, etc)...

if __name__ == "__main__":
    run_all_experiments()

