import torch
from skopt.space import Real, Integer
from self_optimizing_neural_architectures import SelfOptimizingNeuralArchitectures
from self_adaptive_learning_controller import SelfAdaptiveLearningController
from entropy_monitoring import EntropyMonitor  # Ensure this is imported
from quantum_potential_layer import QuantumPotentialLayer
import logging
import sympy
import matplotlib.pyplot as plt
import warnings
import numpy as np
from scipy.stats import chisquare, entropy
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress specific warning from skopt
warnings.filterwarnings("ignore", message="The objective has been evaluated at point")

def get_device():
    """Returns the best available device, with fallback to CPU if necessary."""
    if torch.cuda.is_available():
        try:
            # Try using NCCL for distributed training
            torch.cuda.nccl_version()
            return torch.device("cuda")
        except Exception as e:
            print(f"NCCL not available, falling back to CPU: {e}")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

def compute_error_metrics(y_true, y_pred):
    errors = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Handle division by zero in Mean Percentage Error calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        mpe = np.mean(np.where(y_true != 0, (errors / y_true) * 100, 0))

    std_dev = np.std(errors)

    # Chi-Square Statistic
    observed = np.maximum(y_true, 0).astype(int)
    expected = np.maximum(y_pred, 0).astype(int)
    
    # Flatten arrays if they have multiple dimensions
    observed = observed.flatten()
    expected = expected.flatten()

    # Normalize observed and expected to ensure their sums match
    observed_sum = np.sum(observed)
    expected_sum = np.sum(expected)
    if observed_sum != 0 and expected_sum != 0:
        observed = observed * (expected_sum / observed_sum)
    
    # Add a small epsilon to expected to avoid division by zero
    epsilon = 1e-10
    expected = expected + epsilon
    
    # Ensure sums match exactly
    observed_sum = np.sum(observed)
    expected_sum = np.sum(expected)
    if observed_sum != expected_sum:
        expected = expected * (observed_sum / expected_sum)
    
    # Ensure sums are non-zero
    if observed_sum == 0 or expected_sum == 0:
        chi_sq, p_value = np.nan, np.nan
    else:
        chi_sq, p_value = chisquare(observed, expected)
    
    # Shannon Entropy Calculation
    p = np.histogram(y_pred, bins=50, density=True)[0] + 1e-10  # Avoid log(0)
    entropy = -np.sum(p * np.log2(p))

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "Mean Percentage Error": float(mpe),
        "Standard Deviation": float(std_dev),
        "Chi-Square Statistic": chi_sq,
        "p-value": p_value,
        "Shannon Entropy": float(entropy)
    }

class CIMM:
    def __init__(self, model_class, param_space, anchor_data, energy=0.5, preprocess_fn=None, initial_entropy=1.0, learning_rate=0.01, lambda_factor=0.1, max_epochs=100, early_stopping_threshold=0.001):
        """Initialize CIMM and perform pretraining on the anchor dataset."""
        self.optimizer = SelfOptimizingNeuralArchitectures(model_class, param_space)
        self.entropy_monitor = EntropyMonitor(initial_entropy, learning_rate)  # Use the imported EntropyMonitor
        self.qpl = QuantumPotentialLayer(input_dim=10, output_dim=10, scaling_factor=lambda_factor)  # Provide appropriate dimensions
        self.model_instance = model_class(hidden_size=param_space[1].low).to(device)  # Create an instance of the model
        self.controller = SelfAdaptiveLearningController(self.model_instance, initial_entropy, learning_rate, lambda_factor)  # Pass the model instance
        self.max_epochs = max_epochs
        self.early_stopping_threshold = early_stopping_threshold
        self.preprocess_fn = preprocess_fn
        self.energy = energy
        self.training_log = []  # Store training history

        # Store past predictions and errors
        self.past_predictions = []
        self.actual_values = []
        self.error_rates = []
        self.patience = 5

        # Initialize optimizer as a class attribute
        self.optimizer_instance = torch.optim.Adam(self.model_instance.parameters(), lr=self.entropy_monitor.learning_rate, weight_decay=0.001)  # Add weight decay

        # Pretrain the model using anchor data
        self.pretrain(anchor_data)

    def pretrain(self, anchor_data):
        """Trains the model on the initial anchor dataset."""
        best_params, _ = self.optimizer.optimize()
        self.model_instance = self.optimizer.model_class(hidden_size=int(best_params[1])).to(device)

        prev_entropy = self.entropy_monitor.entropy

        for epoch in range(self.max_epochs):
            learning_rate = self.entropy_monitor.monitor(anchor_data)
            for param_group in self.optimizer_instance.param_groups:
                param_group["lr"] = learning_rate  # Dynamically adjust learning rate

            self.optimizer_instance.zero_grad()
            self.model_instance, _ = self.controller.update_model(anchor_data, self.energy)  # Ensure loss is computed
            anchor_data = anchor_data.view(-1, anchor_data.size(-1))  # Ensure correct shape
            loss = torch.nn.MSELoss()(self.model_instance(anchor_data), anchor_data)  # Compute loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_instance.parameters(), max_norm=1.0)  # Clip gradients
            self.optimizer_instance.step()

            print(f"Epoch {epoch}: Learning Rate={learning_rate}, Loss={loss.item()}")

            # Log training history
            self.training_log.append({"epoch": epoch, "loss": loss.item(), "entropy": self.entropy_monitor.entropy})

            # Stop early if entropy stabilizes
            if abs(self.entropy_monitor.entropy - prev_entropy) < self.early_stopping_threshold:
                break
            prev_entropy = self.entropy_monitor.entropy

        print("✅ Pretraining complete. Model is ready for live streaming.")

    def run(self, new_data_point, actual_value=None):
        """Processes a single new data point, updates the model, and integrates feedback."""
        new_data_point = new_data_point.to(device)
        with torch.no_grad():
            prediction = self.model_instance(new_data_point.unsqueeze(0)).cpu().numpy().flatten()

        self.past_predictions.append(prediction)

        # Update model dynamically using feedback
        if actual_value is not None:
            self.update_with_feedback(self.model_instance, new_data_point.unsqueeze(0), [actual_value])

        print(f"Live Data: Prediction={prediction}, Actual={actual_value}")
        return prediction

    def update_with_feedback(self, model, data, actual_value):
        """Adjusts the model dynamically based on real feedback."""
        if isinstance(actual_value, torch.Tensor):
            actual_value = actual_value.squeeze().tolist()  # Convert tensor to list if needed

        self.actual_values.append(actual_value)

        # Convert actual_value to tensor (ensuring it's a compatible shape)
        actual_value_tensor = torch.tensor(actual_value, dtype=torch.float32).to(device)

        past_predictions_tensor = torch.tensor(self.past_predictions[-1], dtype=torch.float32).to(device)

        # Ensure shape compatibility
        min_length = min(len(actual_value_tensor), len(past_predictions_tensor))
        actual_value_tensor = actual_value_tensor[:min_length]
        past_predictions_tensor = past_predictions_tensor[:min_length]

        # Compute error
        error = actual_value_tensor - past_predictions_tensor

        # Dynamically adjust learning parameters based on entropy
        current_entropy = self.entropy_monitor.entropy  # Corrected attribute access
        learning_rate = self.entropy_monitor.learning_rate
        update_frequency = max(1, int(5 + np.log1p(current_entropy)))  # Adjusted heuristic
        gradient_smoothing = min(0.1, 1 / (current_entropy + 1e-10))  # Example adjustment

        print(f"Update Frequency: {update_frequency}")

        # Buffer mechanism for updates
        if len(self.past_predictions) % update_frequency == 0:
            # Update the model using the new error signal
            self.optimizer_instance.zero_grad()
            model_output = model(data).reshape(-1)
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(model_output, actual_value_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_instance.parameters(), max_norm=1.0)  # Clip gradients
            self.optimizer_instance.step()

            print(f"Updated model based on feedback: Loss={loss.item()}, Learning Rate={learning_rate}, Update Frequency={update_frequency}, Gradient Smoothing={gradient_smoothing}")

    def evaluate_model(self, model, validation_data):
        """Evaluate the model's performance on a validation dataset."""
        model.eval()
        validation_data = validation_data.to(device)
        with torch.no_grad():
            output = model(validation_data).cpu().numpy()
            # Dummy target for demonstration purposes
            target = np.zeros(validation_data.size(0))  # Ensure target size matches output size
            
            metrics = compute_error_metrics(target, output)
            logging.info(f"Validation Metrics: {metrics}")
            
            print(f"Validation Metrics: {metrics}")
            
        return metrics

    def solve_issue(self, issue_data, issue_energy):
        """Dynamically solve a new issue using the CIMM model."""
        logging.info("Solving new issue dynamically.")
        updated_model = self.run(issue_data, issue_energy)
        logging.info("Issue solved successfully.")
        return updated_model

    def early_stopping(self, validation_data):
        # Implement early stopping based on entropy
        model_entropy = entropy(torch.softmax(self.model_instance(validation_data), dim=1).detach().numpy(), axis=1)
        if np.mean(model_entropy) < self.entropy_threshold:
            logging.info("Early stopping triggered due to low entropy.")
            return True
        return False

# Example usage
if __name__ == "__main__":
    class ExampleModel(torch.nn.Module):
        def __init__(self, hidden_size):
            super(ExampleModel, self).__init__()
            self.hidden_size = int(hidden_size)
            self.linear1 = torch.nn.Linear(10, 128)
            self.linear2 = torch.nn.Linear(128, 64)
            self.linear3 = torch.nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = self.linear3(x)
            return x

    param_space = [
        Real(1e-5, 1e-1, name='learning_rate'),
        Integer(10, 100, name='hidden_size'),
    ]

    # Example anchor data
    anchor_data = torch.randn(100, 10)

    # Initialize CIMM with the example model and anchor data
    cimm = CIMM(ExampleModel, param_space, anchor_data)

    # Example new data point
    new_data_point = torch.randn(10)
    actual_value = 0.5  # Example actual value

    # Run CIMM with the new data point
    prediction = cimm.run(new_data_point, actual_value)
    print(f"Prediction: {prediction}")

    # Evaluate the model with some validation data
    validation_data = torch.randn(50, 10)
    metrics = cimm.evaluate_model(cimm.model_instance, validation_data)
    print(f"Validation Metrics: {metrics}")
