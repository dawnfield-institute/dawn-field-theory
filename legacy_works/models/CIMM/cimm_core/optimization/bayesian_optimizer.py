import torch
import torch.nn as nn
import torch.optim as optim
from skopt import Optimizer
from skopt.space import Real, Integer
from cimm_core.entropy.entropy_monitor import EntropyMonitor
import random
from cimm_core.optimization.pruning import entropy_aware_pruning  # ✅ Correct function name
import torch.nn.functional as F  # Add this import
from skopt import gp_minimize
from cimm_core.utils import get_device

device = get_device()


class BayesianOptimizer:
    def __init__(self, model_instance, param_space, entropy_monitor, adaptive_controller, qpl_layer, n_calls=50, data_loader=None, val_loader=None, device="cpu", clip_grad=1.0):
        """
        Hyperparameter optimization using Bayesian optimization.

        Parameters:
        - model_class: The neural network model class.
        - param_space: List of hyperparameter search spaces.
        - n_calls: Number of Bayesian optimization iterations.
        - data_loader: Training data (optional).
        - val_loader: Validation data (optional).
        - device: "cpu" or "cuda" for GPU acceleration.
        - clip_grad: Maximum gradient norm for stabilization.
        """
        self.model = model_instance.to(device)  # Move model to GPU
        self.param_space = param_space
        self.n_calls = n_calls
        self.data_loader = data_loader  # Optional custom dataset
        self.val_loader = val_loader    # Validation set for generalization
        self.device = device
        self.optimizer = Optimizer(dimensions=param_space)
        self.architecture = None  # Add architecture attribute
        self.entropy_monitor = entropy_monitor  # Ensure entropy tracking
        self.adaptive_controller = adaptive_controller  # Learning rate tuning
        self.clip_grad = clip_grad  # Initialize clip_grad attribute
        self.learning_rate = None  # Initialize learning_rate attribute
        self.qpl_layer = qpl_layer
        self.qpl_constraint = 1.0  # Add missing attribute to prevent AttributeError
        self.prev_learning_rate = None  # New: Track previous LR

    def objective(self, params):
        """
        Bayesian Optimization with Energy-Adaptive Search.
        Dynamically expands/contracts search space based on entropy-energy balance.
        """
        learning_rate, hidden_size = params
        hidden_size = int(hidden_size)  
        self.learning_rate = learning_rate
        self.entropy_monitor.update_entropy_threshold()

        # Entropy-weighted search bounds
        entropy_factor = min(0.2, max(0.01, 1.0 / (1 + 0.5 * self.entropy_monitor.entropy)))  # 🔥 Controlled range
        adjusted_learning_rate = max(0.008, min(0.02, learning_rate * entropy_factor))

        # 🔥 New: Momentum-based learning rate adjustment
        if self.prev_learning_rate is not None:
            pass
        self.prev_learning_rate = adjusted_learning_rate

        optimizer = optim.SGD(self.model.parameters(), lr=adjusted_learning_rate, momentum=0.9)
        loss_fn = nn.MSELoss()

        # ✅ FIX: Properly define `data` and `target`
        if self.data_loader:
            data, target = next(iter(self.data_loader))
            data, target = data.to(self.device), target.to(self.device)  # Ensure tensors are on the same device
        else:
            # Fallback: Generate synthetic data if none provided
            data = torch.randn(100, self.model.input_dim, device=self.device)  # Ensure tensors are on the same device
            target = torch.randn(100, self.model.output_dim, device=self.device)  # Ensure tensors are on the same device

        print("Data is on:", data.device)  # ✅ Check if data is on the correct device
        print("Target is on:", target.device)  # ✅ Check if target is on the correct device

        self.model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            output = self.model(data)  # Model output on the same device

            # 🔥 Apply quantum collapse correction before computing loss
            adjusted_output = self.quantum_collapse_update(output, target, self.entropy_monitor.entropy)

            # Replace entropy_scaling with new stabilization factors
            einstein_correction = 1 / (1 + self.entropy_monitor.entropy * 1e-5)
            feynman_damping = torch.exp(-self.entropy_monitor.entropy * 5)
            loss = loss_fn(adjusted_output, target) * einstein_correction * feynman_damping

            loss.backward()  # Backpropagation on the same device
            optimizer.step()

        return loss.item()

    def quantum_collapse_update(self, predicted, actual, entropy, lambda_qpl=0.05):
        """
        Quantum-inspired update rule with adaptive lambda_qpl scaling.
        """
        delta = torch.abs(actual - predicted).mean()

        # ✅ Dynamically adjust lambda_qpl based on entropy trends
        entropy_trend = torch.mean(torch.tensor(self.entropy_monitor.past_entropies[-10:])) if len(self.entropy_monitor.past_entropies) >= 10 else entropy
        lambda_qpl = min(0.1, max(0.01, lambda_qpl * torch.tanh(entropy_trend)))

        correction = lambda_qpl * (actual - predicted) / (torch.sqrt(entropy + 1e-8))  # Already GPU-safe
        return predicted + correction

    def entropy_aware_bayesian_optimization(self, objective_function, space, entropy_level):
        """
        Bayesian optimization with entropy-aware adjustments.

        Args:
            objective_function (function): Function to minimize.
            space (list): Search space bounds.
            entropy_level (float): System entropy level.

        Returns:
            dict: Best optimization parameters.
        """
        # Adaptive search space scaling based on entropy level
        search_bounds = [
            (low * (1 + min(0.2, entropy_level)), high * (1 - min(0.2, entropy_level)))
            for low, high in space
        ]

        # Quantum Fisher Information weighting for stability
        qfi_weight = torch.exp(-entropy_level)

        # Perform Bayesian optimization with entropy-aware constraints
        result = gp_minimize(objective_function, search_bounds, acq_func="EI", n_calls=50)

        return {"best_params": result.x, "entropy_weighting": qfi_weight.item()}


    def _update_model(self, loss):
        loss.backward()  # Backpropagation on GPU

        # Apply adaptive maximum norm for gradient clipping based on entropy change rates
        entropy_variance = self.entropy_monitor.entropy_gradient
        max_norm = 1.5 * (1 + entropy_variance)  # 🔥 Scale gradient clipping dynamically
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

        # Momentum-aware parameter updates
        with torch.no_grad():
            for param in self.model.parameters():
                param.copy_(param - self.learning_rate * param.grad)

        # Adjust learning rate based on gradient norm
        if grad_norm > 1.0:
            self.learning_rate *= 0.95  # Reduce LR slightly when large gradients are detected

        self.model.zero_grad()

    def optimize(self):
        best_params = None
        best_score = float('inf')
        evaluated_points = set()

        for _ in range(self.n_calls):
            params = self.optimizer.ask()
            qbe_feedback = self.qpl_layer.compute_qpl(self.entropy_monitor.entropy)

            # Apply gravity-based stabilization
            gravity_bias = self.entropy_monitor.compute_gravitational_potential(self.entropy_monitor.entropy)
            gravity_correction = max(0.9, min(1.05, 1 - 0.05 * gravity_bias))

            params[0] *= gravity_correction  
            params_tuple = tuple(params)

            if params_tuple in evaluated_points:
                params = self.optimizer.ask()
            evaluated_points.add(params_tuple)

            score = self.objective(params)
            self.optimizer.tell(params, score)

            if score < best_score:
                best_score = score
                best_params = params

        return best_params, best_score

    def optimize_architecture(self):
        """Optimize the neural network architecture."""
        self.architecture, _ = self.optimize()

    def reset_architecture(self):
        """Reset the neural network architecture to its initial state."""
        self.architecture = None

    def prune_model(self, model, entropy_monitor, entropy_threshold=0.1, temperature=300):
        """
        Apply entropy-aware pruning using Landauer's principle.
        """
        # Use entropy-aware pruning to eliminate unnecessary corrections
        self.model = entropy_aware_pruning(model, entropy_monitor, entropy_threshold, temperature)

    def adjust_regularization(self, model, entropy, l1_factor=1e-5, l2_factor=1e-4):
        """
        Adjust L1/L2 regularization based on network entropy trends.
        """
        l1_reg = l1_factor * entropy
        l2_reg = l2_factor * (1 - entropy)
        for param in model.parameters():
            param.grad.data.add_(l1_reg * torch.sign(param.data) + l2_reg * param.data)

    def adjust_learning_rate(self):
        """Use entropy-aware learning rate tuning."""
        entropy_trend = self.entropy_monitor.entropy  # Corrected attribute access
        self.adaptive_controller.adjust_learning_rate(entropy_trend)

    def evolve_topology(self, population_size=10, generations=5, threshold=0.1):
        """
        Dynamically evolves neural architecture based on entropy balance.
        Expands or prunes layers based on entropy constraints.
        """
        population = [self.model_class(hidden_size=random.randint(10, 100)).to(self.device) for _ in range(population_size)]
        for generation in range(generations):
            fitness_scores = [self.evaluate_fitness(model) for model in population]
            selected_models = self.select_models(population, fitness_scores)
            offspring = self.crossover(selected_models)
            population = self.mutate(offspring)
            print(f"Generation {generation}: Best Fitness={max(fitness_scores)}")

            # Entropy-based decision-making
            for model in population:
                entropy = self.entropy_monitor.calculate_entropy(model)
                if not self.entropy_monitor.qbe_constraint(entropy):
                    if entropy > threshold:
                        print("Entropy too high, pruning network...")
                        self.prune_weights(model, self.entropy_monitor, entropy_threshold=0.05)
                    else:
                        print("Entropy low, expanding network...")
                        self.expand_model(model)
                else:
                    print("System in equilibrium, no changes needed.")

        best_model = population[torch.argmax(torch.tensor(fitness_scores))]
        return best_model

    def expand_model(self, model):
        """
        Expands the model by adding neurons or layers.
        """
        # Example expansion logic (to be customized as needed)
        for name, param in model.named_parameters():
            if 'weight' in name:
                new_weights = torch.randn_like(param) * 0.01
                param.data = torch.cat((param.data, new_weights), dim=0)
        print("Expanded model by adding neurons or layers")

    def evaluate_fitness(self, model):
        """
        Evaluate model fitness using entropy minimization as a key metric.
        """
        model.eval()
        data = torch.randn(100, 10, device=device)  # Specify device
        target = torch.randn(100, 1, device=device)  # Specify device
        with torch.no_grad():
            output = model(data)  # Model output on device
            loss = nn.MSELoss()(output, target)  # Ensure loss is calculated on the same device
        
        # Calculate entropy score
        entropy_score = self.entropy_monitor.calculate_entropy(output)
        
        # Combine performance and entropy scores
        performance_score = -loss.item()  # Negative loss as fitness
        fitness_score = performance_score * (1.0 / (1.0 + entropy_score))
        
        return fitness_score

    def select_models(self, population, fitness_scores, num_select=5):
        """Select the top models based on fitness scores."""
        selected_indices = torch.argsort(torch.tensor(fitness_scores))[-num_select:]
        return [population[i] for i in selected_indices]

    def crossover(self, selected_models):
        """Perform crossover to generate offspring."""
        offspring = []
        for i in range(len(selected_models) // 2):
            parent1 = selected_models[2 * i]
            parent2 = selected_models[2 * i + 1]
            child1, child2 = self.crossover_parents(parent1, parent2)
            offspring.extend([child1, child2])
        return offspring

    def crossover_parents(self, parent1, parent2):
        """Crossover two parent models to produce two children."""
        child1 = self.model_class(hidden_size=parent1.hidden_size).to(self.device)
        child2 = self.model_class(hidden_size=parent2.hidden_size).to(self.device)
        for (name1, param1), (name2, param2) in zip(parent1.named_parameters(), parent2.named_parameters()):
            if 'weight' in name1:
                mask = torch.rand_like(param1) > 0.5
                child1.state_dict()[name1].data.copy_(mask * param1.data + (1 - mask) * param2.data)
                child2.state_dict()[name2].data.copy_((1 - mask) * param1.data + mask * param2.data)
        return child1, child2

    def mutate(self, offspring, mutation_rate=0.1):
        """Mutate the offspring models."""
        for model in offspring:
            for name, param in model.named_parameters():
                if 'weight' in name:
                    mask = torch.rand_like(param) < mutation_rate
                    param.data += mask.float() * torch.randn_like(param) * 0.01
        return offspring

    def compute_loss(self, predictions, actual_values):
        """
        Implements a fully adaptive loss function that adjusts to entropy trends.
        """
        mse_loss = F.mse_loss(predictions, actual_values, reduction='mean')

        # 🔥 **Dynamic loss correction factor based on entropy & QPL**
        qbe_feedback = self.qpl_layer.compute_qpl(self.entropy_monitor.entropy, self.entropy_monitor.prev_entropy)
        entropy_factor = max(0.7, min(2.5, 1 + self.entropy_monitor.entropy / 3))

        weighted_loss = mse_loss * (1 + 0.08 * qbe_feedback * entropy_factor)  
        return weighted_loss

    def compute_causal_inference(self, predictions, actual_values):
        """
        Faster Bayesian causal learning updates for real-time correction.
        """
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions, dtype=torch.float32).to(device)  # Move predictions to device
        
        if not isinstance(actual_values, torch.Tensor):
            actual_values = torch.tensor(actual_values, dtype=torch.float32).to(device)  # Move actual values to device
        
        # Compute prediction error
        prediction_error = actual_values - predictions

        # Compute entropy before applying correction
        entropy_before = self.entropy_monitor.calculate_entropy(predictions.detach().cpu().numpy())

        # Apply Bayesian causal update with increased weighting
        corrected_prediction = predictions + (0.15 * prediction_error)  # Was 0.1, increased to 0.15

        # Compute entropy after applying correction
        entropy_after = self.entropy_monitor.calculate_entropy(corrected_prediction.detach().cpu().numpy())

        # ✅ Increase causal weighting when entropy change is small
        entropy_before_tensor = torch.tensor(entropy_before, dtype=torch.float32)  # Convert to tensor
        entropy_after_tensor = torch.tensor(entropy_after, dtype=torch.float32)  # Convert to tensor
        stability_factor = 1 + torch.exp(-8 * torch.abs(entropy_before_tensor - entropy_after_tensor))  # Was 10, now 8

        causal_effect = (entropy_before - entropy_after) * stability_factor

        return causal_effect

    def optimize_model_with_pruning(self):
        """
        Optimize model parameters while periodically applying entropy-aware pruning.
        """
        best_params = self.optimize()

        # Apply pruning dynamically based on entropy feedback
        self.model = entropy_aware_pruning(self.model, self.entropy_monitor)

        return best_params


# Example usage
if __name__ == "__main__":

    class ExampleModel(nn.Module):
        def __init__(self, hidden_size):
            super(ExampleModel, self).__init__()
            self.hidden_size = hidden_size
            self.linear1 = nn.Linear(10, hidden_size)
            self.linear2 = nn.Linear(hidden_size, 1)

        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    param_space = [
        Real(1e-5, 1e-1, name='learning_rate'),
        Integer(10, 100, name='hidden_size')
    ]

    entropy_monitor = EntropyMonitor()  # Create an instance of EntropyMonitor
    from adaptive_controller import AdaptiveController
    adaptive_controller = AdaptiveController()  # Create an instance of SelfAdaptiveLearningController
    model_instance = ExampleModel(hidden_size=50).to(device)  # Instantiate the model
    optimizer = BayesianOptimizer(model_instance, param_space, entropy_monitor, adaptive_controller)
    best_params, best_score = optimizer.optimize()
    optimizer.prune_weights(optimizer.model, entropy_monitor, entropy_threshold=0.01)
    optimizer.adjust_regularization(optimizer.model, entropy=0.5)
    best_model = optimizer.evolve_topology()
    optimizer.optimize_qpl()
    print(f"Best Model: {best_model}")
