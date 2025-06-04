import torch
import torch.nn as nn
import torch.optim as optim
from skopt import Optimizer
from skopt.space import Real, Integer
import numpy as np
from entropy_monitoring import EntropyMonitor
import random

class SelfOptimizingNeuralArchitectures:
    def __init__(self, model_class, param_space, n_calls=50, data_loader=None, val_loader=None, device="cpu", clip_grad=1.0):
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
        self.model_class = model_class
        self.param_space = param_space
        self.n_calls = n_calls
        self.data_loader = data_loader  # Optional custom dataset
        self.val_loader = val_loader    # Validation set for generalization
        self.device = device
        self.optimizer = Optimizer(dimensions=param_space)
        self.architecture = None  # Add architecture attribute
        self.entropy_monitor = EntropyMonitor()  # Add entropy_monitor attribute
        self.model = None  # Initialize model attribute
        self.clip_grad = clip_grad  # Initialize clip_grad attribute
        self.learning_rate = None  # Initialize learning_rate attribute

    def objective(self, params):
        """
        Optimizes model architecture while ensuring learning rate stays within defined bounds.
        """
        learning_rate, hidden_size = params
        hidden_size = int(hidden_size)  
        self.learning_rate = learning_rate  # Set learning_rate attribute

        self.model = self.model_class(hidden_size=hidden_size).to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)  # Use SGD with momentum
        loss_fn = nn.MSELoss()

        # Factor in entropy trends
        entropy_factor = 1.0 / (1 + self.entropy_monitor.entropy)
        
        # Normalize entropy to prevent over-adjustments
        normalized_entropy = np.clip(entropy_factor, 0.2, 5.0)
        
        # Scale learning rate using normalized entropy factor
        adjusted_learning_rate = max(min(learning_rate * normalized_entropy, 0.02), 0.005)

        optimizer.param_groups[0]['lr'] = adjusted_learning_rate

        # Print entropy and learning rate for monitoring
        print(f"Entropy: {self.entropy_monitor.entropy}, Learning Rate: {adjusted_learning_rate}")

        # Prepare data (support custom dataset or generate random data)
        if self.data_loader:
            data, target = next(iter(self.data_loader))
        else:
            data, target = torch.randn(100, 10).to(self.device), torch.zeros(100, 1).to(self.device)

        # Training loop with early stopping
        self.model.train()
        min_loss, patience, patience_counter = float("inf"), 10, 0
        for epoch in range(50):
            optimizer.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            self._update_model(loss)

            # Early stopping check
            if loss.item() < min_loss:
                min_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break  # Stop training early if no improvement

        # Evaluate on validation set if available
        if self.val_loader:
            self.model.eval()
            val_data, val_target = next(iter(self.val_loader))
            val_output = self.model(val_data.to(self.device))
            val_loss = loss_fn(val_output, val_target.to(self.device)).item()
            return val_loss  # Optimize based on validation performance

        return min_loss  # Return best training loss if no validation set is available

    def _update_model(self, loss):
        loss.backward()

        # Apply fixed maximum norm for gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)

        # Momentum-aware parameter updates
        with torch.no_grad():
            for param in self.model.parameters():
                param.copy_(param - self.learning_rate * param.grad)

        # Adjust learning rate based on gradient norm
        if grad_norm > 1.0:
            self.learning_rate *= 0.95  # Reduce LR slightly when large gradients are detected

        self.model.zero_grad()

    def optimize(self):
        """Run Bayesian optimization to find the best hyperparameters."""
        best_params = None
        best_score = float('inf')
        evaluated_points = set()
        max_retries = 10  # Add a maximum retry limit
        retries = 0

        for _ in range(self.n_calls):  # Number of iterations for optimization
            params = self.optimizer.ask()
            params_tuple = tuple(params)
            if params_tuple in evaluated_points:
                params = self.optimizer.ask()  # Get a new point if already evaluated
            evaluated_points.add(params_tuple)
            score = self.objective(params)
            self.optimizer.tell(params, score)
            if score < best_score:
                best_score = score
                best_params = params

            # Early stopping if a good enough score is found
            if best_score < 0.01:
                break

        # Ensure hidden_size is an integer and learning_rate is a float before returning
        best_params[0] = float(best_params[0])
        best_params[1] = int(best_params[1])
        
        # Enforce strict bounds on learning rate
        best_params[0] = max(0.005, min(0.02, best_params[0]))
        
        while not (0.005 <= best_params[0] <= 0.02) and retries < max_retries:
            retries += 1
            params = self.optimizer.ask()
            params_tuple = tuple(params)
            if params_tuple in evaluated_points:
                continue  # Skip values that don't match bounds
            evaluated_points.add(params_tuple)
            score = self.objective(params)
            self.optimizer.tell(params, score)
            if score < best_score:
                best_score = score
                best_params = params
            best_params[0] = max(0.005, min(0.02, float(best_params[0])))
            best_params[1] = int(best_params[1])

        if retries >= max_retries:
            raise ValueError("Optimization failed to find valid parameters within the retry limit.")
        
        # Reduce n_calls dynamically based on retries
        self.n_calls = min(self.n_calls, retries + 5)
        
        self.architecture = best_params  # Store the best architecture
        return best_params, best_score

    def optimize_architecture(self):
        """Optimize the neural network architecture."""
        self.architecture, _ = self.optimize()

    def reset_architecture(self):
        """Reset the neural network architecture to its initial state."""
        self.architecture = None

    def prune_weights(self, model, threshold=0.01):
        """Dynamically prune low-impact neurons in real-time."""
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = torch.abs(param) > threshold
                param.data *= mask.float()

    def adjust_regularization(self, model, entropy, l1_factor=1e-5, l2_factor=1e-4):
        """Adjust L1/L2 regularization based on network entropy."""
        l1_reg = l1_factor * entropy
        l2_reg = l2_factor * (1 - entropy)
        for param in model.parameters():
            param.grad.data.add_(l1_reg * torch.sign(param.data) + l2_reg * param.data)

    def evolve_topology(self, population_size=10, generations=5):
        """Use genetic algorithms to test different network structures."""
        population = [self.model_class(hidden_size=random.randint(10, 100)).to(self.device) for _ in range(population_size)]
        for generation in range(generations):
            fitness_scores = [self.evaluate_fitness(model) for model in population]
            selected_models = self.select_models(population, fitness_scores)
            offspring = self.crossover(selected_models)
            population = self.mutate(offspring)
            print(f"Generation {generation}: Best Fitness={max(fitness_scores)}")

        best_model = population[np.argmax(fitness_scores)]
        return best_model

    def evaluate_fitness(self, model):
        """Evaluate the fitness of a model based on its performance."""
        model.eval()
        data = torch.randn(100, 10).to(self.device)  # Example data
        target = torch.randn(100, 1).to(self.device)  # Example target
        with torch.no_grad():
            output = model(data)
            loss = nn.MSELoss()(output, target)
        return -loss.item()  # Negative loss as fitness

    def select_models(self, population, fitness_scores, num_select=5):
        """Select the top models based on fitness scores."""
        selected_indices = np.argsort(fitness_scores)[-num_select:]
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

    optimizer = SelfOptimizingNeuralArchitectures(ExampleModel, param_space)
    best_params, best_score = optimizer.optimize()
    optimizer.prune_weights(optimizer.model)
    optimizer.adjust_regularization(optimizer.model, entropy=0.5)
    best_model = optimizer.evolve_topology()
    print(f"Best Model: {best_model}")