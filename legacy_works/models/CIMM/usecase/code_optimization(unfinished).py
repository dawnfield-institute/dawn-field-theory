import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import numba
import time
import torch
import torch.nn as nn
import torch.optim as optim
from cimm_core.cimm import CIMM
from cimm_core.entropy.entropy_monitor import EntropyMonitor
from skopt.space import Real, Integer

# Optimize Torch Performance
#torch.set_num_threads(os.cpu_count())

# Define a Matrix Multiplication Model for CIMM
class MatrixMultiplicationModel(nn.Module):
    def __init__(self):
        super(MatrixMultiplicationModel, self).__init__()
        self.scale = nn.Parameter(torch.ones(1))  # Introduce a trainable parameter
    
    def forward(self, data):
            # Split the input tensor into A and B
            if data.dim() == 2:
                dim1, dim2 = data.shape
                A = data[:, :dim2//2]
                B = data[:, dim2//2:].transpose(0, 1)  # Transpose B to make it compatible for matrix multiplication
            elif data.dim() == 3:
                batch_size, dim1, dim2 = data.shape
                A = data[:, :, :dim2//2]
                B = data[:, :, dim2//2:].transpose(1, 2)  # Transpose B to make it compatible for matrix multiplication
            else:
                raise ValueError("Unsupported tensor shape")
            return self.scale * torch.matmul(A, B)

class CodeOptimizationUseCase:
    def __init__(self, learning_rate=0.01):
        self.model = MatrixMultiplicationModel()
        self.entropy_monitor = EntropyMonitor(initial_entropy=1.0, learning_rate=learning_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  # Now has trainable params
        self.loss_fn = nn.MSELoss()
    
    def execute(self, A, B):
        # Concatenate A and B along the last dimension
        data = torch.cat((A, B), dim=1)
        return self.model(data)  # Pass concatenated tensor
    
    def optimize(self, A, B):
        param_space = [
            Real(1e-5, 1e-1, name='learning_rate'),
        ]
        A_batches = torch.stack([A[i, :].unsqueeze(0) for i in range(A.shape[0])])  # Shape: (batch, 1, dim)
        B_batches = torch.stack([B[:, i].unsqueeze(1) for i in range(B.shape[1])])  # Shape: (dim, batch, 1)
        
        # Reshape B_batches to match A_batches shape
        B_batches = B_batches.permute(1, 0, 2)  # Shape: (batch, 1, dim)
        B_batches = B_batches.reshape(A_batches.shape)  # Reshape to match A_batches shape
        
        dataset = torch.stack([torch.cat((A_batches[i], B_batches[i]), dim=1) for i in range(len(A_batches))])
        
        cimm = CIMM(lambda: MatrixMultiplicationModel(), param_space, dataset)
        predictions = []
        for i in range(10):
            A_batch, B_batch = dataset[i].chunk(2, dim=1)  # Ensure correct tuple extraction
            prediction = cimm.run((A_batch, B_batch))  # Ensure correct data format
            actual_value = torch.matmul(A_batch, B_batch)  # Compute true result
            cimm.give_feedback((A_batch, B_batch), actual_value)  # Provide feedback
            predictions.append(prediction)
        return predictions

# Define MetaCIMM for Self-Optimizing CIMM
class MetaCIMM:
    def __init__(self):
        self.param_space = [
            Real(1e-5, 1e-1, name='learning_rate'),
            Real(0.1, 10.0, name='entropy_factor')
        ]
    
    def optimize_inner_cimm(self, A, B):
        best_performance = float("inf")
        best_learning_rate = None
        
        for trial in range(5):  # Iterate to refine CIMM hyperparameters
            learning_rate = np.random.uniform(1e-5, 1e-1)
            entropy_factor = np.random.uniform(0.1, 10.0)
            
            inner_cimm = CodeOptimizationUseCase(learning_rate=learning_rate)
            optimized_results = inner_cimm.optimize(A, B)
            
            execution_time = time.time()
            compiled_result = inner_cimm.execute(A, B)
            compiled_time = time.time() - execution_time
            
            if compiled_time < best_performance:
                best_performance = compiled_time
                best_learning_rate = learning_rate
                
        print(f"Best Learning Rate: {best_learning_rate}, Best Execution Time: {best_performance:.4f} sec")
        return best_learning_rate

# Generate test matrices
size = 1000
A = np.random.rand(size, size)
B = np.random.rand(size, size)
A_torch = torch.tensor(A, dtype=torch.float32)
B_torch = torch.tensor(B, dtype=torch.float32)

# Instantiate MetaCIMM and run self-optimization
meta_optimizer = MetaCIMM()
best_learning_rate = meta_optimizer.optimize_inner_cimm(A_torch, B_torch)

# Final optimized execution using best hyperparameters
final_optimizer = CodeOptimizationUseCase(learning_rate=best_learning_rate)
optimized_results = final_optimizer.optimize(A_torch, B_torch)

# Measure performance
start_time = time.time()
naive_result = np.dot(A, B)
naive_time = time.time() - start_time

start_time = time.time()
compiled_result = final_optimizer.execute(A_torch, B_torch)
compiled_time = time.time() - start_time

# Performance comparison
print(f"Naive Execution Time: {naive_time:.15f} sec")
print(f"Naive Result: {naive_result}")
print(f"CIMM-Optimized Execution Time: {compiled_time:.15f} sec")
print(f"CIMM-Optimized Result: {compiled_result}")