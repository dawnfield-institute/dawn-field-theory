import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from cimm_core.cimm_core_manager import CIMMCoreManager
from cimm_core.cimm import CIMM
from cimm_core.models.base_cimm_model import BaseCIMMModel
from cimm_core.entropy.entropy_monitor import EntropyMonitor
from cimm_core.learning.reinforcement_learning import QBEReinforcementLearner
from agents.base_agent import BaseAgent
from skopt.space import Real, Integer
from astroquery.sdss import SDSS  # Import astrophysical data
import logging
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)

class DarkMatterAgent(BaseAgent):
    """
    Agent for predicting dark matter distributions using entropy-balancing intelligence.
    """
    def __init__(self, manager, hidden_size=64):
        self.manager = manager
        self.hidden_size = hidden_size
        
        self.raw_data = self.fetch_dark_matter_data()
        self.processed_data = self.preprocess_dark_matter_data(self.raw_data)
        self.processed_data = self.processed_data[:, :9]  # Select first 9 columns
        anchor_size = max(1, int(0.1 * len(self.processed_data))) 
        anchor_data = self.processed_data[:anchor_size].clone().detach()
        self.streaming_data = self.processed_data[anchor_size:]
        print(f"Processed Data Shape: {self.processed_data.shape}")
        print(f"Anchor Size: {anchor_size}")
        print(f"Anchor Data Shape: {anchor_data.shape}")
        print(f"Streaming Data Shape: {self.streaming_data.shape}")
        param_space = [
            Real(1e-5, 1e-1, name='learning_rate'),
            Integer(10, 100, name='hidden_size'),
        ]
        
        super().__init__("DarkMatterAI", DarkMatterModel, param_space, manager, anchor_data, hidden_size)
        logging.info("Dark Matter Agent Initialized.")

    def fetch_dark_matter_data(self):
        """
        Retrieves real dark matter dataset while handling SDSS API inconsistencies.
        """
        logging.info("Fetching expanded dark matter dataset from SDSS...")

        try:
            query = """
                SELECT TOP 1000 p.objID, p.ra, p.dec, p.z, 
                    COALESCE(s.velDisp, -1) AS velDisp,  -- Use -1 if NULL
                    COALESCE(s.velDispErr, -1) AS velDispErr,
                    COALESCE(s.spectroFlux_r, -1) AS spectroFlux_r,
                    COALESCE(s.spectroFlux_g, -1) AS spectroFlux_g,
                    COALESCE(s.spectroFlux_i, -1) AS spectroFlux_i,
                    COALESCE(s.spectroFlux_z, -1) AS spectroFlux_z,
                    p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, p.psfMag_z
                FROM PhotoObj AS p
                LEFT JOIN SpecObj AS s ON p.objID = s.bestObjID
                WHERE p.z BETWEEN 0 AND 10
            """

            result = SDSS.query_sql(query)

            # ✅ Fix: Handle cases where SDSS returns an error instead of data
            if result is None or len(result) == 0:
                raise ValueError("No valid data retrieved from SDSS.")

            # ✅ Fix: Ensure data is properly formatted as a table
            expected_columns = {"objID", "ra", "dec", "z", 
                                "velDisp", "velDispErr", 
                                "spectroFlux_r", "spectroFlux_g", "spectroFlux_i", "spectroFlux_z",
                                "psfMag_u", "psfMag_g", "psfMag_r", "psfMag_i", "psfMag_z"}
            result_columns = set(result.columns)

            if not expected_columns.issubset(result_columns):
                raise ValueError(f"Unexpected column format. Retrieved columns: {result_columns}")

            # Convert result to numpy array
            data = np.array([result["ra"], result["dec"], result["z"], 
                            result["velDisp"], result["velDispErr"], 
                            result["spectroFlux_r"], result["spectroFlux_g"], result["spectroFlux_i"], 
                            result["spectroFlux_z"], result["psfMag_u"], result["psfMag_g"], result["psfMag_r"], 
                            result["psfMag_i"], result["psfMag_z"]]).T

            return torch.tensor(data, dtype=torch.float32)

        except Exception as e:
            logging.error(f"SDSS query failed: {e}")
            logging.info("Using fallback synthetic dataset.")

            # Generate fallback data with correct shape
            fallback_data = np.random.uniform(low=0, high=360, size=(5000, 10))
            return torch.tensor(fallback_data, dtype=torch.float32)

    def preprocess_dark_matter_data(self, raw_data):
        logging.info("Applying advanced preprocessing...")

        # Normalize raw data
        raw_mean = raw_data.mean(dim=0)
        raw_std = raw_data.std(dim=0)
        raw_std[raw_std == 0] = 1e-8  # Prevent division by zero
        raw_data = (raw_data - raw_mean) / raw_std

        # Apply Gaussian smoothing
        smoothed_data = gaussian_filter(raw_data.numpy(), sigma=1.5)

        # Apply Fourier Transform for frequency-domain features
        frequency_features = np.abs(fft(smoothed_data, axis=0))

        # Ensure Data is 2D for PCA
        smoothed_data = np.atleast_2d(smoothed_data)

        # ✅ Fix: Dynamically set `n_components` based on available features
        num_features = smoothed_data.shape[1]
        num_components = min(2, num_features)  # Ensure PCA has enough features

        if num_components > 1:
            pca = PCA(n_components=num_components)
            pca_features = pca.fit_transform(smoothed_data)
        else:
            logging.warning(f"Not enough features for PCA. Skipping PCA transformation.")
            pca_features = np.zeros((smoothed_data.shape[0], 2))  # Placeholder zeros

        # Ensure feature consistency
        num_samples = smoothed_data.shape[0]
        frequency_features = np.resize(frequency_features, (num_samples, min(3, frequency_features.shape[1])))
        pca_features = np.resize(pca_features, (num_samples, 2))

        # ✅ Fix: Ensure all feature arrays have the same number of rows
        combined_features = np.hstack([smoothed_data, frequency_features, pca_features])

        # Convert to tensor
        processed_data = torch.tensor(combined_features, dtype=torch.float32)

        return processed_data

    def calculate_entropy(self, data_point):
        """Compute entropy using a dynamic adjustment instead of a static method."""
        probability_distribution = torch.abs(data_point) / torch.sum(torch.abs(data_point) + 1e-9)
        entropy = -torch.sum(probability_distribution * torch.log(probability_distribution + 1e-9))
        
        # Normalize entropy within a dynamic range
        entropy = entropy / (torch.max(entropy) + 1e-9)

        return entropy.item()
    
    def predict_dark_matter_distribution(self):
        """
        Runs AI-driven dark matter prediction based on entropy balancing.
        """
        predictions = []
        entropy_values = []

        for i, new_data_point in enumerate(self.streaming_data[:-1]):  # Ensure we have a next value
            prediction, __, ____ = self.agent_instance.run(new_data_point)
            predictions.append(prediction)

            if i >= len(self.streaming_data):
                break

            # ✅ Use the next data point as ground truth
            actual_value = self.streaming_data[i + 1]

            # ✅ Fix: Ensure `actual_value` is a scalar before passing to `give_feedback`
            if isinstance(actual_value, torch.Tensor):
                actual_value = actual_value.item() if actual_value.numel() == 1 else float(actual_value.mean())  # Ensure scalar
            elif isinstance(actual_value, (list, np.ndarray)):
                actual_value = float(actual_value[0]) if len(actual_value) == 1 else float(np.mean(actual_value))  # Ensure scalar

            entropy = self.calculate_entropy(new_data_point)
            entropy_values.append(entropy)
            logging.info(f"Data Point {i}: Entropy = {entropy}")
            logging.info(f"Actual Value {i}: {actual_value}")

            # ✅ Provide feedback to model
            self.agent_instance.give_feedback(new_data_point, actual_value)

            if abs(entropy - self.agent_instance.entropy_monitor.prev_entropy) > 0.02:
                logging.info("Updating model due to significant entropy change.")
                self.agent_instance.reinforcement_update(reward_signal=-entropy, entropy_level=entropy)

            self.agent_instance.entropy_monitor.prev_entropy = entropy

        self.plot_live_predictions(predictions, self.streaming_data[1:])  # Adjust plotting to match new ground truth
        return predictions


    def plot_live_predictions(self, predictions, actuals):

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(actuals)), actuals, label="Actual Values", color="blue", alpha=0.3)
        plt.plot(range(len(predictions)), predictions, color="red", linewidth=3, linestyle="-")

        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.title("Live Predictions vs. Actual Values")
        plt.legend()
        plt.grid(True)
        plt.show()


class DarkMatterModel(BaseCIMMModel):
    """
    AI model for predicting dark matter entropy structures.
    """
    def __init__(self, hidden_size):
        super(DarkMatterModel, self).__init__(input_size=9, hidden_size=hidden_size, output_size=1)
    
    def forward(self, x):
        return self.common_forward(x)

if __name__ == "__main__":
    manager = CIMMCoreManager()
    dark_matter_agent = DarkMatterAgent(manager)
    predictions = dark_matter_agent.predict_dark_matter_distribution()
    metrics = dark_matter_agent.agent_instance.evaluate_model(dark_matter_agent.streaming_data)
    print(f"Error Metrics: {metrics}")