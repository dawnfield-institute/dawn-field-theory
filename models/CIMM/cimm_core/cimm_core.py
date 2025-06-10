import torch
import numpy as np
import uuid
from cimm_core.cimm import CIMM  # Import the base CIMM model

class CIMMCoreWrapper:
    """
    A wrapper for individual CIMM instances, allowing each agent to have its own model.
    """

    def __init__(self, model_class, param_space, anchor_data, hidden_size=64):
        """
        Initializes an independent CIMMCore instance.
        """
        self.model_class = model_class
        self.hidden_size = hidden_size
        self.learning_rate = 0.01

        # FIX: Ensure model receives the correct arguments
        if model_class.__init__.__code__.co_argcount == 2:  # Only (self, hidden_size)
            self.cimm_instance = CIMM(lambda: self.model_class(self.hidden_size), param_space, anchor_data)
        else:  # Models that require (input_size, hidden_size, output_size)
            self.cimm_instance = CIMM(lambda: self.model_class(4, self.hidden_size, 1), param_space, anchor_data)

        #Expose entropy monitor directly
        self.entropy_monitor = self.cimm_instance.entropy_monitor
        self.agent_id = str(uuid.uuid4())

    def run(self, input_data):
        """ Forwards the `run()` call to the CIMM instance """
        return self.cimm_instance.run(input_data)  # FIX: Provide direct access to `run()`

    def give_feedback(self, data_point, actual_value):
        """ Forwards the `give_feedback()` call to the CIMM instance """
        return self.cimm_instance.give_feedback(data_point, actual_value)  # FIX: Provide direct access to `run()`

    def update_entropy_state(self, new_entropy_state):
        """
        Allows an agent to update its entropy state based on external knowledge transfer.
        """
        self.store_quantum_state(new_entropy_state, qbe_feedback=0.002)

    def evaluate_model(self, validation_data):
        """ Forwards the `evaluate_model()` call to the CIMM instance """
        return self.cimm_instance.evaluate_model(self.cimm_instance.model_instance, validation_data)

    def query_entropy(self, state_data):
        return self.cimm_instance.compute_error_metrics(torch.tensor(state_data, dtype=torch.float32))

    def reinforcement_update(self, reward_signal, entropy_level):
        new_lr = max(0.001, min(0.05, self.learning_rate * (1 + 0.1 * reward_signal - 0.05 * entropy_level)))
        self.learning_rate = new_lr
        return new_lr

    def store_quantum_state(self, entropy_history, qbe_feedback):
        self.entropy_history = entropy_history
        return "state_stored"

    def retrieve_quantum_state(self):
        return self.entropy_history

    def get_info(self):
        return {
            "agent_id": self.agent_id,
            "learning_rate": self.learning_rate,
            "entropy_history": self.entropy_history,
        }

    def get_entropy_state(self):
        """
        Retrieves the current entropy state of the agent.
        """
        return self.entropy_monitor.get_current_entropy()

    def set_entropy_state(self, entropy_value):
        """
        Sets the entropy state of the agent.
        """
        if hasattr(self, "entropy_monitor") and self.entropy_monitor:
            self.entropy_monitor.set_entropy(entropy_value)