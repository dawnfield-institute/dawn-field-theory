import threading
import queue
import time
import numpy as np
import os
import json
from cimm_core.cimm_core import CIMMCoreWrapper
from skopt.space import Real, Integer
from collections import deque
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

class BaseAgent(threading.Thread):
    """
    A standardized base class for agents using CIMM intelligence, now threaded.
    """

    def __init__(self, agent_name, model_class, param_space, manager, anchor_data, hidden_size=64, role="predictor", mesh=None):
        super().__init__()
        # Removed daemon=True for graceful shutdown
        self.running = False
        self.inbox = queue.Queue()
        self.outbox = queue.Queue()
        self.agent_name = agent_name

        self.manager = manager
        self.agent_id = self.manager.register_agent(agent_name, model_class, param_space, anchor_data, hidden_size)
        self.agent_instance = self.manager.get_agent(self.agent_id)
        self.role = role
        self.mesh = mesh  # Injected at init
        self.memory = deque(maxlen=100)  # stores last 100 prediction outcomes
        self.accuracy_history = []  # Track moving accuracy over time
        self.model = model_class(input_size=4, hidden_size=hidden_size, output_size=1).to(device)  # Move model to device

    def save_memory(self):
        if hasattr(self, "memory"):
            filename = f"memory_{self.agent_name}_{self.agent_id}.json"
            with open(filename, "w") as f:
                json.dump(list(self.memory), f)  # Convert deque to list for JSON serialization
            print(f"[{self.agent_name}] 💾 Memory saved to {filename}")

    def load_memory(self):
        filename = f"memory_{self.agent_name}_{self.agent_id}.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.memory = deque(json.load(f), maxlen=100)  # Load as deque with maxlen
            print(f"[{self.agent_name}] 🔁 Memory loaded from {filename}")
        else:
            self.memory = deque(maxlen=100)  # Initialize empty deque if no file exists

    def run(self):
        self.load_memory()  # Load memory at the start
        self.running = True
        print(f"[{self.agent_name}] Node running in thread.")
        while self.running:
            try:
                msg = self.inbox.get(timeout=0.5)
                self.handle_message(msg)
            except queue.Empty:
                continue  # No messages, just wait
        self.save_memory()  # Save memory before shutting down

    def set_mesh(self, mesh):
        self.mesh = mesh

    def handle_message(self, msg):
        print(f"[{self.agent_name}] Received message: {msg}")  # Debug log for received messages
        msg_type = msg.get("type")  # ✅ Safely get the type or None

        if msg_type == "predict":
            result = self.agent_instance.run(msg["data"])

            # If result is a tuple, unpack it
            if isinstance(result, tuple):
                prediction, prob_vector, adjusted_output, confidence = result

                # Memory-based trend adjustment
                if len(self.memory) > 0:
                    recent_trend = np.mean([m["prediction"] for m in self.memory if "prediction" in m])
                    prediction += 0.1 * (recent_trend - prediction)

                # Repack the result with updated prediction
                result = (prediction, prob_vector, adjusted_output, confidence)

            # Optionally incorporate memory context into prediction if result is a dict
            elif isinstance(result, dict) and len(self.memory) > 0:
                recent_trend = np.mean([m["prediction"] for m in self.memory if "prediction" in m])
                result["prediction"] += 0.1 * (recent_trend - result["prediction"])

            self.outbox.put({"from": self.agent_id, "type": "result", "result": result})

            # Send result to Supervisor via mesh
            self.mesh.send_to(self.manager.supervisor_id, {
                "type": "result",
                "from": self.agent_id,
                "result": result
            })

        elif msg_type == "feedback":
            self.receive_feedback(msg["data"])

        elif msg_type == "forward_to":
            self.outbox.put({
                "type": "predict",
                "data": msg["payload"]
            })

        elif msg_type == "result":
            if self.role == "supervisor":
                self.agent_instance.observe(msg["from"], msg["result"])
            else:
                print(f"[{self.agent_name}] 🔄 Received result from another agent: {msg}")
                result = msg["result"]

                # Default values
                prediction = 0.0
                confidence = 0.0

                if isinstance(result, tuple):
                    prediction, prob_vector, adjusted_output, confidence = result
                    if self.memory:
                        recent_trend = np.mean([m["prediction"] for m in self.memory if "prediction" in m])
                        prediction += 0.1 * (recent_trend - prediction)
                    result = (prediction, prob_vector, adjusted_output, confidence)

                elif isinstance(result, dict):
                    prediction = result.get("prediction", 0.0)
                    confidence = result.get("confidence", 0.0)
                    if self.memory:
                        recent_trend = np.mean([m["prediction"] for m in self.memory if "prediction" in m])
                        prediction += 0.1 * (recent_trend - prediction)
                    result["prediction"] = prediction

                msg["result"] = result

                self.memory.append({
                    "input": msg.get("input"),
                    "prediction": prediction,
                    "confidence": confidence,
                    "timestamp": time.time()
                })

                consensus = result[2][-1]  # Assuming last element = consensus, tweak if needed
                error = abs(prediction - consensus)
                accuracy = max(0.0, 1.0 - error)
                self.accuracy_history.append(accuracy)

                if len(self.accuracy_history) > 20:
                    self.accuracy_history.pop(0)  # Keep moving window

                moving_avg = sum(self.accuracy_history) / len(self.accuracy_history)
                print(f"[{self.agent_name}] 📈 Moving Avg Accuracy: {moving_avg:.3f}")

        elif msg_type == "shutdown":
            self.running = False
            print(f"[{self.agent_name}] Node shutting down.")

        else:
            print(f"[{self.agent_name}] ⚠️ Unknown or missing message type: {msg}")

    def send(self, data):
        self.inbox.put(data)
        input_tensor = self.process_input(data)  # Wrap and move input to device

    def receive(self):
        try:
            return self.outbox.get_nowait()
        except queue.Empty:
            return None

    def query_entropy(self, state_data):
        return self.agent_instance.query_entropy(state_data)

    def reinforcement_update(self, reward_signal, entropy_level):
        return self.agent_instance.reinforcement_update(reward_signal, entropy_level)

    def store_quantum_state(self, entropy_history, qbe_feedback):
        return self.agent_instance.store_quantum_state(entropy_history, qbe_feedback)

    def retrieve_quantum_state(self):
        return self.agent_instance.retrieve_quantum_state()

    def get_info(self):
        return self.agent_instance.get_info()
    

    def receive_feedback(self, feedback_data):
        if hasattr(self.agent_instance, "apply_feedback"):
            self.agent_instance.apply_feedback(feedback_data)
        else:
            self.apply_feedback(feedback_data)

    def apply_feedback(self, feedback_data):
        """
        Hook for applying feedback. Override this in subclasses if needed.
        """
        print(f"[{self.agent_name}] Applying feedback: {feedback_data}")

    def process_input(self, data):
        """
        Wraps input data into a tensor and moves it to the appropriate device.
        """
        input_tensor = torch.tensor(data, dtype=torch.float32).to(device)  # Ensure input is on device
        return input_tensor