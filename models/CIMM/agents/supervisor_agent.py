import threading
from queue import Queue
import numpy as np
import uuid  # Added import for uuid

class SupervisorAgent(threading.Thread):
    def __init__(self, manager, agent_ids):
        super().__init__()
        self.manager = manager
        self.agent_ids = agent_ids
        self.inbox = Queue()
        self.running = True
        self.supervisor_id = "Supervisor"
        self.tick_count = 0  # Counter to track ticks for periodic actions
        self.trust_score = {agent_id: 1.0 for agent_id in agent_ids}  # Initialize trust scores
        self.agent_id = f"supervisor-{uuid.uuid4()}"  # Register the agent with a unique ID
        self.agent_name = "SupervisorAgent"
        self.collected_results = {}  
        self.running = True
        # Initialize trust weights for each agent
        self.trust_weights = {agent_id: 1.0 for agent_id in self.agent_ids}

        print(f"[Supervisor] Initialized to monitor agents: {agent_ids}")

    def send(self, message):
        self.inbox.put(message)
    def set_mesh(self, mesh):
        self.mesh = mesh
    def get_entropy_state(self):
        # Supervisors don't have entropy to share, so return a neutral or null state
        return 0.0  # Or None, or a special flag, depending on expected downstream logic
    def broadcast(self, msg_type, data):
        for agent_id in self.agent_ids:
            # Adjust message weight based on trust score
            adjusted_data = data.copy()
            adjusted_data["trust_weight"] = self.trust_score[agent_id]
            self.manager.send_to(agent_id, {
                "type": msg_type,
                "from": self.supervisor_id,
                "data": adjusted_data
            })

    def qbe_confidence(self, weights, wavefunction):
        """QBE-inspired confidence signal calculation using entropy-weight collapse."""
        # Normalize weights to represent probabilistic spread
        weights = np.array(weights)
        weights = weights / np.sum(weights + 1e-8)

        # Wavefunction modulation term
        wavefunction = np.array(wavefunction)
        wave_intensity = np.std(wavefunction)

        # Entropic balance = low spread + low oscillation
        entropy_score = -np.sum(weights * np.log(weights + 1e-8))
        coherence_penalty = wave_intensity

        # Final signal
        qbe_signal = 1.0 / (1.0 + entropy_score + coherence_penalty)
        return float(np.clip(qbe_signal, 0.0, 1.0))

    def handle_message(self, msg):
        msg_type = msg.get("type")
        if msg_type == "result":
            self.analyze_result(msg)
            from_agent_id = msg.get("from")
            result = msg.get("result")
            confidence = result[3]

            # Update trust weight based on confidence
            self.trust_weights[from_agent_id] *= 0.95 + (confidence * 0.1)
            self.trust_weights[from_agent_id] = min(max(self.trust_weights[from_agent_id], 0.1), 2.0)
        elif msg_type == "shutdown":
            print("[Supervisor] Received shutdown command.")
            self.running = False
        elif msg_type == "aggregate_predictions":
            predictions = msg["data"]  # Dict: {agent_id: (pred, confidence)}

            # Weight predictions by trust
            weighted_sum = 0.0
            total_weight = 0.0
            for agent_id, (prediction, _) in predictions.items():
                weight = self.trust_weights.get(agent_id, 1.0)
                weighted_sum += prediction * weight
                total_weight += weight
            consensus_prediction = weighted_sum / total_weight

            for agent_id, (pred, conf) in predictions.items():
                reward = max(0.0, 1 - abs(pred - consensus_prediction))  # closer to consensus = better
                correction = np.array([consensus_prediction - pred] * 4)  # basic delta for now
                self.manager.send_feedback(self.supervisor_id, agent_id, {
                    "reward_score": reward,
                    "correction": correction.tolist()
                })

            # Evaluate which agents performed well/poorly
            for agent_id, result in predictions.items():
                prediction, weights, confidence = result

                # If confidence is low or prediction is off: send feedback
                if confidence < 0.7:
                    feedback = [-0.1 * w for w in weights]  # Negative feedback
                else:
                    feedback = [0.05 * w for w in weights]  # Positive feedback

                self.manager.send_feedback(agent_id, feedback)

        elif msg["type"] == "dispatch_feedback":
            self.dispatch_feedback(self.manager)
        elif msg_type == "predict":
            print("[Supervisor] ⚠️ Received 'predict' message. Ignoring as it is not applicable.")
        else:
            print(f"[Supervisor] ⚠️ Unknown message: {msg}")

    def analyze_result(self, msg):
        agent_id = msg.get("from")
        prediction, weights, wavefunction, _ = msg.get("result")

        # 🧠 Use QBE-inspired entropy logic to recalculate confidence
        dynamic_conf = self.qbe_confidence(weights, wavefunction)

        print(f"[Supervisor] 🔬 Dynamic QBE-Confidence for {agent_id}: {dynamic_conf:.4f}")

        # Update trust score based on confidence
        self.trust_score[agent_id] = 0.9 * self.trust_score[agent_id] + 0.1 * dynamic_conf
        print(f"[Supervisor] 🤝 Updated trust score for {agent_id}: {self.trust_score[agent_id]:.4f}")

        # Reinforce if confidence is too low
        if dynamic_conf < 0.6:
            self.manager.send_to(agent_id, {
                "type": "reinforce",
                "from": self.supervisor_id,
                "data": {
                    "feedback": "adjust_weights",
                    "meta": {"confidence": dynamic_conf}
                }
            })

    def dispatch_feedback(self, manager):
        for agent_id, result in self.collected_results.items():
            prediction, weights, entropy, confidence = result

            if confidence < 0.7:
                feedback = [-0.1 * w for w in weights]
            else:
                feedback = [0.05 * w for w in weights]

            manager.send_feedback(agent_id, feedback)

    def handle_message(self, msg):
        if msg.get("type") == "shutdown":
            self.running = False
            print("[Supervisor] Shutting down supervisor thread.")

    def run(self):
        print("[Supervisor] 👁️ Monitoring agent mesh.")
        while self.running:
            try:
                msg = self.inbox.get(timeout=1)
                self.handle_message(msg)
            except:
                pass  # Timeout, continue to next iteration

            # Increment tick count and trigger periodic entropy sync every 5 ticks
            self.tick_count += 1
            if self.tick_count % 5 == 0:
                print("[Supervisor] 🔄 Synchronizing entropy across agents.")
                self.manager.synchronize_entropy()

                # 🔍 Log top agents by trust
                top_agents = sorted(self.trust_score.items(), key=lambda x: -x[1])[:3]
                print(f"[Supervisor] 🎯 Top trusted agents:")
                for agent_id, score in top_agents:
                    print(f"  - {agent_id[:8]}...: Trust = {score:.3f}")

        print("[Supervisor] 🛑 Supervisor shutting down.")
