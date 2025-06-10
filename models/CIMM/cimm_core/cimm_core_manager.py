from cimm_core.cimm_core import CIMMCoreWrapper
from usecase.prime_structure_usecase import PrimeStructureUseCase
from usecase.stock_prediction_usecase import StockPredictionUseCase
from skopt.space import Real, Integer

class CIMMCoreManager:
    """
    Manages multiple agents, each with its own independent CIMMCore instance.
    """

    def __init__(self):
        self.agents = {}
        self.agent_roles = {}         # Tracks agent roles
        self.agent_supervision = {}   # Tracks who supervises whom
        self.supervisor_id = None     # Track system supervisor

    def register_supervisor(self, supervisor):
        """Register the SupervisorAgent manually into the agent registry."""
        self.agents[supervisor.agent_id] = supervisor
        print(f"Supervisor Agent Registered: ID: {supervisor.agent_id}")
        self.supervisor_id = supervisor.agent_id

    # ----------------------------
    # 📌 Register a New Agent
    # ----------------------------
    def register_agent(self, agent_name: str, model_class, param_space, anchor_data, hidden_size=64, role="worker", supervisor=None):
        """
        Creates a new agent, assigning it a specialized CIMMCore instance.
        """
        agent = CIMMCoreWrapper(model_class, param_space, anchor_data, hidden_size)  # FIX: Removed unnecessary params
        self.agents[agent.agent_id] = agent
        self.agent_roles[agent.agent_id] = role
        self.agent_supervision[agent.agent_id] = supervisor
        print(f"Registered Agent: {agent_name} | ID: {agent.agent_id}")
        return agent.agent_id

    # ----------------------------
    # 📌 Get Agent Instance
    # ----------------------------
    def get_agent(self, agent_id: str):
        """
        Retrieves an agent instance by its ID.
        """
        if agent_id not in self.agents:
            raise ValueError("Agent not found")
        return self.agents[agent_id]

    # ----------------------------
    # 📌 List All Agents
    # ----------------------------
    def list_agents(self):
        """
        Returns a list of all registered agent IDs.
        """
        return list(self.agents.keys())
    
    def collective_decision(self, agent_ids, input_data):
        """
        Runs decision-making across multiple agents and picks the most stable response.
        """
        lowest_entropy = float("inf")
        best_agent = None

        for agent_id in agent_ids:
            agent = self.get_agent(agent_id)
            entropy = agent.query_entropy(input_data)
            
            if entropy < lowest_entropy:
                lowest_entropy = entropy
                best_agent = agent

        return best_agent.run(input_data) if best_agent else None
    
    def share_entropy_state(self, sender_agent_id, receiver_agent_id):
        """
        Allows one agent to share its entropy state with another agent.
        """
        if sender_agent_id in self.agents and receiver_agent_id in self.agents:
            sender_entropy = self.agents[sender_agent_id].retrieve_quantum_state()
            self.agents[receiver_agent_id].store_quantum_state(sender_entropy, qbe_feedback=0.002)
            return f"Entropy state transferred from {sender_agent_id} to {receiver_agent_id}"
        return "Agent not found."
    
    def synchronize_entropy(self, threshold=0.05):
        """
        Automatically synchronizes entropy states across all agents.
        The agent with the lowest entropy state is selected as the primary knowledge source.
        Other agents adjust their entropy states to match.
        """
        if not self.agents:
            return "No agents available for synchronization."

        # Gather entropy levels from all agents
        entropy_levels = {
            agent_id: self.agents[agent_id].query_entropy([]) for agent_id in self.agents
        }

        # Find the agent with the most stable knowledge (lowest entropy)
        most_stable_agent = min(entropy_levels, key=entropy_levels.get)
        stable_entropy = entropy_levels[most_stable_agent]

        # Share its entropy state with all other agents, if necessary
        for agent_id, entropy_value in entropy_levels.items():
            if agent_id != most_stable_agent and abs(entropy_value - stable_entropy) > threshold:
                self.share_entropy_state(most_stable_agent, agent_id)

        return f"Entropy synchronization complete. Agent {most_stable_agent} was used as the reference."

    def synchronize_entropy(self):
        """
        Synchronizes entropy states across all agents by averaging their entropy states.
        Skips agents that do not have the set_entropy_state method.
        """
        entropies = [agent.get_entropy_state() for agent in self.agents.values()]
        avg_entropy = sum(entropies) / len(entropies)
        
        for agent in self.agents.values():
            if hasattr(agent, "set_entropy_state"):  # Skip agents without this method
                agent.set_entropy_state(avg_entropy)

    def get_supervisor_id(self):
        for agent_id, agent in self.agents.items():
            if agent.__class__.__name__ == "SupervisorAgent":
                return agent_id
        return None

    def send_feedback(self, sender_agent_id, target_agent_id, correction_vector, reward_score):
        """
        Sends feedback from one agent to another.
        """
        if sender_agent_id in self.agents and target_agent_id in self.agents:
            feedback_data = {
                "type": "feedback",
                "data": {
                    "correction": correction_vector,
                    "reward": reward_score
                }
            }
            self.agents[target_agent_id].receive_message(feedback_data)
            return f"Feedback sent from {sender_agent_id} to {target_agent_id}"
        return "Agent not found."

    def send_feedback(self, agent_id, feedback_data):
        if agent_id in self.agent_threads:
            self.agent_threads[agent_id].put({
                "type": "feedback",
                "data": feedback_data
            })

    def broadcast(self, message, role_filter=None):
        """
        Sends a message to all agents or filters by role.
        """
        for agent_id, agent in self.agents.items():
            if role_filter is None or self.agent_roles.get(agent_id) == role_filter:
                agent.send(message)

    def send_result_to_supervisor(self, from_agent_id, result):
        """
        Sends prediction results to the supervisor agent.
        """
        if self.supervisor_id:
            self.agents[self.supervisor_id].send({
                "type": "result",
                "from": from_agent_id,
                "result": result
            })

    def apply_global_feedback(self, correction_vector, reward_score):
        """
        Broadcasts a correction vector and reward score to all agents.
        """
        for agent_id in self.agents:
            self.send_feedback(self.supervisor_id, agent_id, correction_vector, reward_score)

    def get_all_entropy_states(self):
        """
        Retrieves the entropy states of all agents.
        """
        return {agent_id: agent.get_entropy_state() for agent_id, agent in self.agents.items()}

