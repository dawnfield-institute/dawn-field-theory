# run_trading_sim.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
import numpy as np

from momentum_agent import MomentumAgent
from agents.volatility_agent import VolatilityAgent
from agents.trend_agent import TrendAgent
from cimm_core.cimm_core_manager import CIMMCoreManager
from agentic_mesh_runtime import AgenticMeshRuntime
from supervisor_agent import SupervisorAgent
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

# 🧬 Generate fake market data (simulate live stream)
def generate_market_data(n=100, features=10):
    return torch.randn(n, features, device=device)  # Specify device for tensor

# 🧵 Spin up agent nodes
def initialize_agents():
    print("🔧 Initializing Agents...")
    manager = CIMMCoreManager()

    agents = [
        MomentumAgent(manager),
        VolatilityAgent(manager),
        TrendAgent(manager)
    ]

    # Collect agent IDs after registration
    agent_ids = list(manager.agents.keys())

    # Create list of all predictor agent IDs
    predictor_ids = [agent.agent_id for agent in agents if agent.role == "predictor"]

    # Initialize Supervisor
    supervisor = SupervisorAgent(manager, agent_ids=predictor_ids)
    manager.register_supervisor(supervisor)  # Register SupervisorAgent manually
    agents.append(supervisor)  # Add SupervisorAgent last

    # Add no-op receive method to SupervisorAgent
    def receive(self):
        return None  # Supervisor doesn't do prediction returns
    setattr(SupervisorAgent, "receive", receive)

    # Initialize AgenticMeshRuntime and connect agents
    mesh = AgenticMeshRuntime(agents, manager, supervisor)  # Pass manager as an argument
    mesh.connect(agents[0], agents[2])  # Example: Connect MomentumAgent to TrendAgent
    
    for agent in agents:
        agent.mesh = mesh
        agent.start()
        print(f"✅ {agent.agent_name} started.")

    return manager, agents, mesh, supervisor

# 🧠 Simulate trading decisions from all agents
def simulate_trading_loop(agents, supervisor, mesh, num_ticks=10):
    print("🚀 Simulating agentic trading...")
    for tick in range(num_ticks):
        print(f"\n⏱️ Tick {tick + 1}")
        new_market_data = torch.randn(4, device=device)  # Specify device for tensor

        for agent in agents:
            input_tensor = torch.tensor(new_market_data, dtype=torch.float32, device=device)  # Specify device
            agent.send({"type": "predict", "data": input_tensor})  # Send input tensor

        time.sleep(1.0)  # Allow agents to process
        # Every 5 ticks, synchronize entropy
        if tick % 5 == 0:
            entropies = [
                agent.get_entropy_state()
                for agent in manager.agents.values()
                if hasattr(agent, "get_entropy_state") and callable(agent.get_entropy_state)
            ]

            avg_entropy = np.mean(entropies)
            for agent in manager.agents.values():
                if hasattr(agent, "set_entropy_state") and callable(agent.set_entropy_state):
                    agent.set_entropy_state(avg_entropy)
        for agent in agents:
            if isinstance(agent, SupervisorAgent):
                continue
            result = agent.receive()
            if result:
                pred, probs, alts, confidence = result["result"]
                print(f"📊 [{agent.agent_name}] → Prediction: {round(pred, 4)} | Confidence: {round(confidence, 4)}")

                # Route prediction result to connected agents
                mesh.route(agent.agent_id, result)

        # Dispatch global feedback from the Supervisor
        supervisor.dispatch_feedback(manager)

# ✅ Main Entry
if __name__ == "__main__":
    manager, agents, mesh, supervisor = initialize_agents()
    for agent in agents:
        agent.set_mesh(mesh)

    supervisor.set_mesh(mesh) 
    simulate_trading_loop(agents, supervisor, mesh, num_ticks=10)

    print("\n🛑 Shutting down...")
    for agent in agents:
        agent.send({"type": "shutdown"})


    # Dispatch feedback to the supervisor
    supervisor_id = manager.supervisor_id  # implement this if needed
    mesh.send_to(supervisor_id, {
        "type": "dispatch_feedback"
    })
    supervisor.send({"type": "shutdown"})
    supervisor.join(timeout=3)  # Gracefully waits for thread to exit
    print("✅ All agents shut down.")

