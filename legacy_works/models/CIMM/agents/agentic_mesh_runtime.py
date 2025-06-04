from supervisor_agent import SupervisorAgent  # Import your new agent

# STEP 3 — Build a mesh runtime to route messages between agents

class AgenticMeshRuntime:
    def __init__(self, agents, manager, supervisor):  # Add manager as a parameter
        self.agents = {agent.agent_id: agent for agent in agents}
        self.manager = manager  # Store manager reference
        self.routing = {}  # from_agent_id: [to_agent_id]
        self.agent_map = {agent.agent_name: agent for agent in agents}  # Initialize agent_map

        # Initialize SupervisorAgent
        agent_ids = [agent.agent_id for agent in agents]
        self.supervisor = supervisor

    def connect(self, from_agent, to_agent):
        self.routing.setdefault(from_agent.agent_id, []).append(to_agent.agent_id)

    def route(self, from_agent_id, message):
        targets = self.routing.get(from_agent_id, [])
        for agent_id, agent in self.agents.items():
            if agent_id in targets:
                agent.send(message)

    def send_to(self, recipient_id, message):

        if recipient_id in self.agents.keys():
            self.agents[recipient_id].inbox.put(message)
        else:
            print(f"[Mesh] ⚠️ Unknown recipient: {recipient_id}")


    # Optional: route agent messages to supervisor (if not already inside manager logic)
    def route_results_to_supervisor(self, msg):
        if msg["type"] == "result":
            self.supervisor.send(msg)