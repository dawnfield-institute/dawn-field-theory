"""
Fracton Tools Package - Tool expression framework (placeholder)

This module will provide the tool expression framework for external system
integration. Currently implemented as a stub for the core language development.
"""

class ToolRegistry:
    """Placeholder tool registry."""
    
    def __init__(self):
        self._tools = {}
    
    def get_tool(self, name: str):
        """Get tool by name."""
        return self._tools.get(name)
    
    def register_tool(self, name: str, tool):
        """Register a tool."""
        self._tools[name] = tool


_global_registry = ToolRegistry()


def get_tool_registry():
    """Get the global tool registry."""
    return _global_registry
