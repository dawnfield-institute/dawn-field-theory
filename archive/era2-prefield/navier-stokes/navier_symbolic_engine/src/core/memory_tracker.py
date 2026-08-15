"""Memory and reversibility tracking."""


from typing import List, Any

class MemoryTracker:
    """
    Tracks navigation history and enables reversibility in symbolic pattern navigation.
    Maintains ancestry and supports undo/redo of navigation steps.
    """
    def __init__(self):
        self.history: List[Any] = []
        self.current_index: int = -1

    def record(self, node: Any):
        """
        Record a navigation step.
        """
        self.history = self.history[:self.current_index+1]
        self.history.append(node)
        self.current_index += 1

    def undo(self) -> Any:
        """
        Undo the last navigation step.
        """
        if self.current_index > 0:
            self.current_index -= 1
            return self.history[self.current_index]
        return None

    def redo(self) -> Any:
        """
        Redo the next navigation step.
        """
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            return self.history[self.current_index]
        return None

    def get_trace(self) -> List[Any]:
        """
        Get the full navigation trace up to the current point.
        """
        return self.history[:self.current_index+1]
