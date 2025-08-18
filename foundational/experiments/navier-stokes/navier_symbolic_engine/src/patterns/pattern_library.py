"""Pattern management and storage."""


from .laminar_patterns import LaminarPatterns
from .turbulent_patterns import TurbulentPatterns
from .transitional_patterns import TransitionalPatterns

class PatternLibrary:
    """
    Manages and stores all pattern templates for symbolic navigation.
    """
    def __init__(self):
        self.laminar = LaminarPatterns()
        self.turbulent = TurbulentPatterns()
        self.transitional = TransitionalPatterns()
