"""
Protocol definitions for the Unified Emergence Framework v2.
"""

from typing import Protocol, List, Dict, Any, TYPE_CHECKING
from .models import EmergenceSignature

# Import model classes for type hints only
if TYPE_CHECKING:
    from .models import EmergenceResults, CorrelationMatrix, ValidationMetrics


class DomainAdapter(Protocol):
    """
    Protocol defining the interface that all domain adapters must implement.
    
    This protocol ensures consistent interaction patterns between the framework
    and domain-specific analysis code.
    """
    
    @property
    def domain_name(self) -> str:
        """Return the name of the domain this adapter handles."""
        ...
    
    def extract_patterns(self, domain_results: Dict[str, Any]) -> List[EmergenceSignature]:
        """
        Extract emergence patterns from domain-specific results.
        
        Args:
            domain_results: Raw results from domain analysis
            
        Returns:
            List of emergence signatures representing detected patterns
        """
        ...
    
    def validate_constraints(self, domain_data: Dict[str, Any]) -> List[str]:
        """
        Validate domain-specific constraints and return any violations.
        
        Args:
            domain_data: Domain-specific data to validate
            
        Returns:
            List of constraint violation messages (empty if all constraints satisfied)
        """
        ...


class TestRunner(Protocol):
    """
    Protocol for running domain-specific tests and analyses.
    
    This abstracts the actual test execution so adapters can work with
    different test runners or mock implementations.
    """
    
    def run_domain_tests(self, domain: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run tests for a specific domain with given configuration.
        
        Args:
            domain: Name of the domain to test
            config: Domain-specific configuration
            
        Returns:
            Raw test results from the domain
        """
        ...


class ResultsRepository(Protocol):
    """
    Protocol for storing and retrieving validation results.
    """
    
    def save_results(self, results: 'EmergenceResults') -> str:
        """
        Save validation results.
        
        Args:
            results: Complete validation results to save
            
        Returns:
            Path or identifier where results were saved
        """
        ...
    
    def load_results(self, session_id: str) -> 'EmergenceResults':
        """
        Load validation results by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Loaded validation results, or None if not found
        """
        ...
    
    def list_sessions(self) -> List[str]:
        """
        List all available session IDs.
        
        Returns:
            List of session IDs
        """
        ...


class PatternAnalyzer(Protocol):
    """
    Protocol for analyzing patterns and calculating correlations.
    """
    
    def calculate_correlations(self, signatures: List[EmergenceSignature]) -> 'CorrelationMatrix':
        """
        Calculate correlations between emergence patterns.
        
        Args:
            signatures: List of emergence signatures to analyze
            
        Returns:
            Correlation matrix showing relationships between patterns
        """
        ...
    
    def calculate_metrics(self, signatures: List[EmergenceSignature], 
                         correlation_matrix: 'CorrelationMatrix') -> 'ValidationMetrics':
        """
        Calculate comprehensive validation metrics.
        
        Args:
            signatures: List of emergence signatures
            correlation_matrix: Correlation analysis results
            
        Returns:
            Comprehensive validation metrics
        """
        ...
