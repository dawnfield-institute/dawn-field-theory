"""
Domain adapters for the Unified Emergence Framework v2.
"""

from .gravity_adapter import GravityDomainAdapter
from .med_adapter import MEDDomainAdapter
from .navier_adapter import NavierDomainAdapter
from .tinycimm_adapter import TinyCIMMDomainAdapter
from .hodge_adapter import HodgeDomainAdapter

__all__ = [
    'GravityDomainAdapter',
    'MEDDomainAdapter',
    'NavierDomainAdapter', 
    'TinyCIMMDomainAdapter',
    'HodgeDomainAdapter'
]
