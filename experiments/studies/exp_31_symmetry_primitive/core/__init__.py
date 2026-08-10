# exp_31 core — reuses M7 symmetry infrastructure
import sys
import importlib
from pathlib import Path

# Make M7 core importable under a distinct name to avoid shadowing
M7_ROOT = Path(__file__).resolve().parent.parent.parent / "milestone7"
if str(M7_ROOT) not in sys.path:
    sys.path.insert(0, str(M7_ROOT))

# Import M7's symmetry module directly by file path
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "m7_symmetry",
    M7_ROOT / "core" / "symmetry.py"
)
_m7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m7)

# Re-export everything we need
PHI = _m7.PHI
INV_PHI = _m7.INV_PHI
LN_PHI = _m7.LN_PHI
GAMMA_EM = _m7.GAMMA_EM
XI_BALANCE = _m7.XI_BALANCE
PI = _m7.PI
PHI_FAMILY = _m7.PHI_FAMILY
get_self_referential_maps = _m7.get_self_referential_maps
get_non_self_referential_maps = _m7.get_non_self_referential_maps
is_phi_related = _m7.is_phi_related
iterate_map = _m7.iterate_map
save_results = _m7.save_results
build_ring = _m7.build_ring
build_torus = _m7.build_torus
build_random_regular = _m7.build_random_regular
graph_laplacian = _m7.graph_laplacian
global_symmetry_spectral = _m7.global_symmetry_spectral
local_asymmetry = _m7.local_asymmetry
