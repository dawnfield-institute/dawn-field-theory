"""
TinyCIMM-Navier Live CIMM Experiment Runner

Progressive validation of fluid dynamics learning using True CIMM Architecture.
Validates live pattern crystallization, entropy-driven insights, and real-time adaptation.

Experimental progression:
1. Live pattern discovery (no training loops)
2. Entropy collapse detection (flow insights)  
3. Reynolds regime adaptation (structural dynamics)
4. Turbulent pattern crystallization (breakthrough challenge)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import json
import time
import math
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
try:
    import yaml  # optional for config files
except Exception:
    yaml = None

from tinycimm_navier import TinyCIMMNavier, create_flow_boundary_conditions
from tinycimm_navier_dashboard import generate_tinycimm_navier_dashboards

class LiveCIMMFlowBenchmark:
    """
    LiveCIMMFlowBenchmark
    ---------------------
    Comprehensive experiment runner for TinyCIMM-Navier validation.
    Implements:
    - Progressive validation of symbolic entropy navigation, pattern ancestry, and thermodynamic compliance (Navier theory)
    - Anchored, reproducible outputs with meta, config, and symbolic trace hooks
    - No training loops: pure CIMM architecture, pattern discovery, and entropy-driven adaptation
    TODO: Integrate full symbolic trace and pattern ancestry export for each run (preprint compliance)
    """
    
    def __init__(self, save_results=True, config: Dict | None = None):
        self.save_results = save_results
        self.results = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Default, externally tunable configuration
        default_config: Dict = {
            'seed': 1337,
            'thermo': {
                'temperature_K': 300.0
            },
            'null_controls': {
                'n_permutations': 1000,
                'tau_percentile': 60,           # main collapse threshold from entropy diffs
                'sensitive_tau_percentile': 40, # inclusive threshold to count minor adjustments
                'jitter_sigma': 0.05
            },
            'breakthrough': {
                'collapse_magnitude': 0.08,
                'entropy_budget': 2.5,
                'min_new_patterns': 2,
                'warmup_steps': 5,
                'delta_entropy': 0.15
            },
            'sweep': {
                'num_seeds': 3,
                'complexities': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
            }
        }
        # Load external config if provided (env or file in experiments dir)
        self.config = self._load_config(default_config, config)

        # Reproducibility seeds
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        # Create results directory anchored to experiments/results
        if self.save_results:
            base_results_dir = Path(__file__).resolve().parent / 'results'
            base_results_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir = str(base_results_dir / f"live_cimm_experiment_{self.experiment_id}")
            os.makedirs(self.results_dir, exist_ok=True)
            self._write_meta()

    def _write_meta(self):
        """
        Persist run metadata for auditability and reproducibility.
        Includes config, git commit, Python/torch versions, and TODO: symbolic trace summary.
        """
        # Try to extract git commit (best-effort)
        git_rev = None
        try:
            git_rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[4]).decode().strip()
        except Exception:
            git_rev = 'unknown'
        meta = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'git_commit': git_rev,
            'config': self.config,
            'python': sys.version,
            'torch': torch.__version__,
            'numpy': np.__version__,
        }
        with open(os.path.join(self.results_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def _load_config(self, defaults: Dict, override: Dict | None) -> Dict:
        cfg = json.loads(json.dumps(defaults))  # deep copy
        # External file path: env or local default
        env_path = os.environ.get('TINYCIMM_NAVIER_CONFIG')
        local_yaml = Path(__file__).resolve().parent / 'config.live_cimm.yaml'
        local_json = Path(__file__).resolve().parent / 'config.live_cimm.json'
        cfg_file = None
        if env_path and Path(env_path).exists():
            cfg_file = Path(env_path)
        elif local_yaml.exists():
            cfg_file = local_yaml
        elif local_json.exists():
            cfg_file = local_json
        # Load and merge
        if cfg_file:
            try:
                if cfg_file.suffix.lower() in {'.yml', '.yaml'} and yaml is not None:
                    with open(cfg_file, 'r') as f:
                        loaded = yaml.safe_load(f) or {}
                else:
                    with open(cfg_file, 'r') as f:
                        loaded = json.load(f)
                cfg = self._deep_update(cfg, loaded)
            except Exception as e:
                print(f"Warning: failed to load config {cfg_file}: {e}")
        if override:
            cfg = self._deep_update(cfg, override)
        return cfg

    def _deep_update(self, base: Dict, upd: Dict) -> Dict:
        for k, v in (upd or {}).items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = self._deep_update(base[k], v)
            else:
                base[k] = v
        return base
    
    def run_live_pattern_discovery(self):
        """
        Phase 1: Live pattern discovery validation
        Tests real-time symbolic pattern crystallization and entropy navigation (no training).
        Outputs pattern ancestry, entropy signatures, and baseline comparisons for theory/preprint compliance.
        TODO: Export full symbolic trace for each scenario.
        """
        print("=== Phase 1: Live Pattern Discovery ===")
        
        model = TinyCIMMNavier(device='cpu')
        
        # Test scenarios for pattern discovery
        scenarios = [
            {"name": "poiseuille_flow", "reynolds": 800, "complexity": 0.1, "steps": 50},
            {"name": "couette_flow", "reynolds": 1200, "complexity": 0.15, "steps": 40},
            {"name": "stagnation_flow", "reynolds": 600, "complexity": 0.08, "steps": 30},
        ]
        
        phase_results = {}
        
        for scenario in scenarios:
            print(f"\nTesting {scenario['name']}...")
            
            scenario_results = {
                'patterns_discovered': [],
                'collapse_events': [],
                'entropy_budget_history': [],
                'prediction_times': [],
                'flow_regime_transitions': [],
                'scbf_neural_dynamics': [],  # SCBF tracking
                'thermo_checks': [],
                'baseline_mse_history': []
            }            # Live pattern discovery session
            for step in range(scenario['steps']):
                # Generate flow input
                bc = create_flow_boundary_conditions(scenario['reynolds'], geometry_type="pipe").unsqueeze(0)
                noise = torch.randn(1, 8) * scenario['complexity']
                flow_input = bc + noise
                
                # Live prediction
                start_time = time.time()
                prediction, diagnostics = model.live_predict(flow_input, scenario['reynolds'])
                prediction_time = (time.time() - start_time) * 1000
                
                # Analytical baselines for laminar scenarios
                baseline = None
                if scenario['name'] == 'poiseuille_flow':
                    baseline = self._poiseuille_baseline(bc)
                elif scenario['name'] == 'couette_flow':
                    baseline = self._couette_baseline(bc)
                if baseline is not None:
                    pred_np = prediction.detach().cpu().numpy()
                    mse = float(np.mean((pred_np - baseline) ** 2))
                    scenario_results['baseline_mse_history'].append(mse)

                # Track results
                scenario_results['prediction_times'].append(prediction_time)
                scenario_results['entropy_budget_history'].append(diagnostics['entropy_budget'])
                
                if diagnostics['collapse_event']['flow_insight_detected']:
                    scenario_results['collapse_events'].append({
                        'step': step,
                        'magnitude': diagnostics['collapse_event']['collapse_magnitude'],
                        'type': diagnostics['collapse_event']['insight_type'],
                        'bc_hash': self._hash_tensor(bc),
                        'entropy_signature': diagnostics.get('entropy_signature')
                    })
                    # Basic thermodynamic logging (Landauer bound estimate)
                    landauer = self._estimate_landauer_energy(diagnostics['collapse_event']['collapse_magnitude'])
                    scenario_results['thermo_checks'].append({
                        'step': step,
                        'temperature_K': self.config['thermo']['temperature_K'],
                        'landauer_energy_J': landauer
                    })
                
                if diagnostics['resonant_patterns']:
                    scenario_results['patterns_discovered'].append({
                        'step': step,
                        'pattern': diagnostics['resonant_patterns'][0],
                        'bc_hash': self._hash_tensor(bc),
                        'entropy_signature': diagnostics.get('entropy_signature')
                    })
                
                # SCBF neural dynamics tracking
                if diagnostics.get('scbf_metrics'):
                    scenario_results['scbf_neural_dynamics'].append({
                        'step': step,
                        'neural_score': diagnostics['scbf_metrics'].get('neural_dynamics_score', 0),
                        'entropy_collapse': diagnostics['scbf_metrics'].get('entropy_collapse', {}),
                        'structural_evolution': diagnostics['scbf_metrics'].get('structural_evolution', {}),
                        'neural_ancestry': diagnostics['scbf_metrics'].get('neural_ancestry', {}),
                        'pattern_attractors': diagnostics['scbf_metrics'].get('pattern_attractors', {})
                    })
                
                # Progress report with SCBF metrics
                if step % 10 == 0:
                    scbf_info = ""
                    if diagnostics.get('scbf_metrics'):
                        scbf = diagnostics['scbf_metrics']
                        neural_score = scbf.get('neural_dynamics_score', 0)
                        entropy_collapse = scbf.get('entropy_collapse', {}).get('collapse_magnitude', 0)
                        mutation_rate = scbf.get('structural_evolution', {}).get('mutation_rate', 0)
                        scbf_info = f" | 🧠 Neural: {neural_score:.2f} | 🔬 Collapse: {entropy_collapse:.3f} | 🧬 Mutation: {mutation_rate:.3f}"
                    
                    print(f"  Step {step:2d}: {prediction_time:.1f}ms | "
                          f"Regime: {diagnostics['flow_regime']:>10s} | "
                          f"Patterns: {diagnostics['crystals_discovered']} | "
                          f"Budget: {diagnostics['entropy_budget']:.3f}{scbf_info}")
            
            phase_results[scenario['name']] = scenario_results
            
            # Scenario summary
            avg_time = np.mean(scenario_results['prediction_times'])
            total_patterns = len(set([p['pattern'] for p in scenario_results['patterns_discovered']]))
            total_collapses = len(scenario_results['collapse_events'])
            
            print(f"  Summary: {avg_time:.1f}ms avg | {total_patterns} unique patterns | {total_collapses} collapses")
        
        return phase_results
    
    def run_entropy_collapse_validation(self):
        """
        Phase 2: Entropy collapse detection validation
        Tests sensitivity to symbolic entropy collapse events and regime transitions.
        Outputs collapse events, entropy dynamics, and Landauer energy logs for theory/preprint compliance.
        TODO: Export symbolic trace and collapse ancestry for each scenario.
        """
        print("\n=== Phase 2: Entropy Collapse Detection ===")
        
        model = TinyCIMMNavier(device='cpu')
        
        # Scenarios designed to trigger entropy collapses
        collapse_scenarios = [
            {"name": "regime_transition", "reynolds_sequence": [500, 1000, 2000, 4000, 8000], "complexity": 0.2},
            {"name": "complexity_ramp", "reynolds": 3000, "complexity_sequence": [0.1, 0.3, 0.6, 1.0, 1.5], "base_complexity": 0.2},
            {"name": "pattern_repetition", "reynolds": 1500, "complexity": 0.15, "repeat_pattern": True}
        ]
        
        phase_results = {}
        
        for scenario in collapse_scenarios:
            print(f"\nTesting {scenario['name']}...")
            
            scenario_results = {
                'major_collapses': [],
                'pattern_crystallizations': [],
                'entropy_dynamics': [],
                'insights_timeline': []
            }
            
            if scenario['name'] == 'regime_transition':
                # Test Reynolds regime transitions
                for i, reynolds in enumerate(scenario['reynolds_sequence']):
                    bc = create_flow_boundary_conditions(reynolds, geometry_type="pipe").unsqueeze(0)
                    flow_input = bc + (torch.randn(1, 8) * scenario['complexity'])
                    
                    prediction, diagnostics = model.live_predict(flow_input, reynolds)
                    
                    scenario_results['entropy_dynamics'].append({
                        'step': i,
                        'reynolds': reynolds,
                        'entropy_budget': diagnostics['entropy_budget'],
                        'flow_regime': diagnostics['flow_regime']
                    })
                    
                    if diagnostics['collapse_event']['flow_insight_detected']:
                        scenario_results['major_collapses'].append({
                            'step': i,
                            'reynolds': reynolds,
                            'magnitude': diagnostics['collapse_event']['collapse_magnitude'],
                            'type': diagnostics['collapse_event']['insight_type'],
                            'bc_hash': self._hash_tensor(bc),
                            'entropy_signature': diagnostics.get('entropy_signature')
                        })
                        print(f"  🔮 Collapse at Re={reynolds}: {diagnostics['collapse_event']['insight_type']}")
                        # Thermodynamic log
                        landauer = self._estimate_landauer_energy(diagnostics['collapse_event']['collapse_magnitude'])
                        scenario_results['insights_timeline'].append({'step': i, 'landauer_energy_J': landauer})
            
            elif scenario['name'] == 'complexity_ramp':
                # Test complexity-driven collapses
                reynolds = scenario['reynolds']
                for i, complexity in enumerate(scenario['complexity_sequence']):
                    bc = create_flow_boundary_conditions(reynolds, geometry_type="pipe").unsqueeze(0)
                    flow_input = bc + (torch.randn(1, 8) * complexity)
                    
                    prediction, diagnostics = model.live_predict(flow_input, reynolds)
                    if diagnostics['collapse_event']['flow_insight_detected']:
                        scenario_results['major_collapses'].append({
                            'step': i,
                            'complexity': complexity,
                            'magnitude': diagnostics['collapse_event']['collapse_magnitude'],
                            'bc_hash': self._hash_tensor(bc),
                            'entropy_signature': diagnostics.get('entropy_signature')
                        })
                        print(f"  🔮 Collapse at complexity={complexity:.1f}: {diagnostics['collapse_event']['insight_type']}")
                        # Thermodynamic log
                        landauer = self._estimate_landauer_energy(diagnostics['collapse_event']['collapse_magnitude'])
                        scenario_results['insights_timeline'].append({'step': i, 'landauer_energy_J': landauer})
            
            elif scenario['name'] == 'pattern_repetition':
                # Test repeated pattern recognition
                reynolds = scenario['reynolds']
                base_pattern = create_flow_boundary_conditions(reynolds, geometry_type="pipe").unsqueeze(0)
                
                for i in range(20):
                    # Add small variations to base pattern
                    flow_input = base_pattern + torch.randn(1, 8) * 0.01
                    
                    prediction, diagnostics = model.live_predict(flow_input, reynolds)
                    
                    if diagnostics['resonant_patterns']:
                        scenario_results['pattern_crystallizations'].append({
                            'step': i,
                            'pattern': diagnostics['resonant_patterns'][0]
                        })
            
            phase_results[scenario['name']] = scenario_results
        
        return phase_results
    
    def run_reynolds_adaptation_test(self):
        """
        Phase 3: Reynolds regime adaptation test
        Tests symbolic regime recognition and structural adaptation across Reynolds sweep.
        Outputs regime transitions, entropy budgets, and pattern evolution for theory/preprint compliance.
        TODO: Export regime ancestry and symbolic trace for each sweep.
        """
        print("\n=== Phase 3: Reynolds Regime Adaptation ===")
        
        model = TinyCIMMNavier(device='cpu')
        
        # Reynolds sweep test
        reynolds_sweep = [100, 500, 1000, 2000, 3000, 5000, 8000, 15000, 30000, 50000]
        
        adaptation_results = {
            'regime_recognition': [],
            'entropy_budget_evolution': [],
            'pattern_evolution': [],
            'structural_changes': []
        }
        
        print(f"Testing Reynolds sweep: {reynolds_sweep}")
        
        for reynolds in reynolds_sweep:
            # Generate appropriate complexity for Reynolds number
            complexity = min(1.5, 0.1 + (reynolds / 10000) * 0.5)
            bc = create_flow_boundary_conditions(reynolds, geometry_type="pipe").unsqueeze(0)
            flow_input = bc + (torch.randn(1, 8) * complexity)
            
            # Live prediction
            prediction, diagnostics = model.live_predict(flow_input, reynolds)
            
            adaptation_results['regime_recognition'].append({
                'reynolds': reynolds,
                'detected_regime': diagnostics['flow_regime'],
                'entropy_budget': diagnostics['entropy_budget']
            })
            
            adaptation_results['entropy_budget_evolution'].append(diagnostics['entropy_budget'])
            
            if diagnostics['crystals_discovered'] > 0:
                adaptation_results['pattern_evolution'].append({
                    'reynolds': reynolds,
                    'patterns': diagnostics['crystals_discovered']
                })
            
            print(f"  Re={reynolds:5d}: Regime={diagnostics['flow_regime']:>10s} | "
                  f"Budget={diagnostics['entropy_budget']:.3f} | "
                  f"Patterns={diagnostics['crystals_discovered']}")
        
        return adaptation_results
    
    def run_turbulent_crystallization_challenge(self):
        """
        Phase 4: Turbulent pattern crystallization challenge
        Ultimate test of symbolic pattern discovery and entropy navigation in chaotic regimes.
        Outputs breakthrough stats, pattern ancestry, and entropy/thermodynamic metrics for theory/preprint compliance.
        TODO: Export full symbolic trace and pattern ancestry for each challenge.
        """
        print("\n=== Phase 4: Turbulent Crystallization Challenge ===")
        
        model = TinyCIMMNavier(device='cpu')
        
        # High Reynolds turbulent scenarios with enhanced complexity
        turbulent_challenges = [
            {"name": "pipe_turbulence", "reynolds": 10000, "complexity": 1.3, "steps": 140},  # slightly longer to stabilize
            {"name": "mixing_layer", "reynolds": 25000, "complexity": 1.5, "steps": 120},     # longer horizon
            # For high Re, expand steps and allow per-challenge breakthrough tuning via optional fields
            {"name": "high_re_chaos", "reynolds": 100000, "complexity": 2.5, "steps": 220, "breakthrough_overrides": {"warmup_steps": 16, "delta_entropy": 0.05}},
            {"name": "extreme_turbulence", "reynolds": 200000, "complexity": 3.0, "steps": 260, "breakthrough_overrides": {"warmup_steps": 20, "delta_entropy": 0.05}}
        ]
        
        challenge_results = {}
        
        sweep_cfg = self.config.get('sweep', {}) or {}
        num_seeds = int(sweep_cfg.get('num_seeds', 1))
        comp_values = sweep_cfg.get('complexities') or []
        targeted_cfg = (sweep_cfg.get('targeted') or {}) if isinstance(sweep_cfg, dict) else {}

        def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
            if n == 0:
                return (0.0, 1.0)
            p = k / n
            denom = 1 + z*z/n
            center = (p + z*z/(2*n)) / denom
            half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
            return (max(0.0, center - half), min(1.0, center + half))

        aggregates: Dict[str, Dict[str, Dict[str, float]]] = {}

        def _name_to_re(name: str) -> float:
            if 'high_re_chaos' in name:
                return 100000
            if 'extreme_turbulence' in name:
                return 200000
            if 'mixing_layer' in name:
                return 25000
            if 'pipe_turbulence' in name:
                return 10000
            return 1000

        def _run_challenge(challenge: Dict, complexities: List[float], seeds: int):
            base_name = challenge['name']
            base_steps = int(challenge.get('steps', 200))
            for c in complexities:
                successes = 0
                total = 0
                ttb_list: List[int] = []  # breakthrough-only time-to-breakthrough list
                horizon_list: List[int] = []  # per-run horizons for censoring context
                for si in range(seeds):
                    # Seed per run for sweep reproducibility
                    torch.manual_seed(self.config['seed'] + 1000*si + int(c*100))
                    np.random.seed(self.config['seed'] + 1000*si + int(c*100))

                    print(f"\nTurbulent Challenge: {base_name} (Re={challenge['reynolds']} | seed={si} | c={c})")

                    challenge_cfg = dict(challenge)
                    challenge_cfg['complexity'] = float(c)
                    challenge_cfg['seed'] = si
                    challenge_data = {
                        'breakthrough_detected': False,
                        'breakthrough_step': None,
                        'patterns_discovered': [],
                        'major_insights': [],
                        'entropy_evolution': [],
                        'challenge_config': challenge_cfg,
                        'horizon': int(challenge_cfg.get('steps', base_steps))
                    }

                    prev_budget = None
                    for step in range(int(challenge_cfg.get('steps', base_steps))):
                        # High complexity turbulent input
                        bc = create_flow_boundary_conditions(challenge['reynolds'], geometry_type="pipe").unsqueeze(0)
                        flow_input = bc + (torch.randn(1, 8) * challenge_cfg['complexity'])

                        prediction, diagnostics = model.live_predict(flow_input, challenge['reynolds'])

                        budget = diagnostics['entropy_budget']
                        challenge_data['entropy_evolution'].append(budget)

                        # Hardened breakthrough detection: require warmup and a positive delta
                        # Base thresholds, with optional per-challenge overrides (for very high Re)
                        thresholds = dict(self.config['breakthrough'])
                        if 'breakthrough_overrides' in challenge:
                            thresholds.update(challenge['breakthrough_overrides'])
                        delta_ok = (prev_budget is not None) and ((budget - prev_budget) > thresholds.get('delta_entropy', 0.0))
                        warmup_ok = step >= int(thresholds.get('warmup_steps', 0))
                        conds = [
                            (diagnostics['collapse_event']['flow_insight_detected'] and 
                             diagnostics['collapse_event']['insight_type'] == 'major_flow_insight'),
                            (diagnostics['collapse_event']['flow_insight_detected'] and
                             diagnostics['collapse_event'].get('collapse_magnitude', 0) > thresholds['collapse_magnitude']),
                            (budget > thresholds['entropy_budget'] and delta_ok),
                        ]

                        if warmup_ok and any(conds):
                            if not challenge_data['breakthrough_detected']:
                                challenge_data['breakthrough_detected'] = True
                                challenge_data['breakthrough_step'] = step
                                print(f"  *** TURBULENT BREAKTHROUGH DETECTED at step {step}! ***")
                                print(f"      Trigger: Budget={budget:.3f}, "
                                      f"Magnitude={diagnostics['collapse_event'].get('collapse_magnitude', 0):.3f}")

                            challenge_data['major_insights'].append({
                                'step': step,
                                'magnitude': diagnostics['collapse_event'].get('collapse_magnitude', 0),
                                'entropy_budget': budget,
                                'landauer_energy_J': self._estimate_landauer_energy(
                                    diagnostics['collapse_event'].get('collapse_magnitude', 0)),
                                'bc_hash': self._hash_tensor(bc),
                                'entropy_signature': diagnostics.get('entropy_signature')
                            })

                        prev_budget = budget

                        if diagnostics['crystals_discovered'] > len(challenge_data['patterns_discovered']):
                            new_patterns = diagnostics['crystals_discovered'] - len(challenge_data['patterns_discovered'])
                            challenge_data['patterns_discovered'].extend([step] * new_patterns)
                            print(f"  🔮 New turbulent pattern crystallized at step {step}")

                        # Progress report
                        if step % 20 == 0:
                            print(f"  Step {step:2d}: Budget={budget:.3f} | "
                                  f"Patterns={diagnostics['crystals_discovered']} | "
                                  f"Insights={diagnostics['insights_discovered']}")

                    total += 1
                    horizon_list.append(int(challenge_data['horizon']))
                    if challenge_data['breakthrough_detected']:
                        successes += 1
                        ttb_list.append(int(challenge_data['breakthrough_step']))
                        print(f"  ✅ BREAKTHROUGH: Step {challenge_data['breakthrough_step']} | "
                              f"Patterns: {len(challenge_data['patterns_discovered'])} | "
                              f"Insights: {len(challenge_data['major_insights'])}")
                    else:
                        print(f"  ⚠️  No breakthrough detected | "
                              f"Patterns: {len(challenge_data['patterns_discovered'])}")

                    # Store per-run under a composite key so dashboards can aggregate by name substring
                    run_key = f"{base_name}|s{si}|c{c:.2f}"
                    challenge_results[run_key] = challenge_data

                # Aggregate per challenge/complexity
                rate = successes / total if total else 0.0
                ci_lo, ci_hi = wilson_ci(successes, total)
                # TTB stats among observed breakthroughs only
                ttb_mean = float(np.mean(ttb_list)) if ttb_list else None
                ttb_median = float(np.median(ttb_list)) if ttb_list else None
                aggregates.setdefault(base_name, {})[f"c={c:.2f}"] = {
                    'success': successes,
                    'total': total,
                    'rate': rate,
                    'ci_low': ci_lo,
                    'ci_high': ci_hi,
                    'ttb_mean': ttb_mean,
                    'ttb_median': ttb_median,
                    'reynolds': float(_name_to_re(base_name))
                }

        # Base sweep over default challenges
        for challenge in turbulent_challenges:
            base_complexity = challenge['complexity']
            complexities = comp_values if comp_values else [base_complexity]
            _run_challenge(challenge, [float(x) for x in complexities], num_seeds)

        # Optional targeted high-power sweep around boundaries
        if targeted_cfg:
            print("\n--- Targeted high-power sweep enabled ---")
            cells = targeted_cfg.get('cells') or []
            for cell in cells:
                try:
                    name = str(cell['name'])
                except Exception:
                    continue
                # Try to find base properties from default list; else infer Re
                base = next((c for c in turbulent_challenges if c['name'] == name), None)
                if base is None:
                    base = {
                        'name': name,
                        'reynolds': _name_to_re(name),
                        'complexity': 1.0,
                        'steps': int(cell.get('steps', 260))
                    }
                # Apply overrides from targeted cell
                t_challenge = dict(base)
                if 'steps' in cell:
                    t_challenge['steps'] = int(cell['steps'])
                if 'breakthrough_overrides' in cell:
                    t_challenge['breakthrough_overrides'] = dict(cell['breakthrough_overrides'])
                t_complexities = [float(x) for x in (cell.get('complexities') or [])]
                t_seeds = int(cell.get('num_seeds', seeds)) if (seeds := targeted_cfg.get('num_seeds')) is not None else int(cell.get('num_seeds', num_seeds))
                if not t_complexities:
                    # Fallback to base complexity
                    t_complexities = [float(base['complexity'])]
                _run_challenge(t_challenge, t_complexities, t_seeds)

        # Attach aggregates for downstream dashboards
        challenge_results['sweep_stats'] = aggregates

        return challenge_results

    # --- Thermodynamic utility ---
    def _estimate_landauer_energy(self, collapse_magnitude: float) -> float:
        """Estimate Landauer minimum energy for a collapse event.
        Approximates changed bits as a function of collapse magnitude.
        E_min = k_B * T * ln(2) * bits
        """
        k_B = 1.380649e-23  # Boltzmann constant (J/K)
        T = float(self.config.get('thermo', {}).get('temperature_K', 300.0))
        # Very simple mapping from magnitude to approximate bits changed
        est_bits = max(0.0, collapse_magnitude) * 100.0
        return k_B * T * math.log(2) * est_bits

    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        b = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
        import hashlib
        return hashlib.sha256(b).hexdigest()

    def run_null_controls(self, pattern_results: Dict, collapse_results: Dict) -> Dict:
        """
        Run null/permutation controls for collapse counts using entropy dynamics.
        Returns p-values and null distributions for collapse events, supporting statistical rigor and preprint compliance.
        TODO: Export null control traces and integrate with symbolic ancestry.
        """
        null_summary: Dict[str, Dict] = {}
        rng = np.random.default_rng(self.config.get('seed', 1337))
        nc = self.config.get('null_controls', {})
        num_perm = int(nc.get('n_permutations', 1000))
        tau_pct_main = float(nc.get('tau_percentile', 60))
        tau_pct_sensitive = float(nc.get('sensitive_tau_percentile', 40))

        def _collapse_count(entropy_series: np.ndarray, tau_pct: float) -> Tuple[int, float]:
            diffs = entropy_series[:-1] - entropy_series[1:]
            tau = max(0.0, float(np.percentile(np.abs(diffs), tau_pct)))
            obs = int(np.sum(diffs > tau))
            return obs, tau

        def _permute_counts(entropy_series: np.ndarray, tau: float) -> np.ndarray:
            counts = np.empty(num_perm, dtype=int)
            for k in range(num_perm):
                sh = entropy_series.copy()
                rng.shuffle(sh)
                d = sh[:-1] - sh[1:]
                counts[k] = int(np.sum(d > tau))
            return counts

        def _shift_counts(entropy_series: np.ndarray, tau: float) -> np.ndarray:
            counts = np.empty(num_perm, dtype=int)
            L = len(entropy_series)
            for k in range(num_perm):
                s = rng.integers(0, L)
                sh = np.roll(entropy_series, s)
                d = sh[:-1] - sh[1:]
                counts[k] = int(np.sum(d > tau))
            return counts

        def _jitter_counts(entropy_series: np.ndarray, tau: float, sigma: float) -> np.ndarray:
            counts = np.empty(num_perm, dtype=int)
            for k in range(num_perm):
                noise = rng.normal(0.0, sigma, size=entropy_series.shape)
                sh = entropy_series + noise
                d = sh[:-1] - sh[1:]
                counts[k] = int(np.sum(d > tau))
            return counts

        # Use phase 2 entropy_dynamics series to derive collapse counts
        for scen_name, scen in collapse_results.items():
            series = scen.get('entropy_dynamics', [])
            if not series or len(series) < 3:
                continue
            ent = np.array([x['entropy_budget'] for x in series], dtype=np.float32)

            # Main threshold (more conservative)
            obs_main, tau_main = _collapse_count(ent, tau_pct_main)
            perm_main = _permute_counts(ent, tau_main)
            p_perm = float((np.sum(perm_main >= obs_main) + 1) / (num_perm + 1))

            # Variant 1: circular shift null
            shift_counts = _shift_counts(ent, tau_main)
            p_shift = float((np.sum(shift_counts >= obs_main) + 1) / (num_perm + 1))

            # Inclusive threshold to count minor adjustments
            obs_inclusive, tau_incl = _collapse_count(ent, tau_pct_sensitive)
            perm_inclusive = _permute_counts(ent, tau_incl)
            p_perm_incl = float((np.sum(perm_inclusive >= obs_inclusive) + 1) / (num_perm + 1))

            # Primary entry (permute, main tau)
            null_summary[scen_name] = {
                'observed_collapse_count': int(obs_main),
                'tau': float(tau_main),
                'perm_mean': float(np.mean(perm_main)),
                'perm_std': float(np.std(perm_main)),
                'p_value': p_perm,
                'n_permutations': num_perm,
                'variants': {
                    'circular_shift': {
                        'p_value': p_shift,
                        'null_mean': float(np.mean(shift_counts)),
                        'null_std': float(np.std(shift_counts)),
                        'tau': float(tau_main)
                    },
                    'perm_inclusive': {
                        'p_value': p_perm_incl,
                        'tau': float(tau_incl),
                        'observed_inclusive_count': int(obs_inclusive),
                        'null_mean': float(np.mean(perm_inclusive)),
                        'null_std': float(np.std(perm_inclusive))
                    },
                    'jittered': {
                        'p_value': None,
                        'tau': float(tau_main)
                    }
                }
            }

            # Also expose variants as separate entries for dashboard bar comparisons
            null_summary[f"{scen_name} (shift)"] = {
                'observed_collapse_count': int(obs_main),
                'tau': float(tau_main),
                'perm_mean': float(np.mean(shift_counts)),
                'perm_std': float(np.std(shift_counts)),
                'p_value': p_shift,
                'n_permutations': num_perm
            }
            null_summary[f"{scen_name} (inclusive)"] = {
                'observed_collapse_count': int(obs_inclusive),
                'tau': float(tau_incl),
                'perm_mean': float(np.mean(perm_inclusive)),
                'perm_std': float(np.std(perm_inclusive)),
                'p_value': p_perm_incl,
                'n_permutations': num_perm
            }

            # Jittered null as separate entry
            sigma = float(nc.get('jitter_sigma', 0.05))
            jitter_counts = _jitter_counts(ent, tau_main, sigma)
            p_jitter = float((np.sum(jitter_counts >= obs_main) + 1) / (num_perm + 1))
            null_summary[scen_name]['variants']['jittered']['p_value'] = p_jitter
            null_summary[f"{scen_name} (jitter)"] = {
                'observed_collapse_count': int(obs_main),
                'tau': float(tau_main),
                'perm_mean': float(np.mean(jitter_counts)),
                'perm_std': float(np.std(jitter_counts)),
                'p_value': p_jitter,
                'n_permutations': num_perm
            }

        return null_summary

    # --- Simple analytical baselines (normalized) ---
    def _poiseuille_baseline(self, bc_tensor: torch.Tensor) -> np.ndarray:
        """Approximate normalized Poiseuille features [u,v,p,w]"""
        bc = bc_tensor.detach().cpu().numpy()
        Re_n = float(bc[0, 0]) if bc.ndim == 2 else float(bc[0])
        u_avg = Re_n
        v_avg = 0.0
        dp = -0.5 * Re_n
        vort = 0.05 * Re_n
        return np.array([[u_avg, v_avg, dp, vort]], dtype=np.float32)

    def _couette_baseline(self, bc_tensor: torch.Tensor) -> np.ndarray:
        """Approximate normalized Couette features [u,v,p,w]"""
        bc = bc_tensor.detach().cpu().numpy()
        wall_vel = 0.5
        try:
            wall_vel = float(bc[0, 3])
        except Exception:
            pass
        u_avg = 0.5 * wall_vel
        v_avg = 0.0
        dp = 0.0
        vort = 0.1 * wall_vel
        return np.array([[u_avg, v_avg, dp, vort]], dtype=np.float32)
    
    def run_comprehensive_validation(self):
        """
        Run complete live CIMM validation suite (Navier theory compliant).
        Executes all phases, saves meta/results, and generates dashboards with symbolic entropy, pattern ancestry, and thermodynamic metrics.
        TODO: Export symbolic trace and ancestry summary in meta/results for preprint compliance.
        """
        print("🚀 TinyCIMM-Navier Live CIMM Comprehensive Validation")
        print("True CIMM Architecture: Live Prediction + Pattern Crystallization + Entropy Insights")
        print("=" * 80)

        start_time = time.time()

        # Phase 1: Live pattern discovery
        pattern_results = self.run_live_pattern_discovery()

        # Phase 2: Entropy collapse validation
        collapse_results = self.run_entropy_collapse_validation()

        # Phase 3: Reynolds adaptation
        adaptation_results = self.run_reynolds_adaptation_test()

        # Phase 4: Turbulent crystallization challenge
        turbulent_results = self.run_turbulent_crystallization_challenge()

        # Phase 5: Null/permutation controls
        null_controls = self.run_null_controls(pattern_results, collapse_results)

        total_time = time.time() - start_time

        # Compile comprehensive results
        comprehensive_results = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'total_validation_time': total_time,
            'cimm_architecture': 'live_prediction',
            'training_loops_used': False,
            'phase_1_pattern_discovery': pattern_results,
            'phase_2_entropy_collapse': collapse_results,
            'phase_3_reynolds_adaptation': adaptation_results,
            'phase_4_turbulent_challenge': turbulent_results,
            'phase_5_null_controls': null_controls,
        }

        # Save results
        if self.save_results:
            results_file = f"{self.results_dir}/comprehensive_live_cimm_results.json"
            with open(results_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)

            # Generate comprehensive dashboards
            print("\n🎨 Generating analytical dashboards...")
            try:
                dashboard_paths = generate_tinycimm_navier_dashboards(results_file, self.results_dir)
                print(f"✅ Generated {len(dashboard_paths)} dashboard visualizations:")
                for path in dashboard_paths:
                    print(f"   📊 {os.path.basename(path)}")
            except Exception as e:
                print(f"⚠️ Dashboard generation failed: {e}")

        # Final summary
        print("\n🎯 Live CIMM Validation Complete!")
        print(f"   Total time: {total_time:.1f}s")
        print("   Architecture: True CIMM (no training loops)")
        print(f"   Results saved to: {self.results_dir}")

        self._print_validation_summary(comprehensive_results)

        return comprehensive_results
    
    def _print_validation_summary(self, results):
        """
        Print comprehensive validation summary.
        Includes symbolic entropy, pattern ancestry, and thermodynamic metrics for theory/preprint compliance.
        TODO: Print/export symbolic trace and ancestry summary.
        """
        print(f"\n✨ CIMM Validation Summary:")
        
        # Pattern discovery summary
        phase1 = results['phase_1_pattern_discovery']
        total_patterns = sum(len(set([p['pattern'] for p in scenario['patterns_discovered']])) 
                           for scenario in phase1.values())
        total_collapses = sum(len(scenario['collapse_events']) for scenario in phase1.values())
        
        print(f"   Phase 1 - Pattern Discovery:")
        print(f"     ✅ Unique patterns discovered: {total_patterns}")
        print(f"     ✅ Entropy collapses detected: {total_collapses}")
        
        # Turbulent challenge summary
        phase4 = results['phase_4_turbulent_challenge']
        breakthroughs = 0
        total_turbulent_patterns = 0
        for key, challenge in phase4.items():
            if not isinstance(challenge, dict):
                continue
            if 'breakthrough_detected' in challenge:
                breakthroughs += 1 if challenge.get('breakthrough_detected') else 0
                total_turbulent_patterns += len(challenge.get('patterns_discovered', []))
        
        print(f"   Phase 4 - Turbulent Challenge:")
        print(f"     🚀 Turbulent breakthroughs: {breakthroughs}/{len(phase4)}")
        print(f"     🔮 Turbulent patterns: {total_turbulent_patterns}")
        

def main():
    """Run the live CIMM validation experiment"""
    # Support a lightweight CLI for dashboard regeneration without re-running experiments
    args = sys.argv[1:]
    if '--dashboards-only' in args:
        try:
            idx = args.index('--results')
            results_path = args[idx + 1]
        except Exception:
            print("Error: --dashboards-only requires --results <path_to_results_json>")
            sys.exit(2)
        out_dir = os.path.dirname(results_path)
        print(f"Generating dashboards only from: {results_path}")
        try:
            paths = generate_tinycimm_navier_dashboards(results_path, out_dir)
            print(f"✅ Generated {len(paths)} dashboards")
        except Exception as e:
            print(f"⚠️ Dashboard generation failed: {e}")
            sys.exit(1)
        return None

    benchmark = LiveCIMMFlowBenchmark(save_results=True)
    results = benchmark.run_comprehensive_validation()
    return results

if __name__ == "__main__":
    results = main()
