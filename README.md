# AgentNet-IDS

Code and data for *Detecting Adversarial Agents in Open Agentic Networks: A Behavioral Intrusion Detection Approach* (submitted to ACM TWEB, Special Issue on the Agentic Web).

## Contents

| File | Purpose |
| --- | --- |
| `agentnet_sim.py` | Discrete-time simulator of an open agent network; emits one row per agent per window with 52 behavioural features and labels |
| `generate_datasets.py` | Generates every scenario used in the paper with fixed seeds (selection, main, density, evasion) |
| `feature_selection.py` | Wrapper fitness and binary optimisers: BPSO, BGA, BGWO, WaOA, GOA and the hybrid Walrus-Gazelle optimiser (HWGO) with ablations |
| `run_feature_selection.py` | Runs all optimisers on the selection scenario (3 seeds, 250 fitness evaluations each) |
| `evaluate.py` | Experiments E1 to E7 (feature selection, classifiers, baselines, group ablation, density, evasion, unseen attacks) |
| `make_figures.py` | Tables and figures from `results/` |
| `data/` | Generated scenarios as CSV (one row per agent-window) |
| `results/` | Raw per-fold results, selected subsets, optimiser histories |

## Reproduce

```bash
pip install numpy pandas scikit-learn networkx matplotlib
python generate_datasets.py        # ~4 min on one core
python run_feature_selection.py    # ~40 min on one core
python evaluate.py                 # ~40 min on one core (or: python evaluate.py E1 E3)
python make_figures.py
```

## Dataset schema

Each CSV row is one agent in one observation window.

- 52 feature columns (see `FEATURES` and `FEATURE_GROUPS` in `agentnet_sim.py`)
- `label`: 0 benign, 1 sybil, 2 poisoner, 3 injector, 4 colluder, 5 dropper, 6 flooder
- `subtype`: benign sub-population (regular, hub, newcomer, lowq, critic, aggregator, fleet) or attack name
- `attack_on`: whether the attacker was active in this window (on-off behaviour)
- `reputation`: the registry's Beta reputation score after the window (used by the reputation baseline only)
- `agent_id`, `window`, `scenario_seed`, `attack_frac`, `evasion`

Always split by agent (`scenario_seed` + `agent_id`), never by row, to avoid identity leakage.

## Note on HWGO

`hwgo()` implements the hybridisation of the Walrus Optimisation Algorithm and the Gazelle Optimisation Algorithm described in Section 5.3 of the paper, following Selvaraj, Kavitha Devi and Selvi (ICNGCS 2025). Check the update rules against that paper before reuse.

## Limitations

The data are simulated. Behavioural distributions are design choices, documented in the code, not measurements of a deployed agent network. See Section 8 of the paper.
