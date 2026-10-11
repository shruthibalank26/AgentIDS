"""Run every wrapper feature selector on the held-out *selection* scenario.

The selection scenario (seed 41) is never used for evaluation, so the chosen
subsets carry no information about the evaluation scenarios.
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from agentnet_sim import FEATURES
from feature_selection import WrapperFitness, run_optimiser, ABLATIONS

BUDGET, SEEDS = 250, (0, 1, 2)

def fitness_data():
    d = pd.read_csv("data/selection.csv")
    tr, va = next(StratifiedGroupKFold(3, shuffle=True, random_state=0).split(d, d.label, d.agent_id))
    rng = np.random.default_rng(0)
    y = d.label.values
    ben, att = tr[y[tr] == 0], tr[y[tr] > 0]
    tr = np.concatenate([att, rng.choice(ben, len(ben) // 4, replace=False)])   # speed: 1/4 benign rows
    X = d[FEATURES].values
    return X[tr], y[tr], X[va], y[va]

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    Xtr, ytr, Xva, yva = fitness_data()
    out_path = "results/fs_runs.json"
    runs = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["method"], r["seed"]) for r in runs}
    jobs = [(m, {}) for m in ["HWGO", "WaOA", "GOA", "BPSO", "BGA", "BGWO"]] + \
           [(m, kw) for m, kw in ABLATIONS.items() if m != "HWGO"]
    for seed in SEEDS:
        for name, kw in jobs:
            if (name, seed) in done:
                continue
            fit = WrapperFitness(Xtr, ytr, Xva, yva, n_trees=30, seed=seed)
            r = run_optimiser(name, fit, len(FEATURES), BUDGET, seed, **kw)
            r["mask"] = [FEATURES[k] for k in np.where(r["mask"])[0]]
            runs.append(r)
            json.dump(runs, open(out_path, "w"))
            print(f"{name:28s} seed={seed} fit={r['fitness']:.4f} valF1={r['val_f1']:.4f} "
                  f"k={r['n_features']} {r['seconds']:.0f}s", flush=True)
