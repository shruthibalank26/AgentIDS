"""Generate every AgentNet-IDS scenario used in the paper (deterministic seeds)."""
import os, time
from agentnet_sim import simulate

SCENARIOS = (
    [("selection", dict(attack_frac=0.2, evasion=0.0, seed=41))] +
    [(f"main_s{s}", dict(attack_frac=0.2, evasion=0.0, seed=s)) for s in (42, 43, 44)] +
    [(f"density_{int(f*100):02d}", dict(attack_frac=f, evasion=0.0, seed=50 + k))
     for k, f in enumerate([0.05, 0.10, 0.20, 0.30, 0.40])] +
    [(f"evtrain_s{s}", dict(attack_frac=0.2, evasion=0.5, seed=s)) for s in (60, 61)] +
    [(f"evasion_{int(e*100):03d}", dict(attack_frac=0.2, evasion=e, seed=70 + k))
     for k, e in enumerate([0.0, 0.25, 0.5, 0.75, 1.0])]
)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for name, kw in SCENARIOS:
        path = f"data/{name}.csv"
        if os.path.exists(path):
            continue
        t0 = time.time()
        d = simulate(n_agents=600, n_windows=30, **kw)
        d.to_csv(path, index=False)
        print(name, d.shape, f"{time.time() - t0:.0f}s", flush=True)
