"""Summary tables (results/summary_*.csv) and figures (figures/*.png) from results/."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agentnet_sim import CLASS_NAMES

os.makedirs("figures", exist_ok=True)
ms = lambda s: s.groupby(level=0).agg(["mean", "std"]) if False else None


def table(path, by, cols):
    d = pd.read_csv(path)
    t = d.groupby(by)[cols].agg(["mean", "std"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    return t


if __name__ == "__main__":
    cols = ["macro_f1", "accuracy", "mcc", "dr", "fpr", "agent_dr", "agent_fpr", "n_features", "fit_seconds"]
    e1 = table("results/e1_fs_methods.csv", "method", cols).sort_values("macro_f1_mean", ascending=False)
    e1.round(4).to_csv("results/summary_e1.csv"); print(e1.round(4).to_string())
    per = pd.read_csv("results/e1_fs_methods.csv").groupby("method")[[f"f1_{c}" for c in CLASS_NAMES]].mean()
    per.round(4).to_csv("results/summary_e1_perclass.csv"); print(per.round(3).to_string())
    for f, by in [("e1_hwgo_ablation", "method"), ("e2_classifiers", "method"), ("e4_group_ablation", "method")]:
        if os.path.exists(f"results/{f}.csv"):
            t = table(f"results/{f}.csv", by, [c for c in cols if c != "n_features"])
            t.round(4).to_csv(f"results/summary_{f[:2]}{f[2:]}.csv"); print(t.round(4).to_string())
    if os.path.exists("results/e3_binary_baselines.csv"):
        t = pd.read_csv("results/e3_binary_baselines.csv").groupby("method")[["dr", "fpr", "binary_f1"]].agg(["mean", "std"])
        t.round(4).to_csv("results/summary_e3.csv"); print(t.round(4).to_string())
    for f in ("e5_density", "e6_evasion"):
        if os.path.exists(f"results/{f}.csv"):
            t = pd.read_csv(f"results/{f}.csv"); print(t[["setting", "value", "macro_f1", "dr", "fpr", "agent_dr", "agent_fpr"]].round(4).to_string())
    if os.path.exists("results/e7_unseen.csv"):
        t = pd.read_csv("results/e7_unseen.csv").groupby(["held_out", "model"])[["dr_unseen", "fpr", "agent_dr_unseen"]].mean()
        t.round(4).to_csv("results/summary_e7.csv"); print(t.round(4).to_string())

    # figures
    runs = json.load(open("results/fs_runs.json"))
    plt.figure(figsize=(6, 4))
    for m in ["HWGO", "WaOA", "GOA", "BPSO", "BGA", "BGWO"]:
        H = np.array([r["history"][:250] for r in runs if r["method"] == m])
        plt.plot(np.arange(1, H.shape[1] + 1), H.mean(0), label=m)
    plt.xlabel("Fitness evaluations"); plt.ylabel("Best fitness (mean of 3 seeds)"); plt.legend(); plt.tight_layout()
    plt.savefig("figures/convergence.png", dpi=200)
    cm = np.array(json.load(open("results/e1_confusion.json"))["HWGO"])
    cmn = cm / cm.sum(1, keepdims=True)
    plt.figure(figsize=(6, 5)); plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            plt.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    plt.xticks(range(7), CLASS_NAMES, rotation=45); plt.yticks(range(7), CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig("figures/confusion_hwgo.png", dpi=200)
    print("figures written")
