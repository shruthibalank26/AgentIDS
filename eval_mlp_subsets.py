"""E1b: every feature subset re-evaluated with the MLP classifier."""
import json, numpy as np, pandas as pd
from evaluate import load, cv, filter_subsets, FEATURES
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
main = load([f"main_s{s}" for s in (42, 43, 44)])
best = json.load(open("results/selected_subsets.json"))
mlp = lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=0))
subs = {"All features": FEATURES, **{m: best[m] for m in ["HWGO", "WaOA", "GOA", "BPSO", "BGA", "BGWO"]}, **filter_subsets(len(best["HWGO"]))}
rows = []
for n, f in subs.items():
    r, cm = cv(main, f, mlp, n); rows += r
    if n == "HWGO":
        json.dump(cm.tolist(), open("results/e1_confusion_mlp_hwgo.json", "w"))
    pd.DataFrame(rows).to_csv("results/e1b_fs_methods_mlp.csv", index=False)
    print(n, len(f), round(np.mean([x['macro_f1'] for x in r]), 4), flush=True)
