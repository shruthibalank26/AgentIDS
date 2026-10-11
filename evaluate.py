"""
All experiments reported in the paper. Results are written to results/*.json
and results/*.csv. Run after generate_datasets.py and run_feature_selection.py.

E1  detection with each feature-selection method (5-fold grouped CV, main set)
E2  classifier comparison on the HWGO subset
E3  binary baselines: reputation threshold, Isolation Forest
E4  feature-group ablation
E5  transfer to other attacker densities
E6  transfer to evasive attackers (+ evasion-aware retraining)
E7  unseen attacks (leave-one-attack-out)
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, IsolationForest)
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, accuracy_score, matthews_corrcoef, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from agentnet_sim import FEATURES, FEATURE_GROUPS, CLASS_NAMES

os.makedirs("results", exist_ok=True)
K_FOLDS = 5


def load(names):
    d = pd.concat([pd.read_csv(f"data/{n}.csv") for n in names], ignore_index=True)
    d["uid"] = d["scenario_seed"].astype(str) + "_" + d["agent_id"].astype(str)
    return d


def rf(n=200, seed=0):
    return RandomForestClassifier(n, max_samples=0.5, class_weight="balanced", n_jobs=1, random_state=seed)


def metrics(y, p, prob_attack=None, uid=None):
    m = dict(macro_f1=f1_score(y, p, average="macro"), accuracy=accuracy_score(y, p),
             mcc=matthews_corrcoef(y, p), weighted_f1=f1_score(y, p, average="weighted"))
    yb, pb = (y > 0), (p > 0)
    m["dr"] = pb[yb].mean(); m["fpr"] = pb[~yb].mean()
    m["binary_f1"] = f1_score(yb, pb)
    _, _, f, _ = precision_recall_fscore_support(y, p, labels=range(7), zero_division=0)
    for k, c in enumerate(CLASS_NAMES):
        m[f"f1_{c}"] = f[k]
    if prob_attack is not None and uid is not None:          # agent-level decision
        a = pd.DataFrame(dict(uid=uid, y=yb, s=prob_attack)).groupby("uid").agg(y=("y", "max"), s=("s", "mean"))
        flag = a["s"] > 0.5
        m["agent_dr"] = flag[a["y"]].mean(); m["agent_fpr"] = flag[~a["y"]].mean()
    return m


def cv(d, feats, make_model, name, extra=None, folds=K_FOLDS):
    X, y, g = d[feats].values, d["label"].values, d["uid"].values
    rows, cm = [], np.zeros((7, 7), int)
    for k, (tr, te) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=7).split(X, y, g)):
        t0 = time.time()
        mdl = make_model().fit(X[tr], y[tr])
        p = mdl.predict(X[te]); pa = 1 - mdl.predict_proba(X[te])[:, 0]
        r = metrics(y[te], p, pa, g[te]); r.update(fold=k, method=name, n_features=len(feats),
                                                   fit_seconds=time.time() - t0)
        if extra: r.update(extra)
        rows.append(r); cm += confusion_matrix(y[te], p, labels=range(7))
    return rows, cm


def summarise(rows, by="method"):
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns.drop(["fold"], errors="ignore")
    return df.groupby(by)[num].agg(["mean", "std"])


def filter_subsets(k):
    s = pd.read_csv("data/selection.csv"); X, y = s[FEATURES].values, s["label"].values
    mi = mutual_info_classif(X, y, random_state=0)
    fv, _ = f_classif(X, y); fv = np.nan_to_num(fv)
    imp = RandomForestClassifier(200, class_weight="balanced", random_state=0, n_jobs=1).fit(X, y).feature_importances_
    top = lambda sc: [FEATURES[i] for i in np.argsort(-sc)[:k]]
    return {"MI top-k": top(mi), "ANOVA-F top-k": top(fv), "RF-importance top-k": top(imp)}


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"E1", "E2", "E3", "E4", "E5", "E6", "E7"}
    main = load([f"main_s{s}" for s in (42, 43, 44)])
    runs = json.load(open("results/fs_runs.json"))
    meta = ["HWGO", "WaOA", "GOA", "BPSO", "BGA", "BGWO"]
    best = {}
    for mth in meta + [m for m in {r["method"] for r in runs} if m.startswith("HWGO ")]:
        rs = [r for r in runs if r["method"] == mth]
        if rs:
            best[mth] = min(rs, key=lambda r: r["fitness"])["mask"]     # chosen on selection scenario only
    HW = best["HWGO"]
    json.dump(best, open("results/selected_subsets.json", "w"), indent=1)

    if "E1" in which:
        subsets = {"All features": FEATURES, **{m: best[m] for m in meta}, **filter_subsets(len(HW))}
        allrows, cms = [], {}
        for name, fs in subsets.items():
            r, cm = cv(main, fs, rf, name); allrows += r; cms[name] = cm.tolist()
            print("E1", name, len(fs), round(np.mean([x["macro_f1"] for x in r]), 4), flush=True)
        pd.DataFrame(allrows).to_csv("results/e1_fs_methods.csv", index=False)
        json.dump(cms, open("results/e1_confusion.json", "w"))
        # ablations of HWGO (subsets chosen by each variant)
        abl = []
        for name in [m for m in best if m.startswith("HWGO")]:
            r, _ = cv(main, best[name], rf, name); abl += r
            print("E1-abl", name, round(np.mean([x["macro_f1"] for x in r]), 4), flush=True)
        pd.DataFrame(abl).to_csv("results/e1_hwgo_ablation.csv", index=False)

    if "E2" in which:
        models = {
            "Random forest": rf,
            "Extra trees": lambda: ExtraTreesClassifier(200, class_weight="balanced", n_jobs=1, random_state=0),
            "Hist. gradient boosting": lambda: HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0),
            "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=0)),
            "k-NN (k=5)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
            "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        }
        rows = []
        for name, mk in models.items():
            r, _ = cv(main, HW, mk, name); rows += r
            print("E2", name, round(np.mean([x["macro_f1"] for x in r]), 4), flush=True)
        pd.DataFrame(rows).to_csv("results/e2_classifiers.csv", index=False)

    if "E3" in which:
        X, y, g = main[FEATURES].values, main["label"].values, main["uid"].values
        rows = []
        for k, (tr, te) in enumerate(StratifiedGroupKFold(K_FOLDS, shuffle=True, random_state=7).split(X, y, g)):
            yb_tr, yb_te = y[tr] > 0, y[te] > 0
            rep = main["reputation"].values
            # reputation threshold tuned on the training folds
            ths = np.quantile(rep[tr], np.linspace(0.01, 0.6, 60))
            th = max(ths, key=lambda t: f1_score(yb_tr, rep[tr] < t))
            for name, flag, score in [("Reputation threshold", rep[te] < th, 1 - rep[te])]:
                rows.append(dict(method=name, fold=k, dr=flag[yb_te].mean(), fpr=flag[~yb_te].mean(),
                                 binary_f1=f1_score(yb_te, flag)))
            for fs_name, fs in [("all", FEATURES), ("HWGO", HW)]:
                iso = IsolationForest(n_estimators=200, contamination=float(yb_tr.mean()), random_state=0).fit(main.loc[tr, fs].values)
                flag = iso.predict(main.loc[te, fs].values) == -1
                rows.append(dict(method=f"Isolation forest ({fs_name})", fold=k, dr=flag[yb_te].mean(),
                                 fpr=flag[~yb_te].mean(), binary_f1=f1_score(yb_te, flag)))
            print("E3 fold", k, flush=True)
        pd.DataFrame(rows).to_csv("results/e3_binary_baselines.csv", index=False)

    if "E4" in which:
        rows = []
        for grp, cols in FEATURE_GROUPS.items():
            fs = [f for f in FEATURES if f not in cols]
            r, _ = cv(main, fs, lambda: rf(100), f"without {grp}"); rows += r
            print("E4", grp, round(np.mean([x["macro_f1"] for x in r]), 4), flush=True)
        r, _ = cv(main, FEATURES, lambda: rf(100), "all groups"); rows += r
        pd.DataFrame(rows).to_csv("results/e4_group_ablation.csv", index=False)

    if any(e in which for e in ("E5", "E6")):
        mlp = lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=0))
        finals = {"RF": rf(300).fit(main[HW].values, main["label"].values),
                  "MLP": mlp().fit(main[HW].values, main["label"].values)}

        def test_on(model, d, fs, tag, val):
            p = model.predict(d[fs].values); pa = 1 - model.predict_proba(d[fs].values)[:, 0]
            r = metrics(d["label"].values, p, pa, d["uid"].values); r.update(setting=tag, value=val); return r

    if "E5" in which:
        rows = []
        for f in (5, 10, 20, 30, 40):
            d = load([f"density_{f:02d}"])
            for mn, mdl in finals.items():
                rows.append(test_on(mdl, d, HW, f"density {mn}", f))
            print("E5", f, round(rows[-1]["macro_f1"], 4), flush=True)
        pd.DataFrame(rows).to_csv("results/e5_density.csv", index=False)

    if "E6" in which:
        aug = load([f"main_s{s}" for s in (42, 43, 44)] + ["evtrain_s60", "evtrain_s61"])
        augs = {"RF": rf(300).fit(aug[HW].values, aug["label"].values), "MLP": mlp().fit(aug[HW].values, aug["label"].values)}
        rows = []
        for e in (0, 25, 50, 75, 100):
            d = load([f"evasion_{e:03d}"])
            for mn in ("RF", "MLP"):
                rows.append(test_on(finals[mn], d, HW, f"{mn} trained e=0", e / 100))
                rows.append(test_on(augs[mn], d, HW, f"{mn} trained e=0 and e=0.5", e / 100))
            print("E6", e, round(rows[-2]["macro_f1"], 4), round(rows[-1]["macro_f1"], 4), flush=True)
        pd.DataFrame(rows).to_csv("results/e6_evasion.csv", index=False)

    if "E7" in which:
        X, y, g = main[HW].values, main["label"].values, main["uid"].values
        Xa = main[FEATURES].values
        rows = []
        for held in range(1, 7):
            for k, (tr, te) in enumerate(StratifiedGroupKFold(K_FOLDS, shuffle=True, random_state=7).split(X, y, g)):
                trm = tr[y[tr] != held]
                mdl = RandomForestClassifier(200, max_samples=0.5, class_weight="balanced", random_state=0,
                                             n_jobs=1).fit(X[trm], y[trm] > 0)
                flag = mdl.predict(X[te])
                ag = pd.DataFrame(dict(uid=g[te], y=y[te], s=mdl.predict_proba(X[te])[:, 1])).groupby("uid").agg(y=("y", "max"), s=("s", "mean"))
                rows.append(dict(held_out=CLASS_NAMES[held], fold=k, model="RF (supervised, class unseen)",
                                 dr_unseen=flag[y[te] == held].mean(), fpr=flag[y[te] == 0].mean(),
                                 agent_dr_unseen=(ag.s[ag.y == held] > 0.5).mean()))
                iso = IsolationForest(n_estimators=200, contamination=0.2, random_state=0).fit(Xa[trm])
                fl = iso.predict(Xa[te]) == -1
                rows.append(dict(held_out=CLASS_NAMES[held], fold=k, model="Isolation forest (all features)",
                                 dr_unseen=fl[y[te] == held].mean(), fpr=fl[y[te] == 0].mean()))
            print("E7", CLASS_NAMES[held], flush=True)
        pd.DataFrame(rows).to_csv("results/e7_unseen.csv", index=False)
