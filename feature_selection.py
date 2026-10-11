"""
Wrapper feature selection with binary metaheuristics under a fixed budget of
fitness evaluations (FEs).

Positions live in [0, 1]^D; feature d is selected when x_d > 0.5.
Fitness (minimised) = alpha * (1 - macroF1_val) + (1 - alpha) * |S| / D

Implemented: BPSO, BGA, BGWO, WaOA, GOA and the hybrid Walrus-Gazelle
optimiser (HWGO) used in the paper, plus HWGO ablations.

NOTE. HWGO follows the hybridisation idea of Selvaraj, Kavitha Devi and Selvi
(ICNGCS 2025): WaOA's feeding and escape phases around GOA's Brownian/Levy
movement and predator-success-rate (PSR) jumps. Align the update equations with
that paper's exact formulation before submission.
"""
from __future__ import annotations

import math
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score


class WrapperFitness:
    """Caches evaluations; counts every call against the FE budget."""

    def __init__(self, Xtr, ytr, Xva, yva, alpha=0.98, n_trees=40, seed=0):
        self.Xtr, self.ytr, self.Xva, self.yva = Xtr, ytr, Xva, yva
        self.D = Xtr.shape[1]; self.alpha = alpha; self.n_trees = n_trees; self.seed = seed
        self.cache = {}; self.fe = 0; self.history = []; self.best = (np.inf, None)

    def __call__(self, x):
        mask = x > 0.5
        if not mask.any():
            mask[np.argmax(x)] = True
        key = mask.tobytes()
        self.fe += 1
        if key not in self.cache:
            clf = ExtraTreesClassifier(self.n_trees, n_jobs=1, random_state=self.seed, class_weight="balanced")
            clf.fit(self.Xtr[:, mask], self.ytr)
            f1 = f1_score(self.yva, clf.predict(self.Xva[:, mask]), average="macro")
            self.cache[key] = (self.alpha * (1 - f1) + (1 - self.alpha) * mask.sum() / self.D, f1)
        fit, f1 = self.cache[key]
        if fit < self.best[0]:
            self.best = (fit, mask.copy(), f1)
        self.history.append(self.best[0])
        return fit


def _levy(rng, size, beta=1.5):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
           (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sig, size); v = rng.normal(0, 1, size)
    return 0.05 * u / np.abs(v) ** (1 / beta)


def _init(rng, n, D, fit, obl=False):
    X = rng.random((n, D))
    if obl:                                       # opposition-based initialisation
        Xo = 1 - X
        F = np.array([fit(x) for x in X]); Fo = np.array([fit(x) for x in Xo])
        allX = np.vstack([X, Xo]); allF = np.concatenate([F, Fo])
        keep = np.argsort(allF)[:n]
        return allX[keep], allF[keep]
    return X, np.array([fit(x) for x in X])


def _budget(fit, budget):
    return fit.fe < budget


# --------------------------------------------------------------- baselines
def bpso(fit, D, budget, n=10, seed=0):
    rng = np.random.default_rng(seed)
    X, F = _init(rng, n, D, fit); V = rng.normal(0, 0.1, (n, D))
    P, PF = X.copy(), F.copy(); g = P[PF.argmin()].copy()
    t, T = 0, budget // n
    while _budget(fit, budget):
        w = 0.9 - 0.5 * t / max(1, T)
        for i in range(n):
            V[i] = w * V[i] + 2 * rng.random(D) * (P[i] - X[i]) + 2 * rng.random(D) * (g - X[i])
            prob = 1 / (1 + np.exp(-10 * (V[i])))    # S-shaped transfer
            X[i] = np.where(rng.random(D) < prob, 0.75, 0.25)
            f = fit(X[i])
            if f < PF[i]:
                P[i], PF[i] = X[i].copy(), f
            if not _budget(fit, budget):
                break
        g = P[PF.argmin()].copy(); t += 1


def bga(fit, D, budget, n=10, seed=0, pc=0.9, pm=None):
    rng = np.random.default_rng(seed); pm = pm or 1.0 / D
    X, F = _init(rng, n, D, fit)
    while _budget(fit, budget):
        kids = []
        for _ in range(n // 2):
            a = min(rng.choice(n, 2, replace=False), key=lambda k: F[k])
            b = min(rng.choice(n, 2, replace=False), key=lambda k: F[k])
            c1, c2 = X[a].copy(), X[b].copy()
            if rng.random() < pc:
                cut = rng.integers(1, D); c1[cut:], c2[cut:] = X[b][cut:], X[a][cut:]
            for c in (c1, c2):
                flip = rng.random(D) < pm; c[flip] = 1 - c[flip]; kids.append(c)
        KF = []
        for c in kids:
            if not _budget(fit, budget):
                break
            KF.append(fit(c))
        kids = kids[:len(KF)]
        allX = np.vstack([X] + kids) if kids else X; allF = np.concatenate([F, KF])
        keep = np.argsort(allF)[:n]; X, F = allX[keep], allF[keep]


def bgwo(fit, D, budget, n=10, seed=0):
    rng = np.random.default_rng(seed)
    X, F = _init(rng, n, D, fit); T = budget // n; t = 0
    while _budget(fit, budget):
        order = np.argsort(F); A_, B_, C_ = X[order[0]], X[order[1]], X[order[2]]
        a = 2 - 2 * t / max(1, T)
        for i in range(n):
            new = np.zeros(D)
            for L in (A_, B_, C_):
                A = 2 * a * rng.random(D) - a; C = 2 * rng.random(D)
                new += L - A * np.abs(C * L - X[i])
            X[i] = np.clip(new / 3, 0, 1); F[i] = fit(X[i])
            if not _budget(fit, budget):
                break
        t += 1


def waoa(fit, D, budget, n=10, seed=0):
    """Walrus Optimisation Algorithm (Trojovsky & Dehghani, 2023), binary via threshold."""
    rng = np.random.default_rng(seed)
    X, F = _init(rng, n, D, fit); t = 1
    while _budget(fit, budget):
        for i in range(n):
            sw = X[F.argmin()]
            # phase 1: feeding (exploration toward strongest walrus)
            I = rng.integers(1, 3)
            xn = np.clip(X[i] + rng.random(D) * (sw - I * X[i]), 0, 1); fn = fit(xn)
            if fn < F[i]:
                X[i], F[i] = xn, fn
            if not _budget(fit, budget): break
            # phase 2: migration toward / away from a random walrus
            k = rng.choice([j for j in range(n) if j != i])
            I = rng.integers(1, 3)
            step = (X[k] - I * X[i]) if F[k] < F[i] else (X[i] - X[k])
            xn = np.clip(X[i] + rng.random(D) * step, 0, 1); fn = fit(xn)
            if fn < F[i]:
                X[i], F[i] = xn, fn
            if not _budget(fit, budget): break
            # phase 3: escaping / fighting (local exploitation, shrinking radius)
            lo, hi = 0 / t, 1 / t
            xn = np.clip(X[i] + (lo + (hi - rng.random(D) * lo)) * (2 * rng.random(D) - 1), 0, 1); fn = fit(xn)
            if fn < F[i]:
                X[i], F[i] = xn, fn
            if not _budget(fit, budget): break
        t += 1


def goa(fit, D, budget, n=10, seed=0, psrs=0.34, s=0.88):
    """Gazelle Optimisation Algorithm (Agushaka et al., 2023), binary via threshold."""
    rng = np.random.default_rng(seed)
    X, F = _init(rng, n, D, fit); T = max(1, budget // (2 * n)); t = 0
    while _budget(fit, budget):
        elite = X[F.argmin()]
        CF = (1 - t / T) ** (2 * t / T) if t < T else 0.0
        for i in range(n):
            R = rng.random(D)
            if rng.random() < 0.5:                 # grazing: Brownian exploitation
                RB = rng.normal(0, 1, D)
                xn = X[i] + s * R * RB * (elite - RB * X[i])
            else:                                  # predator spotted: Levy exploration
                RL = _levy(rng, D); mu = 1 if rng.random() < 0.5 else -1
                xn = X[i] + s * mu * CF * RL * (elite - RL * X[i]) if i < n / 2 else \
                    X[i] + s * mu * R * RL * (elite - RL * X[i])
            xn = np.clip(xn, 0, 1); fn = fit(xn)
            if fn < F[i]:
                X[i], F[i] = xn, fn
            if not _budget(fit, budget): break
        # predator success rate (PSR) effect
        for i in range(n):
            if not _budget(fit, budget): break
            r = rng.random()
            if r < psrs:
                U = (rng.random(D) < psrs).astype(float)
                xn = X[i] + CF * rng.random(D) * U
            else:
                a, b = rng.choice(n, 2, replace=False)
                xn = X[i] + (psrs * (1 - r) + r) * (X[a] - X[b])
            xn = np.clip(xn, 0, 1); fn = fit(xn)
            if fn < F[i]:
                X[i], F[i] = xn, fn
        t += 1


def hwgo(fit, D, budget, n=10, seed=0, obl=True, use_goa=True, use_waoa=True, use_psr=True,
         psrs=0.34, s=0.88):
    """Hybrid Walrus-Gazelle Optimiser.

    Per iteration and walrus/gazelle i:
      A (WaOA feeding)  : move toward the strongest individual            [exploration]
      B (GOA movement)  : Levy flight with prob p_t = 1 - t/T, otherwise
                          Brownian grazing around the elite               [adaptive switch]
      C (WaOA escape)   : local search with radius shrinking as 1/t        [exploitation]
    After each iteration:
      D (GOA PSR jump)  : the worse half of the population takes a PSR jump [diversity]
    Greedy selection after every move. OBL initialisation.
    """
    rng = np.random.default_rng(seed)
    X, F = _init(rng, n, D, fit, obl=obl)
    per_iter = n * (int(use_waoa) * 2 + int(use_goa)) + (n // 2 if use_psr else 0)
    T = max(1, (budget - fit.fe) // max(1, per_iter)); t = 1

    def accept(i, xn):
        xn = np.clip(xn, 0, 1); fn = fit(xn)
        if fn < F[i]:
            X[i], F[i] = xn, fn

    while _budget(fit, budget):
        CF = (1 - t / (T + 1)) ** (2 * t / (T + 1))
        for i in range(n):
            best = X[F.argmin()]
            if use_waoa:
                I = rng.integers(1, 3)
                accept(i, X[i] + rng.random(D) * (best - I * X[i]))
                if not _budget(fit, budget): break
            if use_goa:
                R = rng.random(D)
                if rng.random() < 1 - t / (T + 1):
                    RL = _levy(rng, D); mu = 1 if rng.random() < 0.5 else -1
                    accept(i, X[i] + s * mu * CF * RL * (best - RL * X[i]) + 0.1 * CF * (2 * R - 1))
                else:
                    RB = rng.normal(0, 1, D)
                    accept(i, X[i] + s * R * RB * (best - RB * X[i]))
                if not _budget(fit, budget): break
            if use_waoa:
                hi = 1 / t
                accept(i, X[i] + hi * (1 - rng.random(D)) * (2 * rng.random(D) - 1))
                if not _budget(fit, budget): break
        if use_psr:
            for i in np.argsort(F)[n // 2:]:
                if not _budget(fit, budget): break
                r = rng.random()
                if r < psrs:
                    U = (rng.random(D) < psrs).astype(float)
                    accept(i, X[i] + CF * (2 * rng.random(D) - 1) * U)
                else:
                    a, b = rng.choice(n, 2, replace=False)
                    accept(i, X[i] + (psrs * (1 - r) + r) * (X[a] - X[b]))
        t += 1


OPTIMISERS = {
    "BPSO": bpso, "BGA": bga, "BGWO": bgwo, "WaOA": waoa, "GOA": goa, "HWGO": hwgo,
}
ABLATIONS = {
    "HWGO": dict(), "HWGO w/o OBL": dict(obl=False), "HWGO w/o PSR": dict(use_psr=False),
    "HWGO w/o GOA (WaOA core)": dict(use_goa=False), "HWGO w/o WaOA (GOA core)": dict(use_waoa=False),
}


def run_optimiser(name, fit, D, budget, seed, **kw):
    t0 = time.time()
    fn = OPTIMISERS.get(name, hwgo)
    fn(fit, D, budget, seed=seed, **kw)
    return dict(method=name, seed=seed, fitness=fit.best[0], mask=fit.best[1], val_f1=fit.best[2],
                n_features=int(fit.best[1].sum()), fe=fit.fe, seconds=time.time() - t0,
                history=list(fit.history[:budget]))
