"""
AgentNet-IDS simulator.

Discrete-time simulation of an open agentic network: agents register in a
registry, discover providers by capability and reputation, delegate tasks,
return outputs, rate each other and sub-delegate. Six adversarial classes are
injected alongside heterogeneous benign agents (including benign "confusers"
that resemble attackers).

Classes
  0 benign
  1 sybil          - bursts of fake identities that promote each other
  2 poisoner       - ballot-stuffing / bad-mouthing raters
  3 injector       - prompt-injection relay (outputs hijack downstream agents)
  4 colluder       - ring that routes tasks and ratings internally
  5 dropper        - grayhole: accepts tasks, silently drops a fraction
  6 flooder        - request flooding / resource exhaustion

The simulator emits one row per (agent, window) with 52 behavioural features.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

CLASS_NAMES = ["benign", "sybil", "poisoner", "injector", "colluder", "dropper", "flooder"]

FEATURES = [
    # traffic (8)
    "req_out", "req_in", "uniq_out", "uniq_in", "out_in_ratio", "iat_cv", "payload_mean", "payload_std",
    # service (8)
    "accept_rate", "completion_rate", "timeout_rate", "latency_mean", "latency_p95", "quality_mean",
    "quality_std", "subdelegation_rate",
    # rating (8)
    "ratings_given", "rating_given_mean", "rating_dev_consensus", "extreme_rating_frac", "rating_given_std",
    "rating_recv_mean", "rater_diversity_recv", "rating_recv_std",
    # graph (8)
    "reciprocity", "clustering_coef", "top3_partner_share", "pagerank", "in_degree", "out_degree",
    "cycle_participation", "kcore",
    # identity (6)
    "account_age", "reg_burst_size", "desc_nn_sim", "fingerprint_share", "cap_breadth", "cap_claim_gap",
    # content (8)
    "instr_score_mean", "instr_score_max", "url_rate", "out_entropy", "downstream_toolcall_mean",
    "downstream_anomaly_frac", "toolcalls_own", "msg_len_mean",
    # temporal (6)
    "rep_delta", "act_var_hist", "quality_var_hist", "req_out_ewma", "rating_dev_ewma", "instr_ewma",
]
FEATURE_GROUPS = {
    "traffic": FEATURES[0:8], "service": FEATURES[8:16], "rating": FEATURES[16:24],
    "graph": FEATURES[24:32], "identity": FEATURES[32:38], "content": FEATURES[38:46],
    "temporal": FEATURES[46:52],
}
assert len(FEATURES) == 52


@dataclass
class SimConfig:
    n_agents: int = 600
    n_windows: int = 30
    attack_frac: float = 0.20          # share of agents that are adversarial
    attack_mix: tuple = (1, 1, 1, 1, 1, 1)  # relative weights of the 6 attack classes
    evasion: float = 0.0               # 0 = naive attackers, 1 = strongly evasive
    n_caps: int = 10
    desc_dim: int = 16
    seed: int = 0


@dataclass
class Agent:
    idx: int
    cls: int
    sub: str
    join: int
    caps: np.ndarray
    desc: np.ndarray
    fp: int
    quality: float
    lam: float
    group: int = -1                    # attack group (sybil cluster, ring, poison group)
    targets: list = field(default_factory=list)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


class AgentNetSimulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.agents: list[Agent] = []
        self._build_population()

    # ------------------------------------------------------------------ setup
    def _build_population(self):
        c, rng = self.cfg, self.rng
        n_att = int(round(c.n_agents * c.attack_frac))
        w = np.array(c.attack_mix, float); w /= w.sum()
        att_counts = np.floor(w * n_att).astype(int)
        att_counts[: n_att - att_counts.sum()] += 1
        n_ben = c.n_agents - n_att
        next_fp = 0

        # benign sub-populations (confusers included)
        subs = rng.choice(["regular", "hub", "newcomer", "lowq", "critic", "aggregator", "fleet"],
                          size=n_ben, p=[0.46, 0.08, 0.12, 0.10, 0.08, 0.08, 0.08])
        fleet_ids = np.where(subs == "fleet")[0]
        fleet_groups = np.array_split(fleet_ids, max(1, len(fleet_ids) // 6))
        fleet_desc = {}
        for g, ids in enumerate(fleet_groups):
            base = _unit(rng.normal(size=c.desc_dim))
            for i in ids:
                fleet_desc[i] = (base, next_fp)
            next_fp += 1

        for i in range(n_ben):
            s = subs[i]
            caps = (rng.random(c.n_caps) < 0.25).astype(float)
            if caps.sum() == 0:
                caps[rng.integers(c.n_caps)] = 1
            if s == "fleet":
                base, fp = fleet_desc[i]
                desc = _unit(base + 0.08 * rng.normal(size=c.desc_dim))
            else:
                desc = _unit(rng.normal(size=c.desc_dim)); fp = next_fp; next_fp += 1
            join = int(rng.integers(1, c.n_windows - 2)) if s == "newcomer" else int(rng.integers(0, 3))
            q = rng.beta(3, 3) if s == "lowq" else rng.beta(8, 2)
            lam = rng.lognormal(1.3, 0.45) * (4.0 if s == "hub" else 1.0) * (2.0 if s == "aggregator" else 1.0)
            self.agents.append(Agent(len(self.agents), 0, s, join, caps, desc, fp, q, lam))

        ev = c.evasion
        # sybils: clusters of 6-10 identities registered in the same window
        remaining = att_counts[0]; gid = 0
        while remaining > 0:
            k = int(min(remaining, rng.integers(6, 11)))
            base = _unit(rng.normal(size=c.desc_dim)); join = int(rng.integers(0, c.n_windows // 2))
            fp = next_fp; next_fp += 1
            caps0 = (rng.random(c.n_caps) < 0.6).astype(float)
            for _ in range(k):
                noise = 0.05 + 0.5 * ev
                desc = _unit(base + noise * rng.normal(size=c.desc_dim))
                j = join + (int(rng.integers(0, 4)) if rng.random() < ev else 0)
                my_fp = fp if rng.random() > 0.7 * ev else (next_fp := next_fp + 1)
                self.agents.append(Agent(len(self.agents), 1, "sybil", min(j, c.n_windows - 3), caps0.copy(), desc,
                                         my_fp, rng.beta(3, 4), rng.lognormal(1.1, 0.4), group=gid))
            remaining -= k; gid += 1

        def plain(cls, sub, q=(8, 2), lam_mu=1.3, group=-1):
            caps = (rng.random(c.n_caps) < 0.3).astype(float)
            if caps.sum() == 0:
                caps[rng.integers(c.n_caps)] = 1
            nonlocal next_fp
            a = Agent(len(self.agents), cls, sub, int(rng.integers(0, 4)), caps, _unit(rng.normal(size=c.desc_dim)),
                      next_fp, rng.beta(*q), rng.lognormal(lam_mu, 0.45), group=group)
            next_fp += 1
            self.agents.append(a)
            return a

        # poisoners: groups of 4-6 that boost beneficiaries and bad-mouth top providers
        remaining = att_counts[1]; gid = 100
        while remaining > 0:
            k = int(min(remaining, rng.integers(4, 7)))
            for _ in range(k):
                plain(2, "poisoner", group=gid)
            remaining -= k; gid += 1
        for _ in range(att_counts[2]):
            plain(3, "injector")
        remaining = att_counts[3]; gid = 200
        while remaining > 0:
            k = int(min(remaining, rng.integers(4, 7)))
            for _ in range(k):
                plain(4, "colluder", q=(4, 3), group=gid)
            remaining -= k; gid += 1
        for _ in range(att_counts[4]):
            plain(5, "dropper")
        for _ in range(att_counts[5]):
            plain(6, "flooder", lam_mu=1.2)

        # poisoners need beneficiaries (low-quality agents they promote) and victims
        benign_idx = [a.idx for a in self.agents if a.cls == 0]
        for a in self.agents:
            if a.cls == 2:
                a.targets = list(rng.choice(benign_idx, size=3, replace=False))

        self.N = len(self.agents)
        self.cls = np.array([a.cls for a in self.agents])
        self.desc = np.stack([a.desc for a in self.agents])
        self.caps = np.stack([a.caps for a in self.agents])
        self.fp = np.array([a.fp for a in self.agents])
        self.join = np.array([a.join for a in self.agents])
        self.quality = np.array([a.quality for a in self.agents])
        self.lam = np.array([a.lam for a in self.agents])
        self.group = np.array([a.group for a in self.agents])
        self.sub = np.array([a.sub for a in self.agents])

    # ------------------------------------------------------------- simulation
    def run(self) -> pd.DataFrame:
        c, rng, N = self.cfg, self.rng, self.N
        ev = c.evasion
        rep = np.full(N, 0.6)              # Bayesian reputation seen by the registry
        rep_a = np.full(N, 3.0); rep_b = np.full(N, 2.0)
        infected = np.zeros(N)             # prompt-injection infection level (decays)
        hist = {k: np.zeros((c.n_windows, N)) for k in ["req_out", "quality", "rating_dev", "instr"]}
        ewma = {k: np.zeros(N) for k in ["req_out", "rating_dev", "instr"]}

        # static identity features
        sim = self.desc @ self.desc.T
        np.fill_diagonal(sim, -1)
        desc_nn = sim.max(1)
        fp_share = np.array([(self.fp == f).sum() - 1 for f in self.fp])
        burst = np.zeros(N)
        for i in range(N):
            same = (self.join == self.join[i]) & (sim[i] > 0.85)
            burst[i] = same.sum()
        cap_breadth = self.caps.sum(1)

        rows = []
        for w in range(c.n_windows):
            active = self.join <= w
            act_idx = np.where(active)[0]
            # on-off behaviour: attackers are 'on' with prob decreasing in evasion
            on = np.ones(N, bool)
            att = self.cls > 0
            on[att] = rng.random(att.sum()) < (0.9 - 0.45 * ev)

            ev_src, ev_dst, ev_t, ev_pay, ev_acc, ev_done, ev_to, ev_lat, ev_q = ([] for _ in range(9))
            ev_instr, ev_url, ev_ent, ev_len, ev_sub = ([] for _ in range(5))
            ratings = []                   # (rater, ratee, value)
            load = np.zeros(N)
            downstream_tc = np.zeros(N); downstream_cnt = np.zeros(N)
            toolcalls = np.zeros(N)
            sub_edges = []

            prov_pool = act_idx
            base_w = np.exp(3.0 * rep[prov_pool])
            n_req = rng.poisson(self.lam[act_idx] * (1 + 1.5 * infected[act_idx]))
            flood = (self.cls[act_idx] == 6) & on[act_idx]
            n_req[flood] = rng.poisson(self.lam[act_idx][flood] * rng.uniform(6, 12) * (1 - 0.55 * ev))

            for i, nr in zip(act_idx, n_req):
                ci, oi = self.cls[i], on[i]
                if nr == 0:
                    continue
                # request times inside the window (flooders: bursty)
                if ci == 6 and oi:
                    centres = rng.random(max(1, int(2 + 3 * ev)))
                    t = np.clip(rng.choice(centres, nr) + rng.normal(0, 0.01 + 0.08 * ev, nr), 0, 1)
                else:
                    t = rng.random(nr)
                for k in range(nr):
                    # choose provider
                    if ci == 1 and oi and rng.random() < 0.65 * (1 - 0.4 * ev):
                        pool = np.where((self.group == self.group[i]) & active & (np.arange(N) != i))[0]
                        j = rng.choice(pool) if len(pool) else rng.choice(prov_pool)
                    elif ci == 4 and oi and rng.random() < 0.7 * (1 - 0.4 * ev):
                        pool = np.where((self.group == self.group[i]) & active & (np.arange(N) != i))[0]
                        j = rng.choice(pool) if len(pool) else rng.choice(prov_pool)
                    elif ci == 2 and oi and rng.random() < 0.6 * (1 - 0.4 * ev):
                        j = rng.choice(self.agents[i].targets + [p for p in prov_pool[np.argsort(-rep[prov_pool])[:5]]])
                    else:
                        cap = rng.integers(c.n_caps)
                        m = self.caps[prov_pool, cap] + 0.15
                        p = base_w * m
                        p[prov_pool == i] = 0
                        j = rng.choice(prov_pool, p=p / p.sum())
                    load[j] += 1
                    pay = rng.lognormal(7.0, 0.5) * (0.3 if ci == 6 and oi else 1.0)
                    # provider behaviour
                    cj, oj = self.cls[j], on[j]
                    overload = load[j] / (4 * self.lam[j] + 5)
                    acc = rng.random() > min(0.9, 0.02 + 0.3 * max(0, overload - 1))
                    done = acc
                    if acc and cj == 5 and oj:
                        done = rng.random() > (0.55 - 0.25 * ev)
                    if acc and self.sub[j] == "lowq":
                        done = rng.random() > 0.12   # benign but unreliable
                    timeout = acc and not done
                    lat = rng.lognormal(0.0 + 0.4 * min(overload, 3), 0.35) * (3.0 if timeout else 1.0)
                    q = float(np.clip(self.quality[j] + rng.normal(0, 0.1), 0, 1)) if done else 0.0
                    # content of the response
                    if cj == 3 and oj and rng.random() < 0.65 * (1 - 0.45 * ev):
                        instr = rng.beta(6 - 3 * ev, 3 + 2 * ev); url = rng.poisson(1.5)
                        infected[i] = min(1.0, infected[i] + 0.35)
                        tc = rng.poisson(3.0 - 1.2 * ev)
                    elif self.sub[j] == "aggregator":
                        instr = rng.beta(2, 8); url = rng.poisson(0.4); tc = rng.poisson(0.8)
                    elif infected[j] > 0.3 and rng.random() < 0.3:     # benign victims relay unknowingly
                        instr = rng.beta(3, 5); url = rng.poisson(0.6); tc = rng.poisson(1.2)
                    else:
                        instr = rng.beta(1.5, 12); url = rng.poisson(0.15); tc = rng.poisson(0.5)
                    downstream_tc[j] += tc; downstream_cnt[j] += 1
                    toolcalls[i] += tc
                    ent = rng.normal(4.2, 0.3) + (0.4 * instr if cj == 3 else 0)
                    ln = rng.lognormal(6.0, 0.4) * (1.3 if cj == 3 and oj else 1.0)
                    # sub-delegation
                    sub_p = 0.35 if self.sub[j] == "aggregator" else 0.08
                    if cj == 4 and oj:
                        sub_p = 0.5
                    subd = done and rng.random() < sub_p
                    if subd:
                        if cj == 4 and oj:
                            pool = np.where((self.group == self.group[j]) & active & (np.arange(N) != j))[0]
                            s2 = rng.choice(pool) if len(pool) else rng.choice(prov_pool)
                        else:
                            s2 = rng.choice(prov_pool)
                        if s2 != j:
                            sub_edges.append((j, s2))
                    ev_src.append(i); ev_dst.append(j); ev_t.append(t[k]); ev_pay.append(pay); ev_acc.append(acc)
                    ev_done.append(done); ev_to.append(timeout); ev_lat.append(lat); ev_q.append(q)
                    ev_instr.append(instr); ev_url.append(url); ev_ent.append(ent); ev_len.append(ln); ev_sub.append(subd)
                    # rating by consumer i of provider j
                    if rng.random() < (0.7 if ci == 0 else 0.95):
                        honest = 1 + 4 * q if done else 1.0
                        r = honest + rng.normal(0, 0.4)
                        if self.sub[i] == "critic":
                            r -= 1.2
                        if oi and ci in (1, 4) and self.group[j] == self.group[i]:
                            r = 5 - rng.random() * 1.5 * ev
                        if oi and ci == 2:
                            if j in self.agents[i].targets:
                                r = 5 - rng.random() * 1.5 * ev
                            else:
                                r = 1 + rng.random() * 2.0 * ev
                        ratings.append((i, j, float(np.clip(r, 1, 5))))

            E = pd.DataFrame(dict(src=ev_src, dst=ev_dst, t=ev_t, pay=ev_pay, acc=ev_acc, done=ev_done,
                                  to=ev_to, lat=ev_lat, q=ev_q, instr=ev_instr, url=ev_url, ent=ev_ent,
                                  ln=ev_len, sub=ev_sub))
            R = pd.DataFrame(ratings, columns=["rater", "ratee", "r"])
            consensus = R.groupby("ratee")["r"].median() if len(R) else pd.Series(dtype=float)
            if len(R):
                R["dev"] = (R["r"] - R["ratee"].map(consensus)).abs()
                R["ext"] = (R["r"] >= 4.8) | (R["r"] <= 1.2)

            # interaction graph (requests + sub-delegations)
            G = nx.DiGraph(); G.add_nodes_from(act_idx)
            for (a, b), cnt in E.groupby(["src", "dst"]).size().items():
                G.add_edge(a, b, weight=cnt)
            for a, b in sub_edges:
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
            pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {}
            UG = nx.Graph(G)
            UG.remove_edges_from(nx.selfloop_edges(UG))
            clus = nx.clustering(UG)
            kc = nx.core_number(UG) if UG.number_of_edges() else {}
            recip = {}
            for n in G.nodes:
                succ = set(G.successors(n)); pred = set(G.predecessors(n))
                recip[n] = len(succ & pred) / max(1, len(succ | pred))
            # short delegation cycles (length 2-3) through each node
            SG = nx.DiGraph(); SG.add_edges_from(sub_edges + list(zip(E["src"], E["dst"])))
            cyc = np.zeros(N)
            for n in SG.nodes:
                succ = set(SG.successors(n))
                cnt = sum(1 for s in succ if SG.has_edge(s, n))
                cnt += sum(1 for s in succ for s2 in SG.successors(s) if s2 != n and SG.has_edge(s2, n))
                cyc[n] = cnt

            # reputation update (Beta-Bayesian, decayed) - also the baseline detector's score
            rep_prev = rep.copy()
            rep_a *= 0.9; rep_b *= 0.9
            if len(R):
                g = R.groupby("ratee")["r"]
                pos = g.apply(lambda s: ((s - 1) / 4).sum()); cnt = g.size()
                rep_a[pos.index] += pos.values; rep_b[cnt.index] += (cnt - pos).values
            rep = rep_a / (rep_a + rep_b)
            # infection decays
            infected *= 0.5

            A = {k: E[k].to_numpy() for k in E.columns}
            from collections import defaultdict
            out_ix = defaultdict(list); in_ix = defaultdict(list)
            for n_, (s_, d_) in enumerate(zip(A["src"], A["dst"])):
                out_ix[s_].append(n_); in_ix[d_].append(n_)
            rg_ix = defaultdict(list); rr_ix = defaultdict(list)
            if len(R):
                RA = {k: R[k].to_numpy() for k in ["rater", "ratee", "r", "dev", "ext"]}
                for n_, (a_, b_) in enumerate(zip(RA["rater"], RA["ratee"])):
                    rg_ix[a_].append(n_); rr_ix[b_].append(n_)
            def m(x, d=0.0):
                return float(x.mean()) if len(x) else d
            def sd(x):
                return float(x.std(ddof=1)) if len(x) > 1 else 0.0
            for i in act_idx:
                f = {}
                o = np.array(out_ix.get(i, []), int); n = np.array(in_ix.get(i, []), int)
                f["req_out"] = len(o); f["req_in"] = len(n)
                f["uniq_out"] = len(set(A["dst"][o])); f["uniq_in"] = len(set(A["src"][n]))
                f["out_in_ratio"] = (len(o) + 1) / (len(n) + 1)
                if len(o) > 2:
                    d = np.diff(np.sort(A["t"][o])); f["iat_cv"] = d.std() / (d.mean() + 1e-6)
                else:
                    f["iat_cv"] = 0.0
                f["payload_mean"] = m(A["pay"][o]); f["payload_std"] = sd(A["pay"][o])
                acc = A["acc"][n].astype(float); done = A["done"][n].astype(bool)
                f["accept_rate"] = m(acc, 1.0); f["completion_rate"] = m(done.astype(float), 1.0)
                f["timeout_rate"] = m(A["to"][n].astype(float)); lat = A["lat"][n]
                f["latency_mean"] = m(lat); f["latency_p95"] = float(np.quantile(lat, 0.95)) if len(lat) else 0.0
                qd = A["q"][n][done]
                f["quality_mean"] = m(qd); f["quality_std"] = sd(qd)
                f["subdelegation_rate"] = m(A["sub"][n].astype(float))
                g = np.array(rg_ix.get(i, []), int); r = np.array(rr_ix.get(i, []), int)
                f["ratings_given"] = len(g)
                f["rating_given_mean"] = m(RA["r"][g], 3.0) if len(g) else 3.0
                f["rating_dev_consensus"] = m(RA["dev"][g]) if len(g) else 0.0
                f["extreme_rating_frac"] = m(RA["ext"][g].astype(float)) if len(g) else 0.0
                f["rating_given_std"] = sd(RA["r"][g]) if len(g) else 0.0
                f["rating_recv_mean"] = m(RA["r"][r], 3.0) if len(r) else 3.0
                f["rater_diversity_recv"] = len(set(RA["rater"][r])) / len(r) if len(r) else 1.0
                f["rating_recv_std"] = sd(RA["r"][r]) if len(r) else 0.0
                f["reciprocity"] = recip.get(i, 0.0); f["clustering_coef"] = clus.get(i, 0.0)
                if len(o):
                    vc = np.sort(np.bincount(A["dst"][o]))[::-1]; vc = vc[vc > 0]
                    f["top3_partner_share"] = vc[:3].sum() / vc.sum()
                else:
                    f["top3_partner_share"] = 0.0
                f["pagerank"] = pr.get(i, 0.0) * N
                f["in_degree"] = G.in_degree(i); f["out_degree"] = G.out_degree(i)
                f["cycle_participation"] = cyc[i]; f["kcore"] = kc.get(i, 0)
                f["account_age"] = w - self.join[i]
                f["reg_burst_size"] = burst[i]; f["desc_nn_sim"] = desc_nn[i]
                f["fingerprint_share"] = fp_share[i]; f["cap_breadth"] = cap_breadth[i]
                f["cap_claim_gap"] = cap_breadth[i] / (1 + done.sum())
                f["instr_score_mean"] = m(A["instr"][n]); f["instr_score_max"] = float(A["instr"][n].max()) if len(n) else 0.0
                f["url_rate"] = m(A["url"][n].astype(float)); f["out_entropy"] = m(A["ent"][n])
                f["downstream_toolcall_mean"] = downstream_tc[i] / max(1, downstream_cnt[i])
                cons = list(set(A["src"][n]))
                f["downstream_anomaly_frac"] = float(np.mean(infected[cons] > 0.15)) if cons else 0.0
                f["toolcalls_own"] = toolcalls[i]; f["msg_len_mean"] = m(A["ln"][n])
                f["rep_delta"] = rep[i] - rep_prev[i]
                hist["req_out"][w, i] = f["req_out"]; hist["quality"][w, i] = f["quality_mean"]
                hist["rating_dev"][w, i] = f["rating_dev_consensus"]; hist["instr"][w, i] = f["instr_score_mean"]
                lo = max(self.join[i], w - 3)
                h = hist["req_out"][lo:w + 1, i]
                f["act_var_hist"] = h.std() / (h.mean() + 1e-6) if len(h) > 1 else 0.0
                hq = hist["quality"][lo:w + 1, i]
                f["quality_var_hist"] = hq.std() if len(hq) > 1 else 0.0
                for k, col, v in [("req_out", "req_out_ewma", f["req_out"]),
                                  ("rating_dev", "rating_dev_ewma", f["rating_dev_consensus"]),
                                  ("instr", "instr_ewma", f["instr_score_mean"])]:
                    ewma[k][i] = 0.6 * ewma[k][i] + 0.4 * v if w > self.join[i] else v
                    f[col] = ewma[k][i]
                f.update(agent_id=i, window=w, label=self.cls[i], subtype=self.sub[i],
                         attack_on=bool(on[i]) if self.cls[i] > 0 else False, reputation=rep[i])
                rows.append(f)

        df = pd.DataFrame(rows)
        df[FEATURES] = df[FEATURES].fillna(0.0).astype(float)
        df["scenario_seed"] = c.seed; df["attack_frac"] = c.attack_frac; df["evasion"] = c.evasion
        return df


def simulate(**kw) -> pd.DataFrame:
    return AgentNetSimulator(SimConfig(**kw)).run()


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = simulate(n_agents=200, n_windows=8, seed=1)
    print(d.shape, round(time.time() - t0, 1), "s")
    print(d.groupby("label").size())
