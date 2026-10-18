"""Social networks between learners, in iterations (one generation and update round).

The same structures as the JAX code (sapiens/topology.py), as boolean adjacency matrices:
adj[i, j] means learner i sends completions to learner j. `centralized` is handled by the
trainer as a single learner with the pooled sample budget (standard GRPO).
"""
import numpy as np

TOPOLOGIES = ("independent", "fully", "ring", "dynamic", "periodic", "centralized")


class Topology:
    def __init__(self, cfg, rng: np.random.Generator):
        self.kind, self.n, self.cfg, self.rng = cfg["TOPOLOGY"], cfg["NUM_LEARNERS"], cfg, rng
        if self.kind not in TOPOLOGIES:
            raise ValueError(f"Unknown topology {self.kind}, expected one of {TOPOLOGIES}")
        self.home = np.arange(self.n) // cfg["GROUP_SIZE_DYNAMIC"]
        self.group = self.home.copy()
        self.visit_end = -np.ones(self.n, int)

    def adjacency(self, it):
        n, full = self.n, ~np.eye(self.n, dtype=bool)
        if self.kind in ("independent", "centralized") or n == 1:
            return np.zeros((n, n), bool)
        if self.kind == "fully":
            return full
        if self.kind == "ring":
            adj = np.zeros((n, n), bool)
            for k in range(1, self.cfg["RING_K"] // 2 + 1):
                for i in range(n):
                    adj[i, (i + k) % n] = adj[(i + k) % n, i] = True
            return adj & full
        if self.kind == "periodic":
            # isolated for T_LOW iterations, then fully connected for T_HIGH
            return full if it % (self.cfg["T_LOW"] + self.cfg["T_HIGH"]) >= self.cfg["T_LOW"] else np.zeros((n, n), bool)
        return (self.group[:, None] == self.group[None, :]) & full   # dynamic

    def step(self, it):
        """Dynamic only: learners at home visit another group with probability P_VISIT per iteration."""
        groups = self.home.max() + 1
        if self.kind != "dynamic" or groups < 2:
            return
        returning = (self.visit_end >= 0) & (it >= self.visit_end)
        self.group[returning], self.visit_end[returning] = self.home[returning], -1
        leaving = (self.visit_end < 0) & ~returning & (self.rng.random(self.n) < self.cfg["P_VISIT"])
        offsets = self.rng.integers(1, groups, self.n)
        self.group[leaving] = (self.home[leaving] + offsets[leaving]) % groups
        self.visit_end[leaving] = it + self.cfg["VISIT_DURATION"]
