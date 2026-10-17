"""Social network structures for experience sharing.

The graph is an N x N boolean adjacency matrix: adj[i, j] means agent i shares with agent j.
Static structures (independent, fully, ring, small_world) never change. Two dynamic ones do:

* dynamic:  agents live in groups of GROUP_SIZE. Every step, each agent that is at home starts
            a visit to a random other group with probability P_VISIT and returns after
            VISIT_DURATION steps. Agents share only with members of their current group.
* periodic: the group alternates between no sharing for T_LOW steps and full connectivity
            for T_HIGH steps.
"""
import chex
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as onp

TOPOLOGIES = ("independent", "fully", "ring", "small_world", "dynamic", "periodic")


@chex.dataclass(frozen=True)
class TopoState:
    adj: chex.Array        # (N, N) bool
    group: chex.Array      # (N,) current group (dynamic only)
    home: chex.Array       # (N,) home group (dynamic only)
    visit_end: chex.Array  # (N,) step at which the visit ends, -1 when at home


def num_groups(cfg) -> int:
    return -(-cfg["NUM_AGENTS"] // cfg["GROUP_SIZE"])


def _from_groups(group):
    n = group.shape[0]
    return (group[:, None] == group[None, :]) & ~jnp.eye(n, dtype=bool)


def initial_adjacency(cfg, seed: int) -> onp.ndarray:
    """Builds the starting graph on the host. `seed` only matters for small_world."""
    n, topo = cfg["NUM_AGENTS"], cfg["TOPOLOGY"]
    if topo not in TOPOLOGIES:
        raise ValueError(f"Unknown topology {topo}, expected one of {TOPOLOGIES}")
    if topo in ("independent", "periodic") or n == 1:
        # periodic starts in its low-connectivity phase
        return onp.zeros((n, n), bool)
    if topo == "fully":
        return ~onp.eye(n, dtype=bool)
    if topo == "ring":
        adj = onp.zeros((n, n), bool)
        for k in range(1, cfg["RING_K"] // 2 + 1):
            for i in range(n):
                adj[i, (i + k) % n] = adj[(i + k) % n, i] = True
        onp.fill_diagonal(adj, False)
        return adj
    if topo == "small_world":
        g = nx.watts_strogatz_graph(n, min(cfg["SW_K"], n - 1), cfg["SW_BETA"], seed=seed)
        return nx.to_numpy_array(g, dtype=bool)
    # dynamic
    home = onp.arange(n) // cfg["GROUP_SIZE"]
    return (home[:, None] == home[None, :]) & ~onp.eye(n, dtype=bool)


def init_state(cfg, adj0) -> TopoState:
    n = cfg["NUM_AGENTS"]
    home = jnp.arange(n) // cfg["GROUP_SIZE"]
    return TopoState(adj=jnp.asarray(adj0, bool), group=home, home=home,
                     visit_end=-jnp.ones((n,), jnp.int32))


def step(state: TopoState, key: chex.PRNGKey, t, cfg) -> TopoState:
    """Advances the graph by one environment step (t = global step index)."""
    topo, n = cfg["TOPOLOGY"], cfg["NUM_AGENTS"]
    if topo == "periodic":
        period = cfg["T_LOW"] + cfg["T_HIGH"]
        high = (t % period) >= cfg["T_LOW"]
        return state.replace(adj=jnp.where(high, ~jnp.eye(n, dtype=bool), False))
    if topo != "dynamic" or num_groups(cfg) < 2:
        return state

    g = num_groups(cfg)
    k_start, k_target = jax.random.split(key)
    at_home = state.visit_end < 0
    returning = ~at_home & (t >= state.visit_end)
    leaving = at_home & (jax.random.uniform(k_start, (n,)) < cfg["P_VISIT"])
    # offset in [1, g) guarantees the visited group differs from the home group
    target = (state.home + jax.random.randint(k_target, (n,), 1, g)) % g

    group = jnp.where(leaving, target, jnp.where(returning, state.home, state.group))
    visit_end = jnp.where(leaving, t + cfg["VISIT_DURATION"], jnp.where(returning, -1, state.visit_end))
    return state.replace(adj=_from_groups(group), group=group, visit_end=visit_end.astype(jnp.int32))
