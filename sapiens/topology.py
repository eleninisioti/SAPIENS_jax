"""Social network structures for experience sharing.

The graph is an N x N boolean adjacency matrix: adj[i, j] means agent i shares with agent j.
Static structures (independent, fully, ring, small_world) never change. Two dynamic ones do:

* dynamic:  agents live in groups of GROUP_SIZE. Every step, each agent that is at home starts
            a visit to a random other group with probability P_VISIT and returns after
            VISIT_DURATION steps. Agents share only with members of their current group.
* periodic: the group alternates between no sharing for T_LOW steps and full connectivity
            for T_HIGH steps.

Two adaptive structures decide when to connect from what the group does, instead of a schedule:

Both roll out every agent's greedy policy every ADAPT_INTERVAL steps:

* adaptive_diversity: a thermostat on behavioural diversity, the fraction of agents whose greedy
            trajectory differs from the majority trajectory. Once diversity falls to D_LOW or below
            the group has converged and isolates; once it rises above D_HIGH it reconnects fully.
* adaptive_discovery: agents are isolated until one of them beats the best greedy return the
            group has reached (by more than DISCOVERY_MARGIN); the group then connects fully for
            T_HIGH steps so the discovery spreads.

`centralized` is the standard distributed-RL baseline: no experience sharing between learners
because there is a single learner (one network, trained on the pooled data of all actors).
"""
import chex
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as onp

ADAPTIVE = ("adaptive_diversity", "adaptive_discovery")
TOPOLOGIES = ("independent", "fully", "ring", "small_world", "dynamic", "periodic", "centralized") + ADAPTIVE


@chex.dataclass(frozen=True)
class TopoState:
    adj: chex.Array        # (N, N) bool
    group: chex.Array      # (N,) current group (dynamic only)
    home: chex.Array       # (N,) home group (dynamic only)
    visit_end: chex.Array  # (N,) step at which the visit ends, -1 when at home
    connected: chex.Array  # () adaptive_diversity: currently fully connected
    high_until: chex.Array # () adaptive_discovery: fully connected until this step
    best: chex.Array       # () adaptive_discovery: best training return seen by the group


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
    if topo in ("independent", "periodic", "centralized") + ADAPTIVE or n == 1:
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
                     visit_end=-jnp.ones((n,), jnp.int32), connected=jnp.array(False),
                     high_until=jnp.array(-1, jnp.int32), best=jnp.array(-jnp.inf))


def step(state: TopoState, key: chex.PRNGKey, t, cfg, dt=1) -> TopoState:
    """Advances the graph to step t (steps per agent), `dt` steps after the previous call."""
    topo, n = cfg["TOPOLOGY"], cfg["NUM_AGENTS"]
    if topo == "periodic":
        period = cfg["T_LOW"] + cfg["T_HIGH"]
        high = (t % period) >= cfg["T_LOW"]
        return state.replace(adj=jnp.where(high, ~jnp.eye(n, dtype=bool), False))
    full = ~jnp.eye(n, dtype=bool)
    if topo == "adaptive_diversity":
        return state.replace(adj=jnp.where(state.connected, full, False))
    if topo == "adaptive_discovery":
        return state.replace(adj=jnp.where(t < state.high_until, full, False))
    if topo != "dynamic" or num_groups(cfg) < 2:
        return state

    g = num_groups(cfg)
    k_start, k_target = jax.random.split(key)
    at_home = state.visit_end < 0
    returning = ~at_home & (t >= state.visit_end)
    p_leave = 1.0 - (1.0 - cfg["P_VISIT"]) ** dt  # P_VISIT is per step
    leaving = at_home & (jax.random.uniform(k_start, (n,)) < p_leave)
    # offset in [1, g) guarantees the visited group differs from the home group
    target = (state.home + jax.random.randint(k_target, (n,), 1, g)) % g

    group = jnp.where(leaving, target, jnp.where(returning, state.home, state.group))
    visit_end = jnp.where(leaving, t + cfg["VISIT_DURATION"], jnp.where(returning, -1, state.visit_end))
    return state.replace(adj=_from_groups(group), group=group, visit_end=visit_end.astype(jnp.int32))


def adapt(state: TopoState, t, cfg, diversity, returns) -> TopoState:
    """Updates an adaptive structure every ADAPT_INTERVAL steps.

    diversity: fraction of agents not following the group's majority greedy trajectory;
    returns: (N,) greedy return of each agent.
    """
    if cfg["TOPOLOGY"] == "adaptive_diversity":
        connected = jnp.where(state.connected, diversity > cfg["D_LOW"], diversity > cfg["D_HIGH"])
        return state.replace(connected=connected)
    if cfg["TOPOLOGY"] == "adaptive_discovery":
        best_now = returns.max()
        discovered = jnp.isfinite(state.best) & (best_now > state.best + cfg["DISCOVERY_MARGIN"])
        return state.replace(best=jnp.maximum(state.best, best_now),
                             high_until=jnp.where(discovered, t + cfg["T_HIGH"], state.high_until).astype(jnp.int32))
    return state
