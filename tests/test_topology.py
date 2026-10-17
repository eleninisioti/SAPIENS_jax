import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from sapiens import topology
from sapiens.train import DEFAULTS


def cfg(**kw):
    return dict(DEFAULTS, **kw)


@pytest.mark.parametrize("topo", topology.TOPOLOGIES)
@pytest.mark.parametrize("n", [1, 6, 10, 20])
def test_initial_graphs_are_symmetric_without_self_loops(topo, n):
    adj = topology.initial_adjacency(cfg(TOPOLOGY=topo, NUM_AGENTS=n), seed=0)
    assert adj.shape == (n, n)
    assert (adj == adj.T).all() and not adj.diagonal().any()


def test_expected_degrees():
    assert topology.initial_adjacency(cfg(TOPOLOGY="fully", NUM_AGENTS=10), 0).sum(1).tolist() == [9] * 10
    assert topology.initial_adjacency(cfg(TOPOLOGY="ring", NUM_AGENTS=10), 0).sum(1).tolist() == [2] * 10
    assert topology.initial_adjacency(cfg(TOPOLOGY="dynamic", NUM_AGENTS=10), 0).sum(1).tolist() == [1] * 10
    assert topology.initial_adjacency(cfg(TOPOLOGY="independent", NUM_AGENTS=10), 0).sum() == 0


@pytest.mark.parametrize("n", [10, 20])
def test_dynamic_visits_leave_and_return(n):
    c = cfg(TOPOLOGY="dynamic", NUM_AGENTS=n, P_VISIT=0.05, VISIT_DURATION=7)
    state = topology.init_state(c, topology.initial_adjacency(c, 0))
    step = jax.jit(lambda s, k, t: topology.step(s, k, t, c))
    key = jax.random.PRNGKey(0)
    visited, away_since = False, onp.full(n, -1)
    for t in range(400):
        key, k = jax.random.split(key)
        state = step(state, k, t)
        group, home = onp.asarray(state.group), onp.asarray(state.home)
        away = group != home
        visited |= away.any()
        # adjacency always equals "same current group"
        expected = (group[:, None] == group[None, :]) & ~onp.eye(n, dtype=bool)
        onp.testing.assert_array_equal(onp.asarray(state.adj), expected)
        away_since = onp.where(away & (away_since < 0), t, onp.where(away, away_since, -1))
        assert ((t - away_since)[away] <= 7).all(), "visit lasted longer than VISIT_DURATION"
    assert visited


def test_periodic_alternates():
    c = cfg(TOPOLOGY="periodic", NUM_AGENTS=4, T_LOW=3, T_HIGH=2)
    state = topology.init_state(c, topology.initial_adjacency(c, 0))
    degrees = [int(topology.step(state, jax.random.PRNGKey(0), t, c).adj.sum()) for t in range(10)]
    assert degrees == [0, 0, 0, 12, 12] * 2
