import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from sapiens.envs import Env
from sapiens.train import load_config, make_train
from sapiens import topology

SMALL = ["TOTAL_STEPS=4000", "LOG_INTERVAL=1000", "LEARNING_STARTS=500", "BUFFER_SIZE=3000",
         "BATCH_SIZE=32", "NUM_AGENTS=6", "TARGET_UPDATE_INTERVAL=500"]


def _run(overrides, seeds=2):
    cfg = load_config(None, SMALL + overrides)
    env = Env(cfg["ENV_NAME"])
    train = jax.jit(jax.vmap(make_train(cfg, env, return_buffers=True)))
    rngs = jnp.stack([jax.random.PRNGKey(s) for s in range(seeds)])
    tasks = jnp.stack([jax.random.PRNGKey(100 + s) for s in range(seeds)])
    adj = jnp.stack([topology.initial_adjacency(cfg, s) for s in range(seeds)])
    metrics, _, buf = train(rngs, tasks, adj)
    return cfg, metrics, buf


@pytest.mark.parametrize("topo", ["fully", "dynamic", "small_world"])
def test_shared_transitions_are_consistent_with_env_dynamics(topo):
    """Alchemy tasks are deterministic, so (s, a) must always lead to the same (r, s', done).

    The previous implementation paired shared experiences with unrelated next states,
    which this test catches.
    """
    cfg, metrics, buf = _run([f"TOPOLOGY={topo}", "ENV_NAME=Merging-paths-alchemy", "P_VISIT=0.05", "VISIT_DURATION=50"])
    assert metrics["received"].sum() > 0, "no experiences were shared"
    for seed in range(buf.obs.shape[0]):
        seen = {}
        for agent in range(buf.obs.shape[1]):
            size = int(buf.size[seed, agent])
            for i in range(size):
                if bool(buf.done[seed, agent, i]):
                    continue  # next_obs is the reset observation after a terminal step
                key = (buf.obs[seed, agent, i].tobytes(), int(buf.action[seed, agent, i]))
                val = (float(buf.reward[seed, agent, i]), buf.next_obs[seed, agent, i].tobytes())
                assert seen.setdefault(key, val) == val


def test_independent_shares_nothing_and_shapes():
    cfg, metrics, _ = _run(["TOPOLOGY=independent"])
    n_logs = cfg["TOTAL_STEPS"] // cfg["LOG_INTERVAL"]
    assert metrics["eval_return"].shape == (2, n_logs, 6)
    assert metrics["received"].sum() == 0
    assert (metrics["diversity"] > 0).all()
    assert (metrics["group_diversity"] >= metrics["diversity"].max(-1)).all()


def test_gymnax_env_runs():
    _, metrics, _ = _run(["ENV_NAME=CartPole-v1", "TOPOLOGY=ring", "SHARE_INTERVAL=16"], seeds=1)
    assert onp.isfinite(metrics["loss"]).all()


def test_single_path_is_learned():
    """Sanity check that the DQN learns the easy task."""
    cfg, metrics, _ = _run(["ENV_NAME=Single-path-alchemy", "TOPOLOGY=fully", "TOTAL_STEPS=60000",
                            "LOG_INTERVAL=10000", "LEARNING_STARTS=2000", "TARGET_UPDATE_INTERVAL=1000",
                            "LR=1e-3", "EPS_FRACTION=0.3", "BUFFER_SIZE=20000"], seeds=2)
    assert metrics["eval_return"][:, -1].max() >= 36.0 - 1e-6
