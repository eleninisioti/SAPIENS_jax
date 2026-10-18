import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from sapiens.envs import Env
from sapiens.train import LEARNERS, load_config
from sapiens import topology

SMALL = ["TOTAL_STEPS=4000", "LOG_INTERVAL=1000", "LEARNING_STARTS=500", "BUFFER_SIZE=3000",
         "BATCH_SIZE=32", "NUM_AGENTS=6", "TARGET_UPDATE_INTERVAL=500"]


def _run(overrides, seeds=2, small=SMALL):
    cfg = load_config(None, small + overrides)
    env = Env(cfg["ENV_NAME"], cfg["ENV_KWARGS"])
    if cfg["ALGO"] == "dqn":
        make = lambda: LEARNERS["dqn"].make_train(cfg, env, return_buffers=True)
    else:
        make = lambda: (lambda *a: (*LEARNERS["vtrace"].make_train(cfg, env)(*a), None))
    train = jax.jit(jax.vmap(make()))
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


def test_dqn_with_parallel_envs_and_centralized_learner():
    cfg, metrics, _ = _run(["TOPOLOGY=centralized", "NUM_ENVS=8", "TRAIN_INTERVAL=4"], seeds=1)
    assert metrics["received"].sum() == 0
    assert onp.isfinite(metrics["loss"]).all()


VTRACE = ["ALGO=vtrace", "NUM_AGENTS=4", "NUM_ENVS=8", "ROLLOUT_LEN=16", "LR=1e-3", "HIDDEN=[64,64]"]


@pytest.mark.parametrize("topo", ["fully", "dynamic", "independent", "centralized"])
def test_vtrace_runs_and_shares(topo):
    _, metrics, _ = _run(VTRACE + [f"TOPOLOGY={topo}", "TOTAL_STEPS=8192", "LOG_INTERVAL=2048",
                                   "SHARE_BATCH=2", "P_VISIT=0.05", "VISIT_DURATION=200"], seeds=2, small=[])
    assert onp.isfinite(metrics["loss"]).all()
    assert (metrics["received"].sum() > 0) == (topo in ("fully", "dynamic"))


def test_vtrace_learns_cartpole():
    _, metrics, _ = _run(VTRACE + ["ENV_NAME=CartPole-v1", "TOPOLOGY=independent", "NUM_AGENTS=2",
                                   "TOTAL_STEPS=150000", "LOG_INTERVAL=15360", "NUM_ENVS=16", "ROLLOUT_LEN=32",
                                   "LR=2.5e-3", "EVAL_GREEDY=True"], seeds=1, small=[])
    assert metrics["eval_return"][0, -3:].max() >= 195


def test_vtrace_targets_match_n_step_returns_on_policy():
    """With rho = 1 and no termination, V-trace targets are the discounted returns bootstrapped at T."""
    from sapiens.vtrace import vtrace
    r = jnp.array([1.0, 0.0, 2.0])
    v = jnp.array([0.5, 0.2, 0.1])
    vs, _ = vtrace(v, jnp.array(3.0), r, jnp.zeros(3), jnp.ones(3), 0.9, 1.0, 1.0)
    expected = [1 + 0.9 * 0 + 0.81 * 2 + 0.729 * 3, 0 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3]
    onp.testing.assert_allclose(vs, expected, rtol=1e-5)


@pytest.mark.parametrize("algo", ["dqn", "vtrace"])
@pytest.mark.parametrize("topo", ["adaptive_diversity", "adaptive_discovery"])
def test_adaptive_topologies_run(algo, topo):
    extra = VTRACE + ["TOTAL_STEPS=8192", "LOG_INTERVAL=2048"] if algo == "vtrace" else []
    _, metrics, _ = _run(extra + [f"TOPOLOGY={topo}", "ADAPT_INTERVAL=500", "T_HIGH=300"], seeds=2,
                         small=[] if algo == "vtrace" else SMALL)
    assert onp.isfinite(metrics["loss"]).all()
    assert ((metrics["disagreement"] >= 0) & (metrics["disagreement"] <= 1)).all()


def test_minatar_cnn_runs():
    _, metrics, _ = _run(["ENV_NAME=Breakout-MinAtar", "NETWORK=cnn", "HIDDEN=[32]", "TOPOLOGY=periodic",
                          "SHARE_INTERVAL=16", "T_LOW=500", "T_HIGH=200"], seeds=1)
    assert onp.isfinite(metrics["loss"]).all()


@pytest.mark.parametrize("env_name,extra", [
    ("Navix-DoorKey-5x5-v0", ["NETWORK=cnn", "HIDDEN=[32]", "SUCCESS_METRIC=goal"]),
    ("Kheperax-standard", ["SUCCESS_METRIC=goal", "EVAL_STEPS=200"]),
])
def test_navix_and_kheperax_run(env_name, extra):
    _, metrics, buf = _run([f"ENV_NAME={env_name}", "TOPOLOGY=fully", "SHARE_INTERVAL=16"] + extra, seeds=1)
    assert onp.isfinite(metrics["loss"]).all()
    assert metrics["received"].sum() > 0
    assert (metrics["eval_length"] > 0).all()


def test_deceptive_coins_paths():
    env = Env("DeceptiveCoins-v0")
    p = env.default_params
    step = jax.jit(lambda s, a: env.step(jax.random.PRNGKey(0), s, a, p))

    def play(actions):
        _, s = env.reset(jax.random.PRNGKey(0), p)
        total, done = 0.0, False
        for a in actions:
            _, s, r, d = step(s, a)
            total += float(r)
            done = bool(d)
            if done:
                break
        return total, done

    up, down, left, right = 0, 1, 2, 3
    assert play([left] * 6) == (2.0, True)                                 # the deceptive path
    optimal = [right] * 3 + [up, up, right, right] + [down] * 4 + [up, up, right]
    assert play(optimal) == (5.0, True)                                     # the better path
    # the barrier: after committing right the agent cannot reach the left diamond
    assert play([right, left, left, left]) == (0.0, False)
