"""SAPIENS: groups of RL agents that share experiences over a social network.

One run trains NUM_AGENTS agents for every seed in a single jitted function (seeds and agents
are vmapped), so it fits on one GPU. ALGO selects the learner: `dqn` shares replay transitions,
`vtrace` shares trajectories between actor-critic agents.

Usage:
    python -m sapiens.train --config configs/merging_paths.yaml --set TOPOLOGY=dynamic \
        --num_seeds 20 --output results/merging_dynamic
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as onp
import yaml

from sapiens import dqn, topology, vtrace
from sapiens.envs import Env

LEARNERS = {"dqn": dqn, "vtrace": vtrace}

DEFAULTS = dict(
    ENV_NAME="Merging-paths-alchemy",
    ENV_KWARGS={},             # passed to gymnax.make, e.g. {size: 20} for DeepSea
    ALGO="dqn",                # dqn | vtrace
    TOPOLOGY="fully",          # independent | fully | ring | small_world | dynamic | periodic | centralized
    NUM_AGENTS=10,
    NUM_ENVS=1,                # parallel environments per agent
    TOTAL_STEPS=2_000_000,     # environment steps per agent, summed over its environments
    NETWORK="mlp",             # mlp | cnn (MinAtar network: one conv layer, then HIDDEN)
    HIDDEN=[64, 64],
    # LR, GAMMA, EPS_FRACTION, TARGET_UPDATE_INTERVAL: tuned without sharing on the single-path task
    LR=1e-3,
    GAMMA=0.9,
    # DQN
    BUFFER_SIZE=25_000,        # per agent, the same for every topology
    BATCH_SIZE=256,
    EPS_START=1.0,
    EPS_END=0.05,
    EPS_FRACTION=0.1,          # fraction of TOTAL_STEPS over which epsilon decays linearly
    LEARNING_STARTS=10_000,
    TRAIN_INTERVAL=4,          # steps per gradient update
    TARGET_UPDATE_INTERVAL=1_000,
    # V-trace actor-critic
    ROLLOUT_LEN=32,
    VF_COEF=0.5,
    ENT_COEF=0.01,
    MAX_GRAD_NORM=0.5,
    CLIP_RHO=1.0,
    CLIP_C=1.0,
    # experience sharing
    SHARE_INTERVAL=0,          # DQN only. 0: share at the end of each episode (as in the paper); k > 0: every k steps
    P_SHARE=1.0,               # probability that a sharing opportunity is used (V-trace: once per update)
    SHARE_BATCH=16,            # sent to each neighbour per sharing event: transitions (DQN) or trajectories (V-trace)
    # topology parameters, tuned per topology on merging paths with held-out seeds (sweep topo_params)
    RING_K=6,
    SW_K=4,
    SW_BETA=0.05,
    GROUP_SIZE=2,
    P_VISIT=0.001,             # per step
    VISIT_DURATION=160,        # steps
    T_LOW=1_600,
    T_HIGH=160,                # periodic: connected phase; adaptive_discovery: broadcast window
    ADAPT_INTERVAL=1_000,      # adaptive topologies: steps between decisions
    D_LOW=0.15,                # adaptive_diversity: isolate once trajectory diversity is at most this
    D_HIGH=0.55,               # adaptive_diversity: connect once it is above this
    DISCOVERY_MARGIN=0.0,      # adaptive_discovery: improvement over the group's best greedy return that counts
    # logging and evaluation
    LOG_INTERVAL=10_000,
    EVAL_EPISODES=1,
    EVAL_STEPS=None,           # defaults to the environment's episode length
    EVAL_GREEDY=True,          # V-trace: argmax instead of sampling from the policy
    SUCCESS_THRESHOLD=None,    # defaults to the task optimum for alchemy tasks
    SUCCESS_METRIC="return",   # return: eval return >= SUCCESS_THRESHOLD; goal: episode ended before EVAL_STEPS
)


def run(cfg, num_seeds, seed_offset=0):
    env = Env(cfg["ENV_NAME"], cfg["ENV_KWARGS"])
    seeds = onp.arange(seed_offset, seed_offset + num_seeds)
    rngs = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])
    # the task instance (e.g. recipe book) differs across seeds but is shared by the agents of a seed
    task_keys = jnp.stack([jax.random.PRNGKey(10_000 + int(s)) for s in seeds])
    adj0 = jnp.stack([topology.initial_adjacency(cfg, int(s)) for s in seeds])
    train = jax.jit(jax.vmap(LEARNERS[cfg["ALGO"]].make_train(cfg, env)))
    metrics, params = train(rngs, task_keys, adj0)
    jax.block_until_ready(metrics)
    return jax.tree.map(onp.asarray, metrics), params, env


def success(cfg, metrics, env):
    """(seeds, logs, agents) boolean success of each greedy evaluation, or None if undefined."""
    if cfg["SUCCESS_METRIC"] == "goal":
        return metrics["eval_length"] < (cfg["EVAL_STEPS"] or env.max_steps)
    if cfg["SUCCESS_THRESHOLD"] is None:
        return None
    return metrics["eval_return"] >= cfg["SUCCESS_THRESHOLD"] - 1e-6


def parse_value(v):
    # YAML reads "1e-3" as a string, so try numbers first
    if v in ("None", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return yaml.safe_load(v)


def load_config(path=None, overrides=()):
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as f:
            cfg.update({k: parse_value(v) if isinstance(v, str) else v for k, v in (yaml.safe_load(f) or {}).items()})
    for item in overrides:
        key, value = item.split("=", 1)
        if "." in key:  # nested override, e.g. ENV_KWARGS.size=24
            key, sub = key.split(".", 1)
            cfg[key] = dict(cfg.get(key) or {}, **{sub: parse_value(value)})
        else:
            cfg[key] = parse_value(value)
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown config keys {sorted(unknown)}")
    for key in ("TOTAL_STEPS", "LOG_INTERVAL", "LEARNING_STARTS", "TARGET_UPDATE_INTERVAL", "BUFFER_SIZE"):
        cfg[key] = int(cfg[key])
    if cfg["ALGO"] not in LEARNERS:
        raise ValueError(f"ALGO must be one of {sorted(LEARNERS)}")
    update_steps = LEARNERS[cfg["ALGO"]].steps_per_update(cfg)
    if cfg["LOG_INTERVAL"] % update_steps:
        raise ValueError(f"LOG_INTERVAL must be a multiple of the steps per update ({update_steps})")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML file with config overrides")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="further overrides")
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--output", required=True, help="directory for config.yaml, metrics.npz, summary.json")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    os.makedirs(args.output, exist_ok=True)
    print(f"devices: {jax.devices()}", flush=True)
    start = time.time()
    metrics, _, env = run(cfg, args.num_seeds, args.seed_offset)
    elapsed = time.time() - start

    if cfg["SUCCESS_THRESHOLD"] is None:
        cfg["SUCCESS_THRESHOLD"] = env.optimal_return
    with open(os.path.join(args.output, "config.yaml"), "w") as f:
        yaml.safe_dump(dict(cfg, num_seeds=args.num_seeds, seed_offset=args.seed_offset,
                            achievement_names=env.achievement_names), f)
    onp.savez_compressed(os.path.join(args.output, "metrics.npz"), **metrics)

    best = metrics["eval_return"].max(-1)  # (seeds, logs): best agent in the group
    summary = dict(elapsed_sec=elapsed, final_best_eval_return=best[:, -1].tolist(),
                   final_mean_eval_return=metrics["eval_return"][:, -1].mean(-1).tolist())
    ok = success(cfg, metrics, env)
    if ok is not None:
        summary["group_success"] = ok.any(-1).any(-1).mean().item()
        summary["final_agents_successful"] = ok[:, -1].mean().item()
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
