"""SAPIENS: a group of DQN agents that share experiences over a social network.

Everything for one run (all agents, all seeds) is a single jitted function: seeds are vmapped,
agents are vmapped inside each seed, and time is a nest of lax.scan loops

    log iteration (LOG_INTERVAL steps)  ->  learner update (TRAIN_INTERVAL steps)  ->  env step

so the step index is never a batched value and the learning / target-update conditions stay
real branches instead of being evaluated on every step.

Usage:
    python -m sapiens.train --config configs/merging_paths.yaml --set TOPOLOGY=dynamic \
        --num_seeds 20 --output results/merging_dynamic
"""
import argparse
import json
import os
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import yaml

from sapiens import buffer as rb
from sapiens import topology
from sapiens.envs import Env

DEFAULTS = dict(
    ENV_NAME="Merging-paths-alchemy",
    TOPOLOGY="fully",          # independent | fully | ring | small_world | dynamic | periodic
    NUM_AGENTS=10,
    TOTAL_STEPS=2_000_000,     # environment steps per agent
    # DQN
    BUFFER_SIZE=25_000,        # per agent, the same for every topology
    BATCH_SIZE=256,
    LR=1e-4,
    GAMMA=0.99,
    HIDDEN=[64, 64],
    EPS_START=1.0,
    EPS_END=0.05,
    EPS_FRACTION=0.2,          # fraction of TOTAL_STEPS over which epsilon decays linearly
    LEARNING_STARTS=10_000,
    TRAIN_INTERVAL=4,
    TARGET_UPDATE_INTERVAL=10_000,
    # experience sharing
    SHARE_INTERVAL=0,          # 0: share at the end of each episode (as in the paper); k > 0: every k steps
    P_SHARE=1.0,               # probability that a sharing opportunity is used
    SHARE_BATCH=16,            # L_s, experiences sent to each neighbour per sharing event
    # topology parameters
    RING_K=2,
    SW_K=4,
    SW_BETA=0.2,
    GROUP_SIZE=2,
    P_VISIT=0.01,              # per step
    VISIT_DURATION=360,        # steps
    T_LOW=1_600,
    T_HIGH=160,
    # logging and evaluation
    LOG_INTERVAL=10_000,
    EVAL_EPISODES=1,
    EVAL_STEPS=None,           # defaults to the environment's episode length
    SUCCESS_THRESHOLD=None,    # defaults to the task optimum for alchemy tasks
)


class QNetwork(nn.Module):
    action_dim: int
    hidden: tuple

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(self.action_dim)(x)


def _hash_transitions(obs, action, w_obs, w_act):
    """32-bit hash of (s, a); collisions are negligible at buffer scale (~1e-5 per pair)."""
    q = jnp.round(obs * 16).astype(jnp.int32).astype(jnp.uint32)
    return (q * w_obs).sum(-1, dtype=jnp.uint32) + (action.astype(jnp.uint32) + 1) * w_act


def _count_unique(hashes, valid):
    sentinel = jnp.uint32(0xFFFFFFFF)
    s = jnp.sort(jnp.where(valid, hashes, sentinel))
    first = jnp.concatenate([jnp.array([True]), s[1:] != s[:-1]])
    return jnp.sum(first & (s != sentinel))


def make_train(cfg, env: Env, return_buffers=False):
    n = cfg["NUM_AGENTS"]
    n_logs = cfg["TOTAL_STEPS"] // cfg["LOG_INTERVAL"]
    updates_per_log = cfg["LOG_INTERVAL"] // cfg["TRAIN_INTERVAL"]
    eval_steps = cfg["EVAL_STEPS"] or env.max_steps
    share_batch = cfg["SHARE_BATCH"]
    sharing = cfg["TOPOLOGY"] != "independent" and n > 1
    network = QNetwork(env.num_actions, tuple(cfg["HIDDEN"]))
    tx = optax.adam(cfg["LR"])

    def epsilon(t):
        frac = jnp.clip(t / (cfg["EPS_FRACTION"] * cfg["TOTAL_STEPS"]), 0.0, 1.0)
        return cfg["EPS_START"] + frac * (cfg["EPS_END"] - cfg["EPS_START"])

    def train(rng, task_key, adj0):
        env_params = env.params(task_key)
        rng, k_net, k_env, k_hash = jax.random.split(rng, 4)

        params = jax.vmap(network.init, in_axes=(0, None))(
            jax.random.split(k_net, n), jnp.zeros((env.obs_dim,)))
        opt_state = jax.vmap(tx.init)(params)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_env, n), env_params)
        buf = jax.vmap(lambda _: rb.init(cfg["BUFFER_SIZE"], env.obs_dim))(jnp.arange(n))
        k1, k2 = jax.random.split(k_hash)
        w_obs = jax.random.bits(k1, (env.obs_dim,), jnp.uint32) | 1
        w_act = jax.random.bits(k2, (), jnp.uint32) | 1

        zeros = jnp.zeros((n,), jnp.float32)
        state = dict(
            params=params, target=params, opt=opt_state, buf=buf,
            env_state=env_state, obs=obs, topo=topology.init_state(cfg, adj0),
            ep_return=zeros, ret_sum=zeros, ret_count=zeros, loss_sum=zeros, loss_count=zeros,
            received=zeros, prev_hash=jnp.zeros((n,), jnp.uint32), rng=rng,
        )

        def env_step(s, t):
            rng, k_act, k_rand, k_env, k_share, k_sample, k_topo = jax.random.split(s["rng"], 7)
            q = jax.vmap(network.apply)(s["params"], s["obs"])
            explore = jax.random.uniform(k_act, (n,)) < epsilon(t)
            action = jnp.where(explore, jax.random.randint(k_rand, (n,), 0, env.num_actions), q.argmax(-1))
            next_obs, env_state, reward, done = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(k_env, n), s["env_state"], action, env_params)

            own = dict(obs=s["obs"], action=action, reward=reward, next_obs=next_obs, done=done)
            buf = jax.vmap(rb.add)(s["buf"], jax.tree.map(lambda x: x[:, None], own), jnp.ones((n, 1), bool))

            received = s["received"]
            if sharing:
                trigger = done if cfg["SHARE_INTERVAL"] == 0 else jnp.full((n,), t % cfg["SHARE_INTERVAL"] == 0)
                active = trigger & (jax.random.uniform(k_share, (n,)) < cfg["P_SHARE"]) & (buf.size >= share_batch)
                send = s["topo"].adj & active[:, None]                       # (sender, receiver)
                # each sender draws a distinct batch for each receiver: (sender, receiver, L, ...)
                keys = jax.random.split(k_sample, n * n).reshape(n, n, -1)
                batches = jax.vmap(lambda b, ks: jax.vmap(lambda k: rb.sample(b, k, share_batch))(ks))(buf, keys)
                incoming = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1).reshape(n, n * share_batch, *x.shape[3:]), batches)
                in_mask = jnp.repeat(send.T, share_batch, axis=1)             # (receiver, sender * L)
                buf = jax.vmap(rb.add)(buf, incoming, in_mask)
                received = received + in_mask.sum(1)

            ep_return = s["ep_return"] + reward
            s = dict(
                s, buf=buf, env_state=env_state, obs=next_obs, rng=rng, received=received,
                ret_sum=s["ret_sum"] + jnp.where(done, ep_return, 0.0),
                ret_count=s["ret_count"] + done,
                ep_return=jnp.where(done, 0.0, ep_return),
                topo=topology.step(s["topo"], k_topo, t, cfg),
            )
            return s, None

        def dqn_loss(params, target, batch):
            q_next = network.apply(target, batch["next_obs"]).max(-1)
            y = batch["reward"] + (1.0 - batch["done"]) * cfg["GAMMA"] * q_next
            q = jnp.take_along_axis(network.apply(params, batch["obs"]), batch["action"][:, None], -1)[:, 0]
            return jnp.mean((q - jax.lax.stop_gradient(y)) ** 2)

        def learn(s):
            rng, k = jax.random.split(s["rng"])
            batch = jax.vmap(rb.sample, in_axes=(0, 0, None))(s["buf"], jax.random.split(k, n), cfg["BATCH_SIZE"])
            loss, grads = jax.vmap(jax.value_and_grad(dqn_loss))(s["params"], s["target"], batch)
            updates, opt = jax.vmap(tx.update)(grads, s["opt"], s["params"])
            return dict(s, params=optax.apply_updates(s["params"], updates), opt=opt, rng=rng,
                        loss_sum=s["loss_sum"] + loss, loss_count=s["loss_count"] + 1)

        def update_step(s, t0):
            s, _ = jax.lax.scan(env_step, s, t0 + jnp.arange(cfg["TRAIN_INTERVAL"]))
            t = t0 + cfg["TRAIN_INTERVAL"]
            s = jax.lax.cond(t >= cfg["LEARNING_STARTS"], learn, lambda s: s, s)
            s = jax.lax.cond(t % cfg["TARGET_UPDATE_INTERVAL"] < cfg["TRAIN_INTERVAL"],
                             lambda s: dict(s, target=s["params"]), lambda s: s, s)
            return s, None

        def evaluate(params, key):
            """Greedy rollout of every agent from the same start state; returns (return, trajectory hash)."""
            def rollout(p, key):
                obs, es = env.reset(key, env_params)

                def body(c, _):
                    obs, es, alive, ret, h, key = c
                    key, k = jax.random.split(key)
                    a = network.apply(p, obs).argmax()
                    obs, es, r, d = env.step(k, es, a, env_params)
                    h = jnp.where(alive, h * jnp.uint32(1_000_003) + a.astype(jnp.uint32) + 1, h)
                    return (obs, es, alive & ~d, ret + r * alive, h, key), None

                init = (obs, es, jnp.array(True), jnp.array(0.0), jnp.uint32(0), key)
                (_, _, _, ret, h, _), _ = jax.lax.scan(body, init, None, eval_steps)
                return ret, h

            keys = jax.random.split(key, cfg["EVAL_EPISODES"])
            ret, h = jax.vmap(lambda k: jax.vmap(rollout, in_axes=(0, None))(params, k))(keys)
            return ret.mean(0), h[0]

        def log_step(s, t0):
            s, _ = jax.lax.scan(update_step, s, t0 + jnp.arange(updates_per_log) * cfg["TRAIN_INTERVAL"])
            rng, k_eval = jax.random.split(s["rng"])
            eval_return, traj_hash = evaluate(s["params"], k_eval)
            buf = s["buf"]
            hashes = jax.vmap(_hash_transitions, in_axes=(0, 0, None, None))(buf.obs, buf.action, w_obs, w_act)
            valid = jax.vmap(rb.valid_mask)(buf)
            metrics = dict(
                step=t0 + cfg["LOG_INTERVAL"],
                train_return=s["ret_sum"] / jnp.maximum(s["ret_count"], 1),
                episodes=s["ret_count"],
                loss=s["loss_sum"] / jnp.maximum(s["loss_count"], 1),
                eval_return=eval_return,
                traj_hash=traj_hash,
                diversity=jax.vmap(_count_unique)(hashes, valid),
                group_diversity=_count_unique(hashes.reshape(-1), valid.reshape(-1)),
                received=s["received"],
                degree=s["topo"].adj.sum(1),
                group=s["topo"].group,
            )
            s = dict(s, rng=rng, ret_sum=zeros, ret_count=zeros, loss_sum=zeros, loss_count=zeros, received=zeros)
            return s, metrics

        state, metrics = jax.lax.scan(log_step, state, jnp.arange(n_logs) * cfg["LOG_INTERVAL"])
        if return_buffers:
            return metrics, state["params"], state["buf"]
        return metrics, state["params"]

    return train


def run(cfg, num_seeds, seed_offset=0):
    env = Env(cfg["ENV_NAME"])
    seeds = onp.arange(seed_offset, seed_offset + num_seeds)
    rngs = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])
    # the task instance (e.g. recipe book) differs across seeds but is shared by the agents of a seed
    task_keys = jnp.stack([jax.random.PRNGKey(10_000 + int(s)) for s in seeds])
    adj0 = jnp.stack([topology.initial_adjacency(cfg, int(s)) for s in seeds])
    train = jax.jit(jax.vmap(make_train(cfg, env)))
    metrics, params = train(rngs, task_keys, adj0)
    jax.block_until_ready(metrics)
    return jax.tree.map(onp.asarray, metrics), params, env


def parse_value(v):
    # YAML reads "1e-3" as a string, so try numbers first
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
        if key not in DEFAULTS:
            raise KeyError(f"Unknown config key {key}")
        cfg[key] = parse_value(value)
    for key in ("TOTAL_STEPS", "LOG_INTERVAL", "LEARNING_STARTS", "TARGET_UPDATE_INTERVAL", "BUFFER_SIZE"):
        cfg[key] = int(cfg[key])
    if cfg["LOG_INTERVAL"] % cfg["TRAIN_INTERVAL"]:
        raise ValueError("LOG_INTERVAL must be a multiple of TRAIN_INTERVAL")
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
        yaml.safe_dump(dict(cfg, num_seeds=args.num_seeds, seed_offset=args.seed_offset), f)
    onp.savez_compressed(os.path.join(args.output, "metrics.npz"), **metrics)

    best = metrics["eval_return"].max(-1)  # (seeds, logs): best agent in the group
    summary = dict(elapsed_sec=elapsed, final_best_eval_return=best[:, -1].tolist(),
                   final_mean_eval_return=metrics["eval_return"][:, -1].mean(-1).tolist())
    if cfg["SUCCESS_THRESHOLD"] is not None:
        summary["group_success"] = (best >= cfg["SUCCESS_THRESHOLD"] - 1e-6).any(-1).mean().item()
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
