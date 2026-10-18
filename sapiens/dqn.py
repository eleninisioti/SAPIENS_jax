"""DQN agents that share transitions from their replay buffers over the social network.

Time is a nest of lax.scan loops

    log iteration (LOG_INTERVAL steps)  ->  learner update  ->  vector env step (NUM_ENVS steps)

so the step index is never a batched value and the learning / target-update conditions stay
real branches instead of being evaluated on every step. All step counts are per agent.
"""
import jax
import jax.numpy as jnp
import optax

from sapiens import buffer as rb
from sapiens import topology
from sapiens.common import (MLP, conv_shape, count_unique, disagreement, hash_transitions, hash_weights, init_params,
                            make_evaluate, pooled_grads)


def steps_per_update(cfg):
    return max(cfg["TRAIN_INTERVAL"], cfg["NUM_ENVS"])


def make_train(cfg, env, return_buffers=False):
    n, n_envs = cfg["NUM_AGENTS"], cfg["NUM_ENVS"]
    iters_per_update = max(1, cfg["TRAIN_INTERVAL"] // n_envs)
    grad_steps = max(1, n_envs // cfg["TRAIN_INTERVAL"])
    update_steps = steps_per_update(cfg)
    n_logs = cfg["TOTAL_STEPS"] // cfg["LOG_INTERVAL"]
    updates_per_log = cfg["LOG_INTERVAL"] // update_steps
    share_batch = cfg["SHARE_BATCH"]
    centralized = cfg["TOPOLOGY"] == "centralized"
    sharing = cfg["TOPOLOGY"] not in ("independent", "centralized") and n > 1
    network = MLP(env.num_actions, tuple(cfg["HIDDEN"]), conv_shape(cfg, env))
    tx = optax.adam(cfg["LR"])
    evaluate = make_evaluate(env, lambda p, obs, k: network.apply(p, obs).argmax(),
                             cfg["EVAL_STEPS"] or env.max_steps, cfg["EVAL_EPISODES"])

    def epsilon(t):
        frac = jnp.clip(t / (cfg["EPS_FRACTION"] * cfg["TOTAL_STEPS"]), 0.0, 1.0)
        return cfg["EPS_START"] + frac * (cfg["EPS_END"] - cfg["EPS_START"])

    def group_disagreement(params, obs):
        probe = obs.reshape(-1, env.obs_dim)  # current observations of all agents
        values = jax.vmap(lambda p: network.apply(p, probe))(params)
        return disagreement(values)

    def adapt(s, t):
        returns = jnp.where(s["adapt_count"] > 0, s["adapt_sum"] / jnp.maximum(s["adapt_count"], 1), -jnp.inf)
        topo = topology.adapt(s["topo"], t, cfg, group_disagreement(s["params"], s["obs"]), returns)
        return dict(s, topo=topo, adapt_sum=jnp.zeros((n,)), adapt_count=jnp.zeros((n,)))

    def maybe_adapt(s, t):
        if cfg["TOPOLOGY"] not in topology.ADAPTIVE:
            return s
        return jax.lax.cond(t % cfg["ADAPT_INTERVAL"] < update_steps, lambda s: adapt(s, t), lambda s: s, s)

    vreset = jax.vmap(jax.vmap(env.reset, in_axes=(0, None)), in_axes=(0, None))
    vstep = jax.vmap(jax.vmap(env.step, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, None))

    def train(rng, task_key, adj0):
        env_params = env.params(task_key)
        rng, k_net, k_env, k_hash = jax.random.split(rng, 4)
        params = init_params(network, k_net, n, env.obs_dim, shared=centralized)
        obs, env_state = vreset(jax.random.split(k_env, n * n_envs).reshape(n, n_envs, -1), env_params)
        buf = jax.vmap(lambda _: rb.init(cfg["BUFFER_SIZE"], env.obs_dim))(jnp.arange(n))
        w_obs, w_act = hash_weights(k_hash, env.obs_dim)

        zeros = jnp.zeros((n,), jnp.float32)
        state = dict(
            params=params, target=params, opt=jax.vmap(tx.init)(params), buf=buf,
            env_state=env_state, obs=obs, topo=topology.init_state(cfg, adj0),
            ep_return=jnp.zeros((n, n_envs)), ret_sum=zeros, ret_count=zeros, loss_sum=zeros,
            loss_count=zeros, received=zeros, adapt_sum=zeros, adapt_count=zeros, rng=rng,
        )

        def env_step(s, t):
            rng, k_act, k_rand, k_env, k_share, k_sample, k_topo = jax.random.split(s["rng"], 7)
            q = jax.vmap(jax.vmap(network.apply, in_axes=(None, 0)))(s["params"], s["obs"])
            explore = jax.random.uniform(k_act, (n, n_envs)) < epsilon(t)
            action = jnp.where(explore, jax.random.randint(k_rand, (n, n_envs), 0, env.num_actions), q.argmax(-1))
            next_obs, env_state, reward, done = vstep(
                jax.random.split(k_env, n * n_envs).reshape(n, n_envs, -1), s["env_state"], action, env_params)

            own = dict(obs=s["obs"], action=action, reward=reward, next_obs=next_obs, done=done)
            buf = jax.vmap(rb.add)(s["buf"], own, jnp.ones((n, n_envs), bool))

            received = s["received"]
            if sharing:
                if cfg["SHARE_INTERVAL"] == 0:
                    trigger = done.any(-1)
                else:
                    trigger = jnp.full((n,), t % cfg["SHARE_INTERVAL"] < n_envs)
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
                ret_sum=s["ret_sum"] + jnp.where(done, ep_return, 0.0).sum(-1),
                ret_count=s["ret_count"] + done.sum(-1),
                adapt_sum=s["adapt_sum"] + jnp.where(done, ep_return, 0.0).sum(-1),
                adapt_count=s["adapt_count"] + done.sum(-1),
                ep_return=jnp.where(done, 0.0, ep_return),
                topo=topology.step(s["topo"], k_topo, t + n_envs, cfg, dt=n_envs),
            )
            return s, None

        def dqn_loss(params, target, batch):
            q_next = network.apply(target, batch["next_obs"]).max(-1)
            y = batch["reward"] + (1.0 - batch["done"]) * cfg["GAMMA"] * q_next
            q = jnp.take_along_axis(network.apply(params, batch["obs"]), batch["action"][:, None], -1)[:, 0]
            return jnp.mean((q - jax.lax.stop_gradient(y)) ** 2)

        def learn(s, _):
            rng, k = jax.random.split(s["rng"])
            batch = jax.vmap(rb.sample, in_axes=(0, 0, None))(s["buf"], jax.random.split(k, n), cfg["BATCH_SIZE"])
            if centralized:
                # one learner trained on a batch pooled from all actors' buffers (Ape-X style)
                target0 = jax.tree.map(lambda x: x[0], s["target"])
                loss, grads = pooled_grads(lambda p, b: dqn_loss(p, target0, b), s["params"], batch)
            else:
                loss, grads = jax.vmap(jax.value_and_grad(dqn_loss))(s["params"], s["target"], batch)
            updates, opt = jax.vmap(tx.update)(grads, s["opt"], s["params"])
            return dict(s, params=optax.apply_updates(s["params"], updates), opt=opt, rng=rng,
                        loss_sum=s["loss_sum"] + loss, loss_count=s["loss_count"] + 1), None

        def update_step(s, t0):
            s, _ = jax.lax.scan(env_step, s, t0 + jnp.arange(iters_per_update) * n_envs)
            t = t0 + update_steps
            s = jax.lax.cond(t >= cfg["LEARNING_STARTS"], lambda s: jax.lax.scan(learn, s, None, grad_steps)[0],
                             lambda s: s, s)
            s = jax.lax.cond(t % cfg["TARGET_UPDATE_INTERVAL"] < update_steps,
                             lambda s: dict(s, target=s["params"]), lambda s: s, s)
            return maybe_adapt(s, t), None

        def log_step(s, t0):
            s, _ = jax.lax.scan(update_step, s, t0 + jnp.arange(updates_per_log) * update_steps)
            rng, k_eval = jax.random.split(s["rng"])
            eval_return, traj_hash, achievements, eval_length = evaluate(s["params"], k_eval, env_params)
            buf = s["buf"]
            hashes = jax.vmap(hash_transitions, in_axes=(0, 0, None, None))(buf.obs, buf.action, w_obs, w_act)
            valid = jax.vmap(rb.valid_mask)(buf)
            metrics = dict(
                step=t0 + cfg["LOG_INTERVAL"],
                train_return=s["ret_sum"] / jnp.maximum(s["ret_count"], 1),
                episodes=s["ret_count"],
                loss=s["loss_sum"] / jnp.maximum(s["loss_count"], 1),
                eval_return=eval_return,
                eval_length=eval_length,
                traj_hash=traj_hash,
                achievements=achievements,
                diversity=jax.vmap(count_unique)(hashes, valid),
                group_diversity=count_unique(hashes.reshape(-1), valid.reshape(-1)),
                received=s["received"],
                disagreement=group_disagreement(s["params"], s["obs"]),
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
