"""Pieces shared by the learners: networks, greedy evaluation and diversity metrics."""
import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    """Dense layers on flat observations; with obs_shape set, a conv layer first.

    The conv torso is the MinAtar paper's network (Young & Tian, 2019): 16 3x3 filters, then dense.
    """
    out: int
    hidden: tuple
    obs_shape: tuple = None

    @nn.compact
    def __call__(self, x):
        if self.obs_shape is not None:
            x = x.reshape(*x.shape[:-1], *self.obs_shape)
            x = nn.relu(nn.Conv(16, (3, 3))(x))
            x = x.reshape(*x.shape[:-3], -1)
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(self.out)(x)


class ActorCritic(nn.Module):
    num_actions: int
    hidden: tuple
    obs_shape: tuple = None

    @nn.compact
    def __call__(self, x):
        logits = MLP(self.num_actions, self.hidden, self.obs_shape)(x)
        value = MLP(1, self.hidden, self.obs_shape)(x)[..., 0]
        return logits, value


def conv_shape(cfg, env):
    return tuple(env.obs_shape) if cfg["NETWORK"] == "cnn" else None


def disagreement(values, tolerance=0.01):
    """values: (agents, states, actions) Q-values or logits on shared probe states.

    Fraction of (agent, state) pairs where the group's majority greedy action is not (near-)greedy
    for the agent, i.e. worse than its best action by more than `tolerance` of its value range.
    Equally valued actions are therefore not counted as disagreement (0 = full conformity).
    """
    greedy = values.argmax(-1)
    majority = jax.nn.one_hot(greedy, values.shape[-1]).sum(0).argmax(-1)       # (states,)
    v_majority = jnp.take_along_axis(values, majority[None, :, None], -1)[..., 0]
    v_max, v_min = values.max(-1), values.min(-1)
    return (v_majority < v_max - tolerance * (v_max - v_min)).mean()


def init_params(network, key, n, obs_dim, shared):
    """Per-agent parameters with a leading agent axis; identical copies when `shared`."""
    keys = jax.random.split(key, n)
    if shared:
        keys = jnp.repeat(keys[:1], n, axis=0)
    return jax.vmap(network.init, in_axes=(0, None))(keys, jnp.zeros((obs_dim,)))


def pooled_grads(loss_fn, params, *batches):
    """Centralized learner: one loss on the data of all agents, applied to every (identical) copy."""
    first = jax.tree.map(lambda x: x[0], params)
    pooled = [jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), b) for b in batches]
    loss, grads = jax.value_and_grad(loss_fn)(first, *pooled)
    n = jax.tree.leaves(params)[0].shape[0]
    return jnp.full((n,), loss), jax.tree.map(lambda g: jnp.broadcast_to(g, (n, *g.shape)), grads)


def hash_transitions(obs, action, w_obs, w_act):
    """32-bit hash of (s, a); collisions are negligible at buffer scale (~1e-5 per pair)."""
    q = jnp.round(obs * 16).astype(jnp.int32).astype(jnp.uint32)
    return (q * w_obs).sum(-1, dtype=jnp.uint32) + (action.astype(jnp.uint32) + 1) * w_act


def count_unique(hashes, valid):
    sentinel = jnp.uint32(0xFFFFFFFF)
    s = jnp.sort(jnp.where(valid, hashes, sentinel))
    first = jnp.concatenate([jnp.array([True]), s[1:] != s[:-1]])
    return jnp.sum(first & (s != sentinel))


def hash_weights(key, obs_dim):
    k1, k2 = jax.random.split(key)
    return jax.random.bits(k1, (obs_dim,), jnp.uint32) | 1, jax.random.bits(k2, (), jnp.uint32) | 1


def make_evaluate(env, act, eval_steps, episodes):
    """Rolls out every agent's policy from the same start states and action keys.

    act(params, obs, key) -> action. Returns per agent: mean return, a hash of the first
    episode's outcome (its last observation and action, which determine the final state in
    deterministic tasks; used for conformity / volatility as in the paper: agents conform when they
    end with the same element), achievement rates and mean episode length (episodes still running
    at the end count as eval_steps).
    """
    w_obs, w_act = hash_weights(jax.random.PRNGKey(0), env.obs_dim)

    def rollout(p, key, env_params):
        k_reset, key = jax.random.split(key)
        obs, es = env.reset(k_reset, env_params)

        def body(c, _):
            obs, es, alive, ret, h, ach, length, key = c
            key, k_act, k_step = jax.random.split(key, 3)
            a = act(p, obs, k_act)
            h = jnp.where(alive, hash_transitions(obs, a, w_obs, w_act), h)
            obs, es, r, d = env.step(k_step, es, a, env_params)
            ach = ach | (env.achievements(es) & alive & ~d)
            return (obs, es, alive & ~d, ret + r * alive, h, ach, length + alive, key), None

        init = (obs, es, jnp.array(True), jnp.array(0.0), jnp.uint32(0),
                jnp.zeros((env.num_achievements,), bool), jnp.array(0.0), key)
        (_, _, _, ret, h, ach, length, _), _ = jax.lax.scan(body, init, None, eval_steps)
        return ret, h, ach, length

    def evaluate(params, key, env_params):
        keys = jax.random.split(key, episodes)
        ret, h, ach, length = jax.vmap(lambda k: jax.vmap(rollout, in_axes=(0, None, None))(params, k, env_params))(keys)
        return ret.mean(0), h[0], ach.mean(0), length.mean(0)

    return evaluate
