"""Pieces shared by the learners: networks, greedy evaluation and diversity metrics."""
import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    out: int
    hidden: tuple

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(self.out)(x)


class ActorCritic(nn.Module):
    num_actions: int
    hidden: tuple

    @nn.compact
    def __call__(self, x):
        logits = MLP(self.num_actions, self.hidden)(x)
        value = MLP(1, self.hidden)(x)[..., 0]
        return logits, value


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
    episode's action sequence (for conformity / volatility) and achievement rates.
    """

    def rollout(p, key, env_params):
        k_reset, key = jax.random.split(key)
        obs, es = env.reset(k_reset, env_params)

        def body(c, _):
            obs, es, alive, ret, h, ach, key = c
            key, k_act, k_step = jax.random.split(key, 3)
            a = act(p, obs, k_act)
            obs, es, r, d = env.step(k_step, es, a, env_params)
            h = jnp.where(alive, h * jnp.uint32(1_000_003) + a.astype(jnp.uint32) + 1, h)
            ach = ach | (env.achievements(es) & alive & ~d)
            return (obs, es, alive & ~d, ret + r * alive, h, ach, key), None

        init = (obs, es, jnp.array(True), jnp.array(0.0), jnp.uint32(0),
                jnp.zeros((env.num_achievements,), bool), key)
        (_, _, _, ret, h, ach, _), _ = jax.lax.scan(body, init, None, eval_steps)
        return ret, h, ach

    def evaluate(params, key, env_params):
        keys = jax.random.split(key, episodes)
        ret, h, ach = jax.vmap(lambda k: jax.vmap(rollout, in_axes=(0, None, None))(params, k, env_params))(keys)
        return ret.mean(0), h[0], ach.mean(0)

    return evaluate
