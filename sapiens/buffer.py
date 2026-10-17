"""Per-agent circular replay buffer that stores complete transitions (s, a, r, s', done).

Each agent owns one buffer of fixed capacity. Its own transitions and the transitions it
receives from neighbours go into the same buffer, so capacity is identical across topologies.
"""
import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class Buffer:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: chex.Array
    done: chex.Array
    ptr: chex.Array
    size: chex.Array


FIELDS = ("obs", "action", "reward", "next_obs", "done")


def init(capacity: int, obs_dim: int) -> Buffer:
    return Buffer(
        obs=jnp.zeros((capacity, obs_dim), jnp.float32),
        action=jnp.zeros((capacity,), jnp.int32),
        reward=jnp.zeros((capacity,), jnp.float32),
        next_obs=jnp.zeros((capacity, obs_dim), jnp.float32),
        done=jnp.zeros((capacity,), jnp.bool_),
        ptr=jnp.zeros((), jnp.int32),
        size=jnp.zeros((), jnp.int32),
    )


def add(buf: Buffer, batch: dict, mask: chex.Array) -> Buffer:
    """Writes the rows of `batch` (leading dim B) for which `mask` is True, in order."""
    capacity = buf.obs.shape[0]
    mask = mask.astype(jnp.int32)
    offsets = jnp.cumsum(mask) - 1
    # Masked-out rows get an out-of-bounds index and are dropped by the scatter.
    idx = jnp.where(mask > 0, (buf.ptr + offsets) % capacity, capacity)
    n = mask.sum()
    updates = {k: getattr(buf, k).at[idx].set(batch[k].astype(getattr(buf, k).dtype), mode="drop") for k in FIELDS}
    return buf.replace(
        **updates,
        ptr=(buf.ptr + n) % capacity,
        size=jnp.minimum(buf.size + n, capacity),
    )


def sample(buf: Buffer, key: chex.PRNGKey, n: int) -> dict:
    idx = jax.random.randint(key, (n,), 0, jnp.maximum(buf.size, 1))
    return {k: getattr(buf, k)[idx] for k in FIELDS}


def valid_mask(buf: Buffer) -> chex.Array:
    return jnp.arange(buf.obs.shape[0]) < buf.size
