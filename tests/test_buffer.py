import jax
import jax.numpy as jnp
import numpy as onp

from sapiens import buffer as rb


def _batch(values):
    v = jnp.asarray(values, jnp.float32)
    return dict(obs=v[:, None], action=v.astype(jnp.int32), reward=v, next_obs=v[:, None] + 0.5,
                done=jnp.zeros_like(v, bool))


def test_masked_add_keeps_order_and_skips_masked_rows():
    buf = rb.init(capacity=8, obs_dim=1)
    buf = rb.add(buf, _batch([1, 2, 3, 4]), jnp.array([True, False, True, True]))
    assert int(buf.size) == 3 and int(buf.ptr) == 3
    onp.testing.assert_array_equal(buf.reward[:3], [1, 3, 4])
    onp.testing.assert_array_equal(buf.next_obs[:3, 0], [1.5, 3.5, 4.5])


def test_wraps_around_at_capacity():
    buf = rb.init(capacity=4, obs_dim=1)
    for v in range(6):
        buf = rb.add(buf, _batch([v]), jnp.array([True]))
    assert int(buf.size) == 4
    assert sorted(buf.reward.tolist()) == [2, 3, 4, 5]


def test_sample_only_returns_filled_rows():
    buf = rb.add(rb.init(capacity=100, obs_dim=1), _batch([7, 8]), jnp.array([True, True]))
    s = rb.sample(buf, jax.random.PRNGKey(0), 64)
    assert set(s["reward"].tolist()) <= {7.0, 8.0}
    # sampled transitions stay whole: next_obs belongs to obs
    onp.testing.assert_array_equal(s["next_obs"], s["obs"] + 0.5)
