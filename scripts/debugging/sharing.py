import jax
import jax.numpy as jnp
import numpy as onp


seed = 0
key = jax.random.PRNGKey(seed)
num_agents = 4
neighbors = jnp.array([[1,-1],[0,-1], [3,-1], [2,-1]])

shared_exp = jax.random.randint(key, shape=(4,4),minval=0, maxval=10)


print(onp.array(shared_exp))


def process_agent_share(shared_exp, neighbors):
    dummy_exp = jax.tree_map(lambda x: jnp.take(x, neighbors[0], axis=0), shared_exp)

    def process_neighbor(shared_exp, neighbor):

        def callback():

        jax.debug.callback(callback, metrics, train_state.neighbors, train_state.visiting)



        exp_from_neighbor = jax.tree_map(lambda x: jnp.take(x, neighbor, axis=0), shared_exp)

        new_exp = jax.lax.cond(neighbor != -1, lambda x: x, lambda x: dummy_exp, exp_from_neighbor)

        return new_exp

    new_exp = jax.vmap(process_neighbor, in_axes=(None, 0))(shared_exp, neighbors)
    return new_exp


shared_exp = jax.vmap(process_agent_share, in_axes=(None, 0))(shared_exp, neighbors)
temp = onp.array(shared_exp)
print(shared_exp)



