import jax
import jax.numpy as jnp
import numpy as onp


seed = 0
key = jax.random.PRNGKey(seed)
num_agents = 4
neighbors = jnp.array([[1,3],[0,-1], [3,1], [2,-1]])

group_shared_exp = jax.random.randint(key, shape=(4,4),minval=0, maxval=10)


print(onp.array(group_shared_exp))
neighbor_id = 0
receiver_id = 1
temp = jnp.take(group_shared_exp, neighbor_id, axis=0)
received_exp = jax.tree_map(lambda x: jnp.take(temp, receiver_id, axis=0),
                            group_shared_exp)


def get_exps_for_agent(group_shared_exp, receiver_id):
    agent_neighbors = jnp.take(neighbors, receiver_id, axis=0)
    fixed_neighbor = agent_neighbors[0]

    dummy_exp =  jax.tree_map(lambda x: jnp.take(jnp.take(x, fixed_neighbor, axis=0), receiver_id, axis=0),
                                    group_shared_exp)

    def get_exp_from_neighbor(neighbor_id):
        received_exp = jax.tree_map(lambda x: jnp.take(jnp.take(x, neighbor_id, axis=0), receiver_id, axis=0),
                                    group_shared_exp)
        #received_exp = jax.lax.cond(neighbor_id == -1, lambda x: x, lambda x: dummy_exp, received_exp)

        #return received_exp

        received_exp = jax.lax.cond(neighbor_id == -1,  lambda x: dummy_exp,lambda x: x, received_exp)
        return received_exp


    received_exp = jax.vmap(get_exp_from_neighbor)(agent_neighbors)

    return received_exp




shared_exp = jax.vmap(get_exps_for_agent, in_axes=(None, 0))(group_shared_exp, jnp.arange(num_agents) )
temp = onp.array(shared_exp)
print(shared_exp)



