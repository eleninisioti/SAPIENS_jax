import jax
import jax.numpy as jnp

def build_recipe_book( key):
    # init_items = jnp.arange(params.n_init_items)
    keys = jax.random.split(key, 10)
    n_init_items = 3
    n_total_items = 11
    max_steps_in_episode = 8

    def sample_path(key, path_idx, lucky_path):
        recipe = jnp.zeros([max_steps_in_episode, 4])  # first item, second item, result
        key, current_key = jax.random.split(key)

        init_items = jnp.arange((max_steps_in_episode + n_init_items) * path_idx,
                                (max_steps_in_episode + n_init_items) * path_idx + n_init_items)
        first_item = jax.random.choice(current_key, init_items)
        for step in range(max_steps_in_episode):
            key, current_key = jax.random.split(key)
            second_item = jax.random.choice(current_key, init_items)
            result = step + init_items[-1] +1
            reward = step + 1
            reward = jnp.where(lucky_path, reward * 2, reward)
            new_comb = jnp.array([first_item, second_item, result, reward])
            recipe = recipe.at[step].set(new_comb)

            first_item = result
        return recipe, init_items

    # idxs = jnp.arange(10)
    total_recipe = []

    # pick lucky path
    lucky_path = jax.random.choice(key, 10)
    total_init_items = []
    for i in range(10):
        recipe, init_items = sample_path(keys[i], i, lucky_path == i)
        total_recipe.append(recipe)
        total_init_items.append(init_items)
    recipe = jnp.concatenate(total_recipe, axis=0)
    items = jax.numpy.zeros((n_total_items * 10,))
    total_init_items = jnp.concatenate(total_init_items)
    items = items.at[total_init_items].set(1)

    return recipe, items


build_recipe_book(key=jax.random.PRNGKey(0))
