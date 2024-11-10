import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct


@struct.dataclass
class EnvState:
    recipe_book: jnp.array
    items: jnp.array
    time: int
    held_item: int

@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 16
    init_items: int =3
    num_items: int = 11


class Base(environment.Environment):
    """
    Simplified version of Little Alchemy
    """

    def __init__(self, key, recipe="single-path"):
        super().__init__()
        self.key = key
        self.obs_shape = (4,)



    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: jnp.array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)

        #@jax.jit
        def find_matching_row(A, B):
            # Step 1: Slice A to get the first two columns that correspond to the components of the item
            A_first_two_columns = A[:, :2]

            # Step 2: Compare the first two columns of A with B
            matching_rows = jnp.all(A_first_two_columns == B, axis=1)

            # Step 3: Find the index of the matching row
            # Use jnp.argmax to get the first index where there's a True (assumes one match)
            matching_row_index = lax.cond(
                jnp.any(matching_rows),
                lambda _: jnp.argmax(matching_rows),  # If True, return the index of the matching row
                lambda _: 999,  # If False, return -1 to indicate no match
                operand=None  # The operand is not needed in this case, but is required by lax.cond
            )

            return matching_row_index

        # check that the action items are available

        valid_action = jnp.where(state.items[action], 1.0,0.0)

        # Check that the action combination is valid
        matching_row_index = find_matching_row(state.recipe_book, jnp.concatenate([state.held_item, jnp.expand_dims(action,axis=0)], axis=0))
        result = state.recipe_book[matching_row_index, 2].astype(jnp.int32)

        new_items = state.items.at[result].set(1)
        new_items = jnp.where(jnp.logical_and(matching_row_index!=999,valid_action), new_items, state.items)

        reward = jnp.where(jnp.logical_and(matching_row_index!=999,valid_action), state.recipe_book[matching_row_index, 3], 0.0)
        reward = jnp.where(prev_terminal, 0, reward)

        # check that the item did not already exist
        item_existed = jnp.logical_and(jnp.logical_and(matching_row_index!=999,valid_action), state.items[result])
        reward = jnp.where(item_existed, 0, reward)


        new_held_item= jnp.where(state.held_item==999, action, 999)
        new_held_item = jnp.where(valid_action, new_held_item, 999)

        # Update state dict and evaluate termination conditions
        state = EnvState(recipe_book=state.recipe_book,items=new_items, time=state.time + 1,held_item=new_held_item)
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        recipe_book, items = self.build_recipe_book(self.key, params)

        state = EnvState(
            recipe_book=recipe_book,
            items=items,
            time=0,
            held_item=jnp.array([999])
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        #items = jnp.where(state.items, 1.0, 0.0)
        held_item = jax.nn.one_hot(state.held_item, num_classes = state.items.shape[0])[0]
        return jnp.concatenate([held_item, state.items], axis=0)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done = jnp.all(state.items)

        # Check number of steps in episode termination condition
        done_steps = state.time >= (params.max_steps_in_episode*2)
        done = jnp.logical_or(done, done_steps)
        return done


