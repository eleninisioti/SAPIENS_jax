import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from envs.tiny_alchemy.envs.base import Base
#from gym import spaces

@struct.dataclass
class EnvState:
    recipe_book: jnp.array
    items: jnp.array
    time: int


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 8
    n_init_items: int = 6
    n_total_items: int = 26


class Mergingpaths(Base):
    """
    Simplified version of Little Alchemy
    """

    def __init__(self, key,recipe="single-path"):
        super().__init__(key)
        self.episode_length = 16

        self.obs_shape = (14,)


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    def build_recipe_book(self, key, params):
        #init_items = jnp.arange(params.n_init_items)
        # make first branch
        n_init_items = 3
        n_total_items = 11*2+4
        max_steps_in_episode = 8
        items = jax.numpy.zeros((n_total_items,))

        recipe = jnp.zeros([max_steps_in_episode, 4]) # first item, second item, result
        key, current_key = jax.random.split(key)
        first_item = jax.random.choice(current_key, n_init_items)
        for step in range(max_steps_in_episode):

            key, current_key = jax.random.split(key)
            second_item = jax.random.choice(current_key, n_init_items)
            result = step + n_init_items
            reward = step +1
            new_comb = jnp.array([first_item, second_item, result, reward])
            recipe = recipe.at[step].set(new_comb)
            first_item = result

        recipe_first = recipe
        items = items.at[:n_init_items].set(1)

        # make second branch
        n_init_items = 3
        max_steps_in_episode = 8
        recipe = jnp.zeros([max_steps_in_episode, 4]) # first item, second item, result
        key, current_key = jax.random.split(key)
        init_items = jnp.arange(max_steps_in_episode+n_init_items, max_steps_in_episode+n_init_items+n_init_items)

        first_item = jax.random.choice(current_key, init_items)
        for step in range(max_steps_in_episode):

            key, current_key = jax.random.split(key)
            second_item = jax.random.choice(current_key, init_items)
            result = step + 2*n_init_items +max_steps_in_episode
            reward = step +1
            new_comb = jnp.array([first_item, second_item, result, reward])
            recipe = recipe.at[step].set(new_comb)

            first_item = result

        recipe_second = recipe
        items = items.at[init_items].set(1)

        # make middle branch, it will have four elements
        max_steps_in_episode_half = int(max_steps_in_episode/2)
        first_item = (n_init_items+max_steps_in_episode)*2
        recipe = jnp.zeros((max_steps_in_episode, 4)) # first item, second item, result
        reward_init = 12
        recipe=  recipe.at[0].set(jnp.array([recipe_first[1][-2], recipe_second[1][-2], first_item, reward_init]))

        total_init_items = jnp.concatenate([jnp.arange(n_init_items), init_items], axis=0)
        for step in range(1,max_steps_in_episode+1):

            key, current_key = jax.random.split(key)
            second_item = jax.random.choice(current_key, total_init_items)
            result = step + 2*n_init_items +max_steps_in_episode*2
            reward = reward_init + (step+1)*2
            new_comb = jnp.array([first_item, second_item, result,reward ])
            recipe = recipe.at[step].set(new_comb)
            first_item = result
        recipe_middle = recipe

        recipe = jnp.concatenate([recipe_first, recipe_second, recipe_middle], axis=0)


        return recipe, items

    @property
    def name(self) -> str:
        """Environment name."""
        return "Single-path-alchemy"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 26  # number of items

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        num_items = params.n_total_items # actually this is plus the init items
        return spaces.Discrete(num_items)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        num_items = params.n_total_items # actually this is plus the init items

        obs_space = spaces.Box(low=0,high=2,shape=(params.n_total_items*2), dtype=jnp.int32)

        return obs_space

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        num_items = params.max_steps_in_episode

        return spaces.Dict(
            {
                "recipe_book": spaces.Box(low=0, high=params.n_total_items, shape=(params.max_steps_in_episode,3), dtype=jnp.int32) ,
                "items": spaces.Box(low=0,high=2,shape=(params.n_total_items),dtype=jnp.int32)
            }
        )
