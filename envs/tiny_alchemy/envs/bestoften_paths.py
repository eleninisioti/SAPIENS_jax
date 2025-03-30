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
    n_init_items: int = 3*10
    n_total_items: int = 11*10


class Bestoftenpaths(Base):
    """
    Simplified version of Little Alchemy
    """

    def __init__(self, key, recipe="single-path"):
        super().__init__(key)
        self.episode_length = 16

        self.obs_shape = (14,)


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    def build_recipe_book(self, key, params):

        #init_items = jnp.arange(params.n_init_items)
        keys = jax.random.split(key, 10)
        n_init_items = 3
        n_total_items = 11
        max_steps_in_episode = 8

        def sample_path(key, path_idx, lucky_path):
            recipe = jnp.zeros([max_steps_in_episode, 4]) # first item, second item, result
            key, current_key = jax.random.split(key)

            init_items = jnp.arange((max_steps_in_episode + n_init_items)*path_idx,
                                    (max_steps_in_episode + n_init_items)*path_idx + n_init_items)
            first_item = jax.random.choice(current_key, init_items)
            for step in range(max_steps_in_episode):

                key, current_key = jax.random.split(key)
                second_item = jax.random.choice(current_key, init_items)
                result = step + init_items[-1] + 1
                reward = step + 1
                reward = jnp.where(lucky_path, reward*2, reward)
                new_comb = jnp.array([first_item, second_item, result, reward])
                recipe = recipe.at[step].set(new_comb)

                first_item = result
            return recipe, init_items
        #idxs = jnp.arange(10)
        total_recipe = []

        # pick lucky path
        lucky_path = jax.random.choice(key, 10)
        total_init_items = []
        for i in range(10):
            recipe, init_items = sample_path(keys[i], i, lucky_path==i)
            total_recipe.append(recipe)
            total_init_items.append(init_items)
        recipe = jnp.concatenate(total_recipe, axis=0)
        items = jax.numpy.zeros((n_total_items*10,))
        total_init_items = jnp.concatenate(total_init_items)
        items = items.at[total_init_items].set(1)

        return recipe, items

    @property
    def name(self) -> str:
        """Environment name."""
        return "Single-path-alchemy"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 110  # number of items

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
                "recipe_book": spaces.Box(low=0, high=params.n_total_items, shape=(params.max_steps_in_episode,4), dtype=jnp.int32) ,
                "items": spaces.Box(low=0,high=2,shape=(params.n_total_items),dtype=jnp.int32)
            }
        )
