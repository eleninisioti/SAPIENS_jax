"""Deceptive Coins (after Bontrager et al., 2019, "Superstition in the network"), level 1.

A grid world with two paths from the start cell. To the left, a diamond one step away and then a
fire: return 2, found almost immediately. To the right, three diamonds spread out and a better
fire: return 5, but only for an agent that leaves the easy reward behind. Once the agent steps to
one side a barrier closes behind it, so each episode commits to one path. Touching a fire or
running out of time ends the episode.

Observation: (H, W, 3) planes for the agent, the remaining diamonds and the fires.
"""
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

HEIGHT, WIDTH = 5, 13
START = (2, 6)
LEFT_DIAMONDS = ((2, 5),)
RIGHT_DIAMONDS = ((2, 9), (0, 11), (4, 11))
LEFT_FIRE, RIGHT_FIRE = (2, 0), (2, 12)
LEFT_FIRE_REWARD, RIGHT_FIRE_REWARD = 1.0, 2.0
MOVES = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right


def _grid(cells):
    g = jnp.zeros((HEIGHT, WIDTH), bool)
    for r, c in cells:
        g = g.at[r, c].set(True)
    return g


@struct.dataclass
class EnvState:
    pos: chex.Array       # (2,) row, column
    diamonds: chex.Array  # (H, W) remaining diamonds
    side: chex.Array      # -1 left, 0 undecided, 1 right
    time: int


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 20


class DeceptiveCoins(environment.Environment):
    optimal_return = len(RIGHT_DIAMONDS) + RIGHT_FIRE_REWARD   # 5
    deceptive_return = len(LEFT_DIAMONDS) + LEFT_FIRE_REWARD   # 2

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        state = EnvState(pos=jnp.array(START), diamonds=_grid(LEFT_DIAMONDS + RIGHT_DIAMONDS),
                         side=jnp.array(0), time=0)
        return self.get_obs(state), state

    def step_env(self, key: chex.PRNGKey, state: EnvState, action, params: EnvParams):
        new = jnp.clip(state.pos + MOVES[action], jnp.array([0, 0]), jnp.array([HEIGHT - 1, WIDTH - 1]))
        # the barrier behind a committed agent: it cannot cross back to (or past) the start column
        blocked = ((state.side == -1) & (new[1] >= START[1])) | ((state.side == 1) & (new[1] <= START[1]))
        pos = jnp.where(blocked, state.pos, new)
        side = jnp.where(state.side != 0, state.side, jnp.sign(pos[1] - START[1]))

        diamond = state.diamonds[pos[0], pos[1]]
        at_left = (pos[0] == LEFT_FIRE[0]) & (pos[1] == LEFT_FIRE[1])
        at_right = (pos[0] == RIGHT_FIRE[0]) & (pos[1] == RIGHT_FIRE[1])
        reward = diamond.astype(jnp.float32) + LEFT_FIRE_REWARD * at_left + RIGHT_FIRE_REWARD * at_right

        state = EnvState(pos=pos, diamonds=state.diamonds.at[pos[0], pos[1]].set(False), side=side,
                         time=state.time + 1)
        done = at_left | at_right | (state.time >= params.max_steps_in_episode)
        return jax.lax.stop_gradient(self.get_obs(state)), state, reward, done, {"discount": 1.0 - done}

    def get_obs(self, state: EnvState):
        agent = jnp.zeros((HEIGHT, WIDTH)).at[state.pos[0], state.pos[1]].set(1.0)
        fires = _grid((LEFT_FIRE, RIGHT_FIRE)).astype(jnp.float32)
        return jnp.stack([agent, state.diamonds.astype(jnp.float32), fires], axis=-1)

    def is_terminal(self, state: EnvState, params: EnvParams):
        return state.time >= params.max_steps_in_episode

    @property
    def name(self) -> str:
        return "DeceptiveCoins-v0"

    @property
    def num_actions(self) -> int:
        return len(MOVES)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(MOVES))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0, 1, (HEIGHT, WIDTH, 3), jnp.float32)
