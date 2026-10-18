"""Deceptive Coins (after Bontrager et al., 2019, "Superstition in the network"), level 1.

A grid world with two paths from the start cell. To the left, a diamond one step away and then a
fire: return 2, found almost immediately. To the right, three diamonds spread out and a better
fire: return 5, but only for an agent that leaves the easy reward behind. Once the agent steps to
one side a barrier closes behind it, so each episode commits to one path. Touching a fire or
running out of time ends the episode.

Two layouts: v0 is the small level of the 2022 paper; v1 puts the right-hand rewards further away
on a larger grid, so the easy path is relatively more tempting.

Observation: (H, W, 4) planes for the agent, the remaining diamonds, the fires, and the fraction of
the time limit left. Time is observable so that running out of time is not mistaken for a dead end
(Pardo et al., 2018); this matters when agents learn from each other's episodes.
"""
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

LAYOUTS = {
    "v0": dict(height=5, width=13, start=(2, 6), left_diamonds=((2, 5),), left_fire=(2, 0),
               right_diamonds=((2, 9), (0, 11), (4, 11)), right_fire=(2, 12), max_steps=20),
    "v1": dict(height=7, width=21, start=(3, 10), left_diamonds=((3, 9),), left_fire=(3, 6),
               right_diamonds=((3, 15), (0, 18), (6, 18)), right_fire=(3, 20), max_steps=30),
}
LEFT_FIRE_REWARD, RIGHT_FIRE_REWARD = 1.0, 2.0
MOVES = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right


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
    optimal_return = 3 + RIGHT_FIRE_REWARD   # 5
    deceptive_return = 1 + LEFT_FIRE_REWARD  # 2

    def __init__(self, layout="v0"):
        super().__init__()
        for k, v in LAYOUTS[layout].items():
            setattr(self, k, v)
        self.layout = layout

    def _grid(self, cells):
        g = jnp.zeros((self.height, self.width), bool)
        for r, c in cells:
            g = g.at[r, c].set(True)
        return g

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.max_steps)

    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        state = EnvState(pos=jnp.array(self.start), diamonds=self._grid(self.left_diamonds + self.right_diamonds),
                         side=jnp.array(0), time=0)
        return self.get_obs(state), state

    def step_env(self, key: chex.PRNGKey, state: EnvState, action, params: EnvParams):
        start_col = self.start[1]
        new = jnp.clip(state.pos + MOVES[action], jnp.array([0, 0]), jnp.array([self.height - 1, self.width - 1]))
        # the barrier behind a committed agent: it cannot cross back to (or past) the start column
        blocked = ((state.side == -1) & (new[1] >= start_col)) | ((state.side == 1) & (new[1] <= start_col))
        pos = jnp.where(blocked, state.pos, new)
        side = jnp.where(state.side != 0, state.side, jnp.sign(pos[1] - start_col))

        diamond = state.diamonds[pos[0], pos[1]]
        at_left = (pos[0] == self.left_fire[0]) & (pos[1] == self.left_fire[1])
        at_right = (pos[0] == self.right_fire[0]) & (pos[1] == self.right_fire[1])
        reward = diamond.astype(jnp.float32) + LEFT_FIRE_REWARD * at_left + RIGHT_FIRE_REWARD * at_right

        state = EnvState(pos=pos, diamonds=state.diamonds.at[pos[0], pos[1]].set(False), side=side,
                         time=state.time + 1)
        done = at_left | at_right | (state.time >= params.max_steps_in_episode)
        return jax.lax.stop_gradient(self.get_obs(state)), state, reward, done, {"discount": 1.0 - done}

    def get_obs(self, state: EnvState):
        agent = jnp.zeros((self.height, self.width)).at[state.pos[0], state.pos[1]].set(1.0)
        fires = self._grid((self.left_fire, self.right_fire)).astype(jnp.float32)
        time_left = jnp.full((self.height, self.width), 1.0 - state.time / self.max_steps)
        return jnp.stack([agent, state.diamonds.astype(jnp.float32), fires, time_left], axis=-1)

    def is_terminal(self, state: EnvState, params: EnvParams):
        return state.time >= params.max_steps_in_episode

    @property
    def name(self) -> str:
        return f"DeceptiveCoins-{self.layout}"

    @property
    def num_actions(self) -> int:
        return len(MOVES)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(MOVES))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0, 1, (self.height, self.width, 4), jnp.float32)
