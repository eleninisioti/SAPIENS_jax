"""Environment factory: tiny-alchemy innovation tasks, Craftax and any gymnax environment."""
import gymnax
import jax
import jax.numpy as jnp

from envs.tiny_alchemy import envs as alchemy_envs


class Env:
    """Thin wrapper giving every environment the same flat float32 observation interface.

    `task_key` selects the task instance (the recipe book for alchemy tasks; ignored otherwise),
    so that different seeds can be vmapped over different tasks.
    """

    def __init__(self, name: str):
        self.name = name
        self.is_alchemy = name in alchemy_envs.REGISTRY
        self.is_craftax = name.startswith("Craftax")
        self.achievement_names = []
        if self.is_alchemy:
            self.env = alchemy_envs.get_environment(name)
            self.default_params = self.env.default_params
            self.max_steps = int(self.env.episode_length)
        elif self.is_craftax:
            from craftax.craftax_env import make_craftax_env_from_name
            self.env = make_craftax_env_from_name(name, auto_reset=True)
            self.default_params = self.env.default_params
            self.max_steps = int(self.default_params.max_timesteps)
            if "Classic" in name:
                from craftax.craftax_classic.constants import Achievement
            else:
                from craftax.craftax.constants import Achievement
            self.achievement_names = [a.name.lower() for a in sorted(Achievement, key=lambda a: a.value)]
        else:
            self.env, self.default_params = gymnax.make(name)
            self.max_steps = int(self.default_params.max_steps_in_episode)
        self.num_actions = self.env.action_space(self.default_params).n
        self.num_achievements = len(self.achievement_names)
        obs, _ = self.env.reset(jax.random.PRNGKey(0), self.default_params)
        self.obs_dim = int(obs.size)
        self.optimal_return = getattr(self.env, "optimal_return", None)

    def params(self, task_key):
        if self.is_alchemy:
            return self.default_params.replace(task_key=task_key)
        return self.default_params

    def reset(self, key, params):
        obs, state = self.env.reset(key, params)
        return obs.reshape(-1).astype(jnp.float32), state

    def step(self, key, state, action, params):
        obs, state, reward, done, _ = self.env.step(key, state, action, params)
        return obs.reshape(-1).astype(jnp.float32), state, reward.astype(jnp.float32), done

    def achievements(self, state):
        if self.is_craftax:
            return state.achievements.astype(bool)
        return jnp.zeros((0,), bool)
