"""Environment factory: tiny-alchemy innovation tasks, Craftax, NAVIX (MiniGrid), Kheperax and gymnax."""
import gymnax
import jax
import jax.numpy as jnp

from envs.tiny_alchemy import envs as alchemy_envs

# Kheperax wheel-speed commands (left, right): forward, veer left / right, spin left / right
KHEPERAX_ACTIONS = jnp.array([[1.0, 1.0], [0.5, 1.0], [1.0, 0.5], [-1.0, 1.0], [1.0, -1.0]])


def _import_navix():
    """NAVIX's bundled PPO agent imports distrax / TensorFlow Probability, which break on recent
    JAX; we only use its environments, so a placeholder distrax is enough."""
    import sys
    import types
    try:
        import navix
        return navix
    except (AttributeError, ImportError):
        for name in [m for m in sys.modules if m.split(".")[0] in ("navix", "distrax")]:
            del sys.modules[name]
        stub = types.ModuleType("distrax")

        def placeholder(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return type(attr, (), {})

        stub.__getattr__ = placeholder
        sys.modules["distrax"] = stub
        import navix
        return navix


def _select(done, a, b):
    return jax.tree.map(lambda x, y: jnp.where(done, x, y), a, b)


class Env:
    """Thin wrapper giving every environment the same flat float32 observation interface.

    `task_key` selects the task instance (the recipe book for alchemy tasks; ignored otherwise),
    so that different seeds can be vmapped over different tasks.
    """

    def __init__(self, name: str, kwargs=None):
        self.name = name
        self.is_alchemy = name in alchemy_envs.REGISTRY
        self.is_craftax = name.startswith("Craftax")
        self.kind = "navix" if name.startswith("Navix") else "kheperax" if name.startswith("Kheperax") else "gymnax"
        self.achievement_names = []
        if self.kind == "navix":
            navix = _import_navix()
            self.env = navix.make(name, observation_fn=navix.observations.symbolic)
            self.default_params = None
            self.max_steps = int(self.env.max_steps)
            self.num_actions = int(self.env.action_space.n)
        elif self.kind == "kheperax":
            # Kheperax-<map>, e.g. Kheperax-standard, Kheperax-pointmaze, Kheperax-snake
            for old, new in (("tree_leaves", jax.tree.leaves), ("tree_map", jax.tree.map),
                             ("tree_flatten", jax.tree.flatten), ("tree_unflatten", jax.tree.unflatten)):
                if not hasattr(jax, old):  # aliases removed from recent JAX but used by Kheperax
                    setattr(jax, old, new)
            from kheperax.tasks.target import TargetKheperaxConfig, TargetKheperaxTask
            config = TargetKheperaxConfig.get_default_for_map(name.split("-", 1)[1])
            self.env = TargetKheperaxTask(config)
            self.default_params = None
            self.max_steps = int(config.episode_length)
            self.num_actions = len(KHEPERAX_ACTIONS)
        elif name == "DeceptiveCoins-v0":
            from envs.deceptive_coins import DeceptiveCoins
            self.env = DeceptiveCoins()
            self.default_params = self.env.default_params
            self.max_steps = int(self.default_params.max_steps_in_episode)
        elif self.is_alchemy:
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
            self.env, self.default_params = gymnax.make(name, **(kwargs or {}))
            self.max_steps = int(self.default_params.max_steps_in_episode)
        if self.kind == "gymnax":
            self.num_actions = self.env.action_space(self.default_params).n
        self.num_achievements = len(self.achievement_names)
        obs, _ = self._raw_reset(jax.random.PRNGKey(0), self.default_params)
        self.obs_dim = int(obs.size)
        self.obs_shape = tuple(obs.shape)
        self.optimal_return = getattr(self.env, "optimal_return", None)

    def params(self, task_key):
        if self.is_alchemy:
            return self.default_params.replace(task_key=task_key)
        return self.default_params

    def _raw_reset(self, key, params):
        if self.kind == "navix":
            ts = self.env.reset(key)
            return ts.observation, ts
        if self.kind == "kheperax":
            ks = self.env.reset(key)
            ks = ks.replace(reward=jnp.float32(ks.reward), done=jnp.asarray(ks.done))
            return ks.obs, (ks, jnp.int32(0))
        return self.env.reset(key, params)

    def reset(self, key, params):
        obs, state = self._raw_reset(key, params)
        return obs.reshape(-1).astype(jnp.float32), state

    def step(self, key, state, action, params):
        """Steps and auto-resets: after a terminal step the returned obs/state are a fresh episode."""
        if self.kind == "navix":
            ts = self.env.step(state, action)
            done = ts.step_type > 0
            obs, reward = ts.observation, ts.reward
        elif self.kind == "kheperax":
            ks, t = state
            ks = self.env.step(ks, KHEPERAX_ACTIONS[action])
            ks = ks.replace(reward=jnp.float32(ks.reward), done=jnp.asarray(ks.done))
            done = ks.done | (t + 1 >= self.max_steps)
            ts, obs, reward = (ks, t + 1), ks.obs, ks.reward
        else:
            obs, state, reward, done, _ = self.env.step(key, state, action, params)
            return obs.reshape(-1).astype(jnp.float32), state, reward.astype(jnp.float32), done
        reset_obs, reset_state = self.reset(key, params)
        obs = jnp.where(done, reset_obs, obs.reshape(-1).astype(jnp.float32))
        return obs, _select(done, reset_state, ts), jnp.float32(reward), done

    def achievements(self, state):
        if self.is_craftax:
            return state.achievements.astype(bool)
        return jnp.zeros((0,), bool)
