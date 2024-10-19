from envs.tiny_alchemy.envs import single_path
from envs.tiny_alchemy.envs import merging_paths
from envs.tiny_alchemy.envs import bestoften_paths

from envs.tiny_alchemy.envs.base import Base

_envs = {"Single-path-alchemy": single_path.Singlepath,
        "Merging-paths-alchemy": merging_paths.Mergingpaths,
         "Bestoften-paths-alchemy": bestoften_paths.Bestoftenpaths

         }

def get_environment(env_name, **kwargs) -> Base:
  """Returns an environment from the environment registry.

  Args:
    env_name: environment name string
    **kwargs: keyword arguments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  return _envs[env_name](**kwargs)