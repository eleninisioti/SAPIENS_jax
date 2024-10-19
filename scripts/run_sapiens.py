import sys
import os
sys.path.append(".")
sys.path.append("envs")

from sapiens.sapiens import main
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
sys.path.append("envs")
from envs.tiny_alchemy import envs as alchemy_envs

def debug():
    env_name ="CartPole-v1"
    #env_name ="MountainCar-v0"
    #env_name = "Freeway-MinAtar"

    num_agents = 10
    main(env_name, num_agents, connectivity="fully",trial=0, local_mode=True)


def alchemy():
    env_name ="Single-path-alchemy"
    #env_name ="MountainCar-v0"
    #env_name = "Freeway-MinAtar"

    num_agents = 1
    main(env_name, num_agents, connectivity="fully",trial=0, local_mode=True)


if __name__ == "__main__":
    alchemy()