import sys
import os
sys.path.append(".")
sys.path.append("envs")

from sapiens.sapiens import main
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
sys.path.append("envs")
from envs.tiny_alchemy import envs as alchemy_envs

def parametric(env_name):

    for num_agents in [5, 10, 20]:
        for connectivity in ["fully", "dynamic"]:
            main(env_name, num_agents=num_agents, connectivity=connectivity, shared_batch_size=1, prob_visit=0.2, visit_duration=10,  trial=0, local_mode=True)
    main(env_name, num_agents=1,  shared_batch_size=1, prob_visit=0.2,
         visit_duration=10, connectivity="fully", trial=0, local_mode=True)


def alchemy():
    env_name ="Single-path-alchemy"
    #env_name ="MountainCar-v0"
    #env_name = "Freeway-MinAtar"

    num_agents = 1
    main(env_name, num_agents, shared_batch_size=1, prob_visit=0.2, visit_duration=10, connectivity="fully",trial=0, local_mode=True)


if __name__ == "__main__":
    #alchemy()
    env_name ="CartPole-v1"
    #env_name ="MountainCar-v0"
    env_name = "Freeway-MinAtar"
    #env_name = "Single-path-alchemy"
    parametric(env_name)