import sys
import os
sys.path.append(".")
from sapiens.sapiens import main
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"


def debug():
    env_name ="CartPole-v1"
    num_agents = 10

    main(env_name, num_agents, connectivity="dynamic",trial=0)




if __name__ == "__main__":
    debug()