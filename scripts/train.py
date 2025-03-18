import sys
import os
sys.path.append(".")
sys.path.append("envs")
import sys
from sapiens.sapiens import main
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
sys.path.append("envs")
from envs.tiny_alchemy import envs as alchemy_envs

def parametric(env_name):

    lr_values = [1e-4]
    lr = lr_values[0]
    #lr = 0.2
    eps_start_values = [ 1]
    eps_end_values = [0.05]
    visit_duration_values = [10, 80, 160, 320]
    visit_duration_values =[360]
    prob_visit_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    for trial in range(10):

        for prob_visit_value in prob_visit_values:

            for visit_duration in visit_duration_values:
                for eps_start in eps_start_values:
                    #for eps_end in eps_end_values:
                    eps_end = 0.05
                    for num_agents in [10]:
                        for connectivity in [ "dynamic" ]:

                            main(env_name, learning_rate=lr, num_agents=num_agents, connectivity=connectivity, shared_batch_size=1, prob_visit=prob_visit_value, visit_duration=visit_duration,  trial=trial, local_mode=True)



def fully(env_name):

    lr_values = [1e-4]
    lr = lr_values[0]
    #lr = 0.2
    eps_start_values = [ 1]
    eps_end_values = [0.05]
    visit_duration_values = [10, 80, 160, 320]
    visit_duration_values =[360]

    for trial in range(10):

        for visit_duration in visit_duration_values:
            for eps_start in eps_start_values:
                #for eps_end in eps_end_values:
                eps_end = 0.05
                for num_agents in [10]:
                    for connectivity in ["fully" ]:

                        main(env_name, learning_rate=lr, num_agents=num_agents, connectivity=connectivity, shared_batch_size=1, prob_visit=0.01, visit_duration=visit_duration,  trial=trial, local_mode=True)



def alchemy():
    env_name ="Single-path-alchemy"
    #env_name ="MountainCar-v0"
    #env_name = "Freeway-MinAtar"

    num_agents = 5
    main(env_name, num_agents, shared_batch_size=1, prob_visit=0.2, visit_duration=10, connectivity="fully",trial=0, local_mode=True)


def train(env_name):
    #env_name ="MountainCar-v0"
    #env_name = "Freeway-MinAtar"

    
    main(env_name, learning_rate=1e-4, num_agents=10, shared_batch_size=0, prob_visit=0.2, visit_duration=10, connectivity="fully",trial=0, local_mode=True)


if __name__ == "__main__":
    #alchemy()
    #env_name ="CartPole-v1"
    #env_name = "Freeway-MinAtar"
    #env_name ="MountainCar-v0"
    env_name = "Merging-paths-alchemy"
    #env_name = "Bestoften-paths-alchemy"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
    #env_name = "Single-path-alchemy"

    #env_name = "Merging-paths"
    #parametric(env_name)
    tasks = ["Single-path-alchemy",
             "Merging-paths-alchemy",
             "Bestoften-paths-alchemy" ]

    parametric(tasks[int(sys.argv[2])])
