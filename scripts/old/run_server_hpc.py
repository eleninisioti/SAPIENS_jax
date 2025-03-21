import os, sys
sys.path.append(os.getcwd())
from datetime import datetime
import os

def write_file(env, num_agents, learning_rate, connectivity, shared_batch_size, prob_visit, visit_duration, trial):
    top_dir = ""
    current_date = datetime.today().strftime('%Y_%m_%d')
    name = ("/env_" + env + "_numagents_" + str(num_agents) + "_conn_" + connectivity + "_shared_batch_" + str(shared_batch_size)
            + "_prob_visit_" + str(prob_visit) + "_visit_dur_" + str(visit_duration) + "_lr_" + str(learning_rate) + "_trial_" + str(trial))

    file_name = (top_dir + "jz_scripts/" + current_date + name + ".slurm")

    if not os.path.exists(top_dir + "jz_scripts/" + current_date):
        os.makedirs(top_dir + "jz_scripts/" + current_date)

    print(file_name)
    # Open the file in write mode
    with open(file_name, "w") as file:

        file.write("#!/bin/bash"+ "\n")
        job_name = name
        file.write("#SBATCH --job-name=" + job_name + "\n")
        #file.write("#SBATCH -A imi@cpu" + "\n")
        file.write("#SBATCH --time=05:00:00"+ "\n")
        file.write("#SBATCH --cpus-per-task=8"+ "\n")
        file.write("#SBATCH --gres=gpu:a100_40gb " + "\n")
        #

        #file.write("#SBATCH --partition=actlr"+ "\n")
        output_file = name  + "%j.out"
        file.write("#SBATCH --output=" + top_dir + "jz_logs" + output_file + "\n")
        error_file = name  + "%j.err"
        file.write("#SBATCH --error=" + top_dir + "jz_logs" + error_file+ "\n")
        file.write("source /home/enis/.bashrc"+ "\n")

        file.write("module load Anaconda3 "+ "\n")
        file.write("conda activate sapiens "+ "\n")


        file.write("")
        command = "python sapiens/sapiens.py --env " + env + " --n_agents " + str(num_agents) + "  --connectivity " + connectivity + "  --connectivity " + connectivity + " --trial " + str(trial) +  " --visit_duration " + str(visit_duration) + " --prob_visit " + str(prob_visit) + " --shared_batch_size " + str(shared_batch_size) + " --learning_rate " + str(learning_rate) + " --local_mode"
        file.write(command+ "\n")


def alchemy():
    envs = ["Single-path-alchemy", "Merging-paths-alchemy"]
    num_agents_values = [1]
    for env_name in envs:
        for num_agents in num_agents_values:
            for trial in range(3):
                for connectivity in ["fully"]:
                    write_file(env_name, num_agents, connectivity, 1, 0.2, 10, trial)


    num_agents_values = [5, 10, 20]
    for env_name in envs:
        for num_agents in num_agents_values:
            for trial in range(3):
                for connectivity in ["fully", "dynamic"]:
                    write_file(env_name, num_agents, connectivity, 1, 0.2, 10, trial)


    num_agents_values = [5, 10, 20]
    for env_name in envs:
        for num_agents in num_agents_values:
            for prob_visit in [0.05, 0.1, 0.2, 0.5]:
                for visit_duration in [1, 5, 10, 20]:
                    for trial in range(3):
                        for connectivity in ["dynamic"]:
                            write_file(env_name, num_agents, connectivity, 1, prob_visit, visit_duration, trial)

def anal_freeway():
    envs = ["Freeway-MinAtar"]
    num_agents_values = [1, 5, 10, 20, 50]
    for shared_batch_size in [1, 5, 10]:
        for prob_visit in [0.05, 0.1, 0.2, 0.5]:
            for visit_duration in [1, 5, 10, 20]:
                for env_name in envs:
                    for num_agents in num_agents_values:
                        for trial in range(3):
                            for connectivity in ["dynamic"]:
                                write_file(env_name, num_agents, connectivity, shared_batch_size, prob_visit, visit_duration, trial)




def parametric(env_name):

    for num_agents in [10]:
        for connectivity in [ "dynamic"]:
            for trial in range(10):
                write_file(env_name, num_agents=num_agents, learning_rate=0.0001,connectivity=connectivity, shared_batch_size=1, prob_visit=0.01, visit_duration=10,  trial=trial)
    #write_file(env_name, num_agents=1,  shared_batch_size=1, prob_visit=0.2,
    #     visit_duration=10, connectivity="fully", trial=0)



if __name__ == "__main__":
    #env_name ="Single-path-alchemy"
    env_name ="Merging-paths-alchemy"

    parametric(env_name)