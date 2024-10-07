import os, sys
sys.path.append(os.getcwd())
from datetime import datetime
import os

def write_file(env, num_agents, connectivity, trial):
    top_dir = "/lustre/fsn1/projects/rech/imi/utw61ti/sapiens_log/"
    current_date = datetime.today().strftime('%Y_%m_%d')
    name = "/env_" + env + "_numagents_" + str(num_agents) + "_conn_" + connectivity + "_trial_" + str(trial)

    file_name = (top_dir + "jz_scripts/" + current_date + name + ".slurm")

    if not os.path.exists(top_dir + "jz_scripts/" + current_date):
        os.makedirs(top_dir + "jz_scripts/" + current_date)

    # Open the file in write mode
    with open(file_name, "w") as file:

        file.write("#!/bin/bash"+ "\n")
        job_name = name
        file.write("#SBATCH --job-name=" + job_name + "\n")
        file.write("#SBATCH -A imi@a100" + "\n")
        file.write("#SBATCH --gres=gpu:1 "+ "\n")
        file.write("#SBATCH --time=05:00:00"+ "\n")
        file.write("#SBATCH --hint=nomultithread"+ "\n")

        output_file = name  + "%j.out"
        file.write("#SBATCH --output=" + top_dir + "jz_logs" + output_file + "\n")
        error_file = name  + "%j.err"
        file.write("#SBATCH --error=" + top_dir + "jz_logs" + error_file+ "\n")
        file.write("source ~/.bashrc"+ "\n")
        file.write("module load pytorch-gpu/py3/1.11.0 "+ "\n")
        file.write("conda activate sapiens "+ "\n")

        file.write("")
        command = "python sapiens//sapiens.py --env " + env + " --n_agents " + str(num_agents) + "  --connectivity " + connectivity + " --trial " + str(trial)
        file.write(command+ "\n")


def run_campaign():


    envs = ["CartPole-v1", "Freeway-MinAtar", "MountainCar-v0" ]
    num_agents_values = [1, 5, 10, 20]
    for env_name in envs:
        for num_agents in num_agents_values:
            for trial in range(10):
                for connectivity in ["fully", "dynamic"]:
                    write_file(env_name, num_agents, connectivity, trial)


if __name__ == "__main__":
    run_campaign()