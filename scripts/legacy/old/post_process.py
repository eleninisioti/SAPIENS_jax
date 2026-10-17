""" Load evaluation data of trained projects
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
top_dir = "projects/for_postprocess"

envs = ["CartPole-v1", "Freeway-MinAtar", "MountainCar-v0"]
methods = ["single", "fully", "dynamic"]
num_agents_values = [5, 10, 20]
project_dirs = [os.path.join(top_dir, el) for el in os.listdir(top_dir)]
colors = ['lightblue', 'lightgreen', 'crimson', 'magenta', 'coral']
import pandas as pd
# Initialize plot
import copy as cp

def load_config(project_name):
    envs = ["CartPole-v1", "Freeway-MinAtar", "MountainCar-v0"]
    for env in envs:
        if env in project_name:
            env_name = env
            break
    if "fully" in project_name:
        method = "fully"
    elif "dynamic" in project:
        method = "dynamic"
    elif "n_1_" in project:
        method = "single"
    num_agents_values = [1, 5, 10, 20]
    for poss_num_agents in num_agents_values:
        if "n_" + str(poss_num_agents) in project_name:
            num_agents = poss_num_agents
            break
    trial_values = list(range(10))
    for trial in trial_values:
        if "trial_" + str(trial) in project_name:
            break
    return (env_name, method, num_agents, trial)

data_columns = ["env", "method", "num_agents", "trial",  "mean_mean_group_perf", "var_mean_group_perf",  "mean_max_group_perf", "var_max_group_perf"]
data = []
for project in project_dirs:
    env_name, method, num_agents, trial = load_config(project)

    eval_dirs = [os.path.join(project + "/visuals/train_seed_0", el) for el in os.listdir(project + "/visuals/train_seed_0" )]
    mean_group_perfs = []
    max_group_perfs = []
    for eval_dir in eval_dirs:
        group_rewards = []
        for agent in range(num_agents):
            try:
                with open(eval_dir + "/agent_" + str(agent) + "/rewards.txt", "r") as f:
                    content = f.read()
                    content = content[1:-3]
                    # Split the content by commas and store it in a list
                    try:
                        rewards = [float(el) for el in content.split(',')]
                    except ValueError:
                        print(f.read())
                    group_rewards.append(sum(rewards))
            except FileNotFoundError:
                #continue
                print("no file ")
        mean_group_perfs.append(np.mean(group_rewards))
        max_group_perfs.append(np.max(group_rewards))

    mean_mean_group_perf = np.mean(mean_group_perfs)
    var_mean_group_perf = np.var(mean_group_perfs)
    mean_max_group_perf = np.mean(max_group_perfs)
    var_max_group_perf = np.var(max_group_perfs)

    data.append([env_name, method, num_agents, trial, mean_mean_group_perf, var_mean_group_perf, mean_max_group_perf, var_max_group_perf])

data = pd.DataFrame(data, columns=data_columns)
total_data = cp.deepcopy(data)

metrics = ["mean_mean_group_perf", "var_mean_group_perf",  "mean_max_group_perf", "var_max_group_perf"]
top_save_dir = "visuals/"


# plot how dynamic and fully scale with number of agents
for env in envs:
    data_env = total_data[total_data["env"] ==env]

    for method in ["dynamic", "fully"]:
        data_method = data_env[data_env["method"] ==method]

        for metric in metrics:

            sns.barplot(x='num_agents', y=metric, data=data_method, palette=colors)

            save_dir = top_save_dir + "/env_" + str(env) + "_method_" + method + "_metric_" + metric
            plt.savefig(save_dir + ".png", dpi=300)
            plt.clf()




# compare fully, dynamic, single
for num_agents in [1,5, 10,20]:
    for env in envs:
        data_env = data[data["env"] == env]
        data_env = data_env[data_env["num_agents"] == num_agents]

        for metric in metrics:
            sns.barplot(x='method', y=metric, data=data_env, palette=colors)

            if data_env.empty:
                print("check")

            save_dir = top_save_dir + "/env_" + str(env) + "_metric_" + metric + "_num_agents_" + str(num_agents)
            plt.savefig(save_dir + ".png", dpi=300)
            plt.clf()


