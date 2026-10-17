import sys
sys.path.append(".")
import envs
import numpy as np
import yaml
import os
from collections import defaultdict
from aim import Run, Repo
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from envs.tiny_alchemy import envs as alchemy_envs
import gymnax
import jax
from collections import Counter
import numpy as onp


task_successes = {"Single-path-alchemy": 36.0,
                  "Merging-paths-alchemy": 50.0,
                  "bestoftenpaths": 71.0}


METHOD_ALIASES = {"dynamic": "dynamic",
                  "fully": "fully-connected",
                  "independent": "no-sharing"}

METRIC_ALIASES = {"success": "Success",
                  "reward": "Reward"}


custom_colors = ["#f94144",
                 "#f3722c",
                 "#f8961e",
                 "#f9844a",
                 "#f9c74f",
                 "#90be6d",
                 "#43aa8b",
                 "#4d908e",
                 "#577590",
                 "#277da1"]
custom_palette = {
    "no-sharing": custom_colors[0],        # Soft Orange
    "dynamic": custom_colors[5],       # Light Green
    "fully-connected": custom_colors[-1]   # Peach
}


def get_trajectory_metrics(config, trial_dir):
    
    
    env = alchemy_envs.get_environment(config["ENV_NAME"], key=config["FIXED_KEY"])
    episode_length = env.episode_length


    # get action conformism

    traj_metrics = {"action_conformism": [ 0 for checkpoint in range(config["NUM_CHECKPOINTS"])],
                    "path_conformism": [ 0 for checkpoint in range(config["NUM_CHECKPOINTS"])],
                    "volatility": [ 0 for checkpoint in range(config["NUM_CHECKPOINTS"])],
                    "timesteps": [checkpoint for checkpoint in range(config["NUM_CHECKPOINTS"])]}

    for checkpoint in range(config["NUM_CHECKPOINTS"]):
        with open(trial_dir+ "/eval_data/trajectories" + str(checkpoint) + ".pkl", "rb") as f:
            checkpoint_trajs = pickle.load(f)

        checkpoint_trajs = next(iter(checkpoint_trajs))
        #trial_trajs = checkpoint_trajs[eval_trial]
        action_conformism = []
        for step in range(episode_length):
            all_actions = []
            for agent in range(config["NUM_AGENTS"]):
                action = int(checkpoint_trajs["agent_" + str(agent)][0][step]["action"])
                all_actions.append(action) # I was running two eval trials but tasks are deterministic

            counts = Counter(all_actions)
            # Find the element with the maximum count
            majority = max(counts, key=counts.get)
            action_conformism.append( onp.sum([1 if el == majority else 0 for el in all_actions]) / config["NUM_AGENTS"])

        traj_metrics["action_conformism"][checkpoint]=onp.mean(action_conformism)

    total_volatility = []
    for agent in range(config["NUM_AGENTS"]):
        changes = 0
        volatility = []
        for checkpoint in range(1,config["NUM_CHECKPOINTS"]):
            with open(trial_dir+ "/eval_data/trajectories" + str(checkpoint) + ".pkl", "rb") as f:
                checkpoint_trajs = pickle.load(f)
            checkpoint_trajs = next(iter(checkpoint_trajs))

            trajectory = []
            for step in range(episode_length):
                trajectory.append(checkpoint_trajs["agent_" + str(agent)][0][step]["action"])

            with open(trial_dir + "/eval_data/trajectories" + str(checkpoint-1) + ".pkl", "rb") as f:
                checkpoint_trajs = pickle.load(f)
            checkpoint_trajs = next(iter(checkpoint_trajs))

            prev_trajectory = []
            for step in range(episode_length):
                prev_trajectory.append(checkpoint_trajs["agent_" + str(agent)][0][step]["action"])


            if trajectory != prev_trajectory:
                changes += 1



            volatility.append(changes/(config["NUM_CHECKPOINTS"]-1))

        total_volatility.append(volatility)

    final_volatility = [0]

    for checkpoint in range(config["NUM_CHECKPOINTS"]-1):
        final_volatility.append(onp.mean([el[checkpoint] for el in total_volatility]))


    traj_metrics["volatility"] = final_volatility

    #for key, el in traj_metrics.items():
    #    traj_metrics[key] = onp.array([onp.mean(eval_el) for eval_el in el])

    return traj_metrics



def make_barplot(df, save_file, metric_name):
    method_order = ["PPO", "NEAT", "HyperNEAT"]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))



    # Bar width and spacing
    bar_width = 0.25

    #sns.barplot(data=df, x='method', y="mean")
    for idx, row in df.iterrows():
        # Plot the bar for mean
        plt.bar(row['method'], row['mean'], yerr=row['std'], capsize=5, color=custom_palette[row["method"]], edgecolor='black')




    ax.set_ylabel(metric_name)

    # Set legend at the top center with 3 columns
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)


    plt.savefig(save_file, dpi=300)
    plt.clf()



def leaderboard(top_dir):

    if not os.path.exists(top_dir + "/visuals"):
        os.makedirs(top_dir + "/visuals")

    metrics = ["success", "first_success_step_all",  "first_success_step_one"]
    projects = [el for el in os.listdir(top_dir) if ".DS_Store" not in el and ".png" not in el and "visuals" not in el]
    for metric in metrics:
        metric_data = []
        for project in projects:

            project_dir = os.path.join(top_dir, project)

            with open(project_dir + "/eval/eval_info.yaml", "r") as f:
                eval_info = yaml.load(f, Loader=yaml.SafeLoader)

            with open(project_dir + "/trial_0/config.yaml", "r") as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)

            method_name = METHOD_ALIASES[project]

            mean = eval_info["mean_" + metric]
            std = eval_info["std_" + metric]


            info = [method_name, mean, std ]
            metric_data.append(info)


        metric_data = pd.DataFrame(metric_data, columns=["method", "mean", "std"])
        make_barplot(metric_data, save_file=top_dir + "/visuals/" + metric + ".png", metric_name=metric)



    metric_names = ["timesteps",
                    "updates",
                    "loss",
                    "returns",
                    "returns_max",
                    "diversity_mean",
                    "diversity_max",
                    "diversity_proper_mean",
                    "diversity_proper_max",
                    "action_conformism",
                    "volatility"]
    # make lineplots
    plt.figure(figsize=(8, 6))
    for metric_name in metric_names:

        for project in projects:
            project_dir = os.path.join(top_dir, project)

            with open(project_dir+ "/eval/metrics.pkl", "rb") as f:
                metrics = pickle.load(f)
            df = pd.DataFrame(metrics)

            df_sorted = df.sort_values(by='timesteps')

            # Plot using seaborn lineplot, with error bars indicating the standard deviation

            sns.lineplot(data=df_sorted, x='timesteps', y=metric_name, ci='sd', label=project)


        plt.legend()
        plt.savefig(top_dir + "/visuals/" + metric_name + ".png")
        plt.clf()



    # then plot behavioral metrics



def process_project(config, trial_dir):

    logger_hash = config["aim_hash"]

    # load aim info
    aim_dir = "."

    repo = Repo(aim_dir)  # Use `.` for the current directory or provide a specific path

    metric_names = ["timesteps",
                    "updates",
                    "loss",
                    "returns",
                    "returns_max",
                    "group_diversity_mean",

                    "diversity_mean",
                    "diversity_max",
                    "diversity_proper_mean",
                    "diversity_proper_max"]
    all_metrics = {}
    for metric_name in metric_names:
        query = "metric.name == '" + metric_name + "'"  # Example query

        # result = subprocess.run(command, check=True, text=True, capture_output=True)
        # print(result.stdout)
        # Get collection of metrics
        for run_metrics_collection in repo.query_metrics(query).iter_runs():

            if run_metrics_collection.run.hash == logger_hash:

                for metric in run_metrics_collection:
                    # Get run params
                    params = metric.run[...]
                    # Get metric values
                    steps, metric_values = metric.values.sparse_numpy()

        all_metrics[metric_name] = metric_values

    max_reward, num_steps = zip(*sorted(zip(all_metrics["returns_max"], all_metrics["timesteps"])))
    mean_reward, num_steps = zip(*sorted(zip(all_metrics["returns"], all_metrics["timesteps"])))

    try:
        first_success_step_one = num_steps[max_reward.index(task_successes[config["ENV_NAME"]])]
        success = True
    except ValueError:
        success = False
        first_success_step_one = None

    try:
        first_success_step_all = num_steps[mean_reward.index(task_successes[config["ENV_NAME"]])]

    except ValueError:
        first_success_step_all = None

    run_summary = {"success": success,
                   "first_success_step_one": first_success_step_one,
                   "first_success_step_all": first_success_step_all,
                   }
    # compute summary info

    # save metrics with time


    all_beh_metrics = get_trajectory_metrics(config, trial_dir)
    #all_beh_metrics = {}

    return all_metrics, all_beh_metrics, run_summary

def viz_metrics(project_dir, metrics, beh_metrics):
    df = pd.DataFrame(metrics)

    if not os.path.exists(project_dir + "/visuals/"):
        os.makedirs(project_dir + "/visuals/")

    for metric in [el for el in df.columns if el!="timesteps"]:

        grouped = df.groupby('timesteps')[metric]
        df_sorted = df.sort_values(by='timesteps')

        # Plot using seaborn lineplot, with error bars indicating the standard deviation
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_sorted, x='timesteps', y=metric, ci='sd', label='Mean Reward')


        # Customize plot
        #plt.title('Reward vs. Num Steps with Variance')
        plt.xlabel('Number of Steps')
        #plt.ylabel('Reward')
        plt.legend()

        plt.savefig(project_dir + "/visuals/" + metric + ".png")

    df = pd.DataFrame(beh_metrics)

    if not os.path.exists(project_dir + "/visuals/"):
        os.makedirs(project_dir + "/visuals/")

    for metric in [el for el in df.columns if el != "timesteps"]:
        grouped = df.groupby('timesteps')[metric]
        df_sorted = df.sort_values(by='timesteps')

        # Plot using seaborn lineplot, with error bars indicating the standard deviation
        plt.figure(figsize=(8, 6))
        print(df_sorted[metric])
        sns.lineplot(data=df_sorted, x='timesteps', y=metric, ci='sd', label='Mean Reward')

        # Customize plot
        # plt.title('Reward vs. Num Steps with Variance')
        plt.xlabel('Number of Steps')
        # plt.ylabel('Reward')
        plt.legend()

        plt.savefig(project_dir + "/visuals/" + metric + ".png")






def process_projects(task, connectivity):

    project_dir = "projects/leaderboard/" + task + "/" + connectivity + "_parametric"
    
    projects = [os.path.join(project_dir, el) for el in os.listdir(project_dir)]
    for project_dir in projects:
        print(project_dir)
        num_trials = 4

        save_dir = project_dir + "/eval"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        total_eval_summary = defaultdict(list)

        total_metrics = defaultdict(list)
        total_beh_metrics = defaultdict(list)
        for trial in range(num_trials):
            trial_dir = os.path.join(project_dir, "trial_" + str(trial))

            with open(trial_dir + "/config.yaml", "r") as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)

            metric_values, beh_metric_values, run_summary = process_project(config, trial_dir)
            for key, values in run_summary.items():
                total_eval_summary[key].append(values)

            for key, values in metric_values.items():
                total_metrics[key].extend(values)

            for key, values in beh_metric_values.items():
                total_beh_metrics[key].extend(values)


        mean_stats = {}
        for key, val in total_eval_summary.items():
            val = [el  for el in val if el is not None ]

            if len(val):
                mean_value = float(sum(val)/len(val))

                variance = sum((x - mean_value) ** 2 for x in val) / len(val)
                std_dev = math.sqrt(variance)
            else:
                mean_value = 0
                std_dev = 0



            mean_stats["mean_" + key] = mean_value
            mean_stats["std_" + key] = std_dev

        with open(save_dir + "/eval_info.yaml", "w") as f:
            yaml.dump(mean_stats, f)


        with open(save_dir + "/metrics.pkl", "wb") as f:
            pickle.dump(total_metrics, f)


        viz_metrics(save_dir, total_metrics, total_beh_metrics)



def process_all_projects():

    tasks = ["single_path", "merging_paths", "bestoftenpaths"]
    tasks = ["Single-path-alchemy", "Merging-paths-alchemy"]
    tasks = ["Merging-paths-alchemy"]

    connectivities = ["independent", ]
    connectivities = ["fully", "independent", "dynamic"]
    connectivities = ["dynamic"]


    for task in tasks:
        for connectivity in connectivities:
            process_projects(task, connectivity)


def all_leaderboard():
    top_dirs =["projects/leaderboard/Merging-paths-alchemy" ]
    for top_dir in top_dirs:
        leaderboard(top_dir)

if __name__ == "__main__":

   process_all_projects()

   #all_leaderboard()