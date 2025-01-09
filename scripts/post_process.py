import numpy as np
import yaml
import os
from collections import defaultdict
from aim import Run, Repo


def process_project(config):

    logger_hash = config["aim_hash"]

    # load aim info
    aim_dir = "server_aim"

    repo = Repo(aim_dir)  # Use `.` for the current directory or provide a specific path

    metric_names = ["timesteps",
                    "updates",
                    "loss",
                    "returns",
                    "returns_max",
                    "diversity_mean",
                    "diversity_max",
                    "diversity_proper_mean",
                    "diversity_proper_max"]
    metric_values = {}
    for metric in metric_names:
        query = "metric.name == '" + metric + "'"  # Example query

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

    metric_values[metric] = metric_values

    run_summary = {"success": 0, "steps_to_sucess": 0}
    # compute summary info

    # save metrics with time

    return metric_values, run_summary

def process_projects(top_dir):
    num_trials = 4
    projects = os.listdir(top_dir)
    for project in projects:
        project_dir = os.path.join(top_dir, project)

        with open(project_dir + "/config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        total_eval_summary = defaultdict(list)

        for trial in range(num_trials):

            metric_values, run_summary = process_project(config)

            for key, values in run_summary.items():
                total_eval_summary[key].append(values)

        mean_stats = {}
        for key, val in total_eval_summary.items():
            mean_stats["mean_" + key] = float(np.mean(val))
            mean_stats["std_" + key] = float(np.std(val))

        with open(project_dir + "/eval_info.yaml", "w") as f:
            yaml.dump(mean_stats, f)





if __name__ == "__main__":

    top_dir = "projects/for_leaderboard/LA/merging_paths/dynamic"
    process_projects(top_dir)