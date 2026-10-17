"""Writes the runs file for a sweep and submits it as one Slurm array.

Standard library only, so it can run on the cluster front node without a virtualenv:

    python3 scripts/sweep.py topologies            # print the sbatch command
    python3 scripts/sweep.py topologies --submit   # submit it

Each run trains all seeds of one (config, overrides) combination on one GPU.
"""
import argparse
import itertools
import os
import subprocess

TOPOLOGIES = ["independent", "fully", "ring", "small_world", "dynamic", "periodic", "centralized"]
ALCHEMY = ["single_path", "merging_paths", "bestoften_paths"]
TRADITIONAL = ["deepsea", "mountaincar", "minatar_breakout", "minatar_freeway",
               "minatar_spaceinvaders", "minatar_asterix"]

# name -> list of (config, grid of overrides, number of seeds, hours)
SWEEPS = {
    "smoke": [("merging_paths", {"TOPOLOGY": ["fully", "dynamic"], "TOTAL_STEPS": [200_000]}, 4, 1)],
    "pilot": [(c, {"TOPOLOGY": TOPOLOGIES}, 20, 4) for c in ["merging_paths", "deepsea"]],
    # shared DQN hyper-parameters, tuned without sharing on the task without local optima so that
    # the choice cannot favour any topology; then frozen for every topology and task
    "tune": [("single_path", {"TOPOLOGY": ["independent"], "LR": [1e-4, 1e-3], "GAMMA": [0.9, 0.99],
                              "EPS_FRACTION": [0.1, 0.5], "TARGET_UPDATE_INTERVAL": [1_000, 10_000]}, 10, 2)],
    "pilot_vtrace": [(c, {"TOPOLOGY": TOPOLOGIES, "ALGO": ["vtrace"], "NUM_ENVS": [16], "ROLLOUT_LEN": [16],
                          "LR": [1e-3], "LOG_INTERVAL": [10_240]}, 20, 6) for c in ["merging_paths", "deepsea"]],
    "craftax_bench": [("craftax_classic", {"TOPOLOGY": ["dynamic"], "TOTAL_STEPS": [4_194_304]}, 1, 2),
                      ("craftax_classic", {"TOPOLOGY": ["dynamic"], "TOTAL_STEPS": [4_194_304]}, 4, 2)],
    "craftax": [("craftax_classic", {"TOPOLOGY": TOPOLOGIES}, 5, 24)],
    "topologies": [(c, {"TOPOLOGY": TOPOLOGIES}, 20, 12) for c in ALCHEMY + TRADITIONAL[:2]]
                  + [(c, {"TOPOLOGY": TOPOLOGIES}, 10, 24) for c in TRADITIONAL[2:]],
    "group_size": [(c, {"TOPOLOGY": TOPOLOGIES, "NUM_AGENTS": [2, 6, 20, 50]}, 20, 12)
                   for c in ["merging_paths", "deepsea"]],
    "sharing": [(c, {"TOPOLOGY": ["fully", "dynamic"], "SHARE_BATCH": [1, 6, 36], "P_SHARE": [0.35, 0.7, 1.0]}, 20, 12)
                for c in ["merging_paths", "deepsea"]],
    "dynamic_params": [(c, {"TOPOLOGY": ["dynamic"], "P_VISIT": [0.001, 0.01, 0.1], "VISIT_DURATION": [16, 160, 1600]}, 20, 12)
                       for c in ["merging_paths", "bestoften_paths"]],
}


def expand(sweep):
    runs = []
    for config, grid, seeds, hours in SWEEPS[sweep]:
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            name = "__".join([config] + [f"{k}={v}" for k, v in zip(keys, values)] + [f"seeds={seeds}"] * (sweep == "craftax_bench"))
            args = f"--config configs/{config}.yaml --num_seeds {seeds} --set " + " ".join(
                f"{k}={v}" for k, v in zip(keys, values))
            runs.append((name, args, hours))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", choices=sorted(SWEEPS))
    parser.add_argument("--results", default=os.path.expanduser("~/SAPIENS_jax/results"))
    parser.add_argument("--max_parallel", type=int, default=6, help="GPUs used at once (fair to other users)")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    out = os.path.join(args.results, args.sweep)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    runs = expand(args.sweep)
    runs_file = os.path.join(out, "runs.txt")
    with open(runs_file, "w") as f:
        f.writelines(f"{name} {a}\n" for name, a, _ in runs)

    hours = max(h for _, _, h in runs)
    cmd = ["sbatch", f"--array=0-{len(runs) - 1}%{args.max_parallel}", f"--job-name=sapiens-{args.sweep}",
           f"--time={hours}:00:00", f"--output={out}/logs/%A_%a.log", "scripts/job.sh", runs_file]
    print(f"{len(runs)} runs written to {runs_file}")
    print(" ".join(cmd))
    if args.submit:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
