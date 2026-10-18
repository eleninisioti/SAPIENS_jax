"""Writes the runs file for a sweep and submits it as one Slurm array.

Standard library only, so it can run on the cluster front node without a virtualenv:

    python3 scripts/sweep.py topologies                        # print the sbatch command
    python3 scripts/sweep.py topologies --submit               # submit it (REAL HPC)
    python3 scripts/sweep.py topologies --cluster itu --tasks 7-13 --submit   # part of it on ITU HPC

Each run trains all seeds of one (config, overrides) combination on one GPU.
"""
import argparse
import itertools
import os
import subprocess

TOPOLOGIES = ["independent", "fully", "ring", "small_world", "dynamic", "periodic", "centralized",
              "adaptive_diversity", "adaptive_discovery"]
ALCHEMY = ["single_path", "merging_paths", "bestoften_paths"]
TRADITIONAL = ["deepsea", "mountaincar", "minatar_breakout", "minatar_freeway",
               "minatar_spaceinvaders", "minatar_asterix"]

# name -> list of (config, grid of overrides, number of seeds, hours[, first seed])
SWEEPS = {
    "smoke": [("merging_paths", {"TOPOLOGY": ["fully", "dynamic"], "TOTAL_STEPS": [200_000]}, 4, 1)],
    # each topology's own parameters, tuned on seeds disjoint from the reported ones (0-19)
    "topo_params": [("merging_paths", grid, 10, 4, 1000) for grid in [
        {"TOPOLOGY": ["dynamic"], "P_VISIT": [0.001, 0.01, 0.1], "VISIT_DURATION": [16, 160, 1600]},
        {"TOPOLOGY": ["periodic"], "T_LOW": [160, 1600, 16000], "T_HIGH": [16, 160, 1600]},
        {"TOPOLOGY": ["ring"], "RING_K": [2, 4, 6]},
        {"TOPOLOGY": ["small_world"], "SW_BETA": [0.05, 0.2, 0.5]},
    ]],
    # first pass on the standard benchmarks (5 seeds): does DQN learn, and do topologies differ?
    "newenvs": [(c, {"TOPOLOGY": TOPOLOGIES}, 5, 6)
                for c in ["minatar_freeway", "minatar_breakout", "navix_doorkey8", "kheperax"]],
    "adaptive_params2": [("merging_paths", grid, 10, 4, 1000) for grid in [
        {"TOPOLOGY": ["adaptive_discovery"], "T_HIGH": [500, 1600, 5000]},
        {"TOPOLOGY": ["adaptive_diversity"], "D_LOW": [0.05, 0.15, 0.25], "D_HIGH": [0.35, 0.55, 0.75]},
    ]],
    # main alchemy comparison with the frozen topology parameters, on the reported seeds 0-19
    "alchemy_main": [("bestoften_paths", {"TOPOLOGY": TOPOLOGIES[:7]}, 20, 12),
                     ("merging_paths", {"TOPOLOGY": TOPOLOGIES[:7]}, 20, 4)],
    "navix_vtrace": [("navix_doorkey8_vtrace", {"TOPOLOGY": TOPOLOGIES}, 5, 8)],
    "navix_vtrace_long": [("navix_doorkey8_vtrace", {"TOPOLOGY": TOPOLOGIES, "TOTAL_STEPS": [80_609_280]}, 5, 8)],
    # single-learner tuning (no sharing) before any topology comparison on a task
    "tune_kheperax": [("kheperax", {"TOPOLOGY": ["independent"], "LR": [2.5e-4, 1e-3], "EPS_FRACTION": [0.1, 0.5],
                                    "EPS_END": [0.05, 0.2], "GAMMA": [0.99, 0.995]}, 10, 4, 1000),
                      ("kheperax", {"TOPOLOGY": ["independent"], "ALGO": ["vtrace"], "NUM_ENVS": [16], "ROLLOUT_LEN": [32],
                                    "LOG_INTERVAL": [51_200], "LR": [3e-4, 1e-3], "ENT_COEF": [0.01, 0.05]}, 10, 4, 1000)],
    "tune_navix": [("navix_doorkey8_vtrace", {"TOPOLOGY": ["independent"], "LR": [1e-4, 3e-4, 7e-4],
                                              "ENT_COEF": [0.01, 0.03], "TOTAL_STEPS": [40_304_640]}, 10, 6, 1000)],
    "tune_coins": [(c, {"TOPOLOGY": ["independent"], "LR": [1e-4, 1e-3], "GAMMA": [0.9, 0.99],
                        "EPS_FRACTION": [0.1, 0.5]}, 10, 3, 1000) for c in ["deceptive_coins", "deceptive_coins_v1"]],
    # diagnostic: is the collapse under sharing TD divergence, sharing volume, or both?
    "coins_diag": [("deceptive_coins_v1", {"TOPOLOGY": ["periodic", "fully", "independent"], "SHARE_INTERVAL": [16],
                                           "SHARE_BATCH": [1, 16], "DQN_LOSS": ["mse", "huber"], "DQN_GRAD_CLIP": [10]}, 20, 2)],
    "coins_recent": [("deceptive_coins_v1", {"TOPOLOGY": ["periodic", "fully", "dynamic"], "SHARE_INTERVAL": [16],
                                             "SHARE_SOURCE": ["recent", "buffer"]}, 20, 2)],
    "coins_time": [("deceptive_coins_v1", {"TOPOLOGY": ["independent", "centralized", "fully", "periodic", "dynamic"],
                                           "SHARE_INTERVAL": [16]}, 20, 2)],
    # robustness of the main merging result to an observable time limit
    "merging_time": [("merging_paths", {"TOPOLOGY": TOPOLOGIES[:7], "TIME_OBS": [True]}, 20, 4)],
    "tune_coins_v2": [("deceptive_coins_v2", {"TOPOLOGY": ["independent"], "LR": [1e-4, 1e-3], "EPS_FRACTION": [0.1, 0.5],
                                              "EPS_END": [0.05, 0.1]}, 10, 3, 1000)],
    "tune_kheperax_pos": [("kheperax_pos", {"TOPOLOGY": ["independent"], "LR": [2.5e-4, 1e-3], "EPS_FRACTION": [0.1, 0.5],
                                            "EPS_END": [0.05, 0.2]}, 10, 4, 1000),
                          ("kheperax_pos", {"TOPOLOGY": ["independent"], "ALGO": ["vtrace"], "NUM_ENVS": [16], "ROLLOUT_LEN": [32],
                                            "LOG_INTERVAL": [51_200], "LR": [3e-4, 1e-3], "ENT_COEF": [0.01, 0.05]}, 10, 4, 1000)],
    "bestoften_time": [("bestoften_paths", {"TOPOLOGY": TOPOLOGIES[:7], "TIME_OBS": [True]}, 20, 16),
                       ("bestoften_paths", {"TOPOLOGY": ["adaptive_diversity"], "TIME_OBS": [True]}, 20, 16),
                       ("bestoften_paths", {"TOPOLOGY": ["adaptive_discovery"], "T_HIGH": [1600], "TIME_OBS": [True]}, 20, 16)],
    "kheperax_main": [("kheperax_pos", {"TOPOLOGY": TOPOLOGIES, "TOTAL_STEPS": [10_000_000]}, 20, 6)],
    "coins_main": [("deceptive_coins_v1", {"TOPOLOGY": TOPOLOGIES}, 20, 2)],
    # sharing on a fixed clock: at episode ends, agents on the short deceptive path share ~5x more often
    "coins_fixed_share": [("deceptive_coins_v1", {"TOPOLOGY": TOPOLOGIES, "SHARE_INTERVAL": [16]}, 20, 2)],
    "navix6": [("navix_doorkey6", {"TOPOLOGY": TOPOLOGIES}, 5, 6)],
    # adaptive topologies on the reported seeds; discovery uses its own tuned broadcast window
    "alchemy_adaptive": [(c, grid, 20, h) for c, h in [("merging_paths", 4), ("bestoften_paths", 12)] for grid in [
        {"TOPOLOGY": ["adaptive_diversity"]},
        {"TOPOLOGY": ["adaptive_discovery"], "T_HIGH": [1600]},
    ]],
    "single_path": [("single_path", {"TOPOLOGY": TOPOLOGIES}, 20, 2)],
    "deepsea_size": [("deepsea", {"ENV_KWARGS.size": [24, 32], "EVAL_STEPS": [32],
                                  "TOPOLOGY": ["independent", "fully", "ring", "periodic", "centralized"]}, 20, 4)],
    "pilot2": [(c, {"TOPOLOGY": TOPOLOGIES}, 20, 4) for c in ["merging_paths", "deepsea"]],
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
    for config, grid, seeds, hours, *first_seed in SWEEPS[sweep]:
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            name = "__".join([config] + [f"{k}={v}" for k, v in zip(keys, values)] + [f"seeds={seeds}"] * (sweep == "craftax_bench"))
            offset = f" --seed_offset {first_seed[0]}" if first_seed else ""
            args = f"--config configs/{config}.yaml --num_seeds {seeds}{offset} --set " + " ".join(
                f"{k}={v}" for k, v in zip(keys, values))
            runs.append((name, args, hours))
    return runs


# extra sbatch options per cluster (command-line options override the #SBATCH lines in job.sh)
CLUSTERS = {
    "real": [],
    # ITU HPC: scavenge has no preemption and a 1 day limit; skip the 8-11 GB desktop / 1080 Ti GPUs
    "itu": ["--partition=scavenge", "--cpus-per-task=4", "--mem=48G", "--mail-type=NONE",
            "--exclude=desktop[1,6-8,12,15],cn16,cn17"],
}
# convolutions fail with cuDNN errors on the ITU V100 nodes with the current JAX / CUDA wheels
ITU_NO_V100 = "--exclude=desktop[1,6-8,12,15],cn[3-6],cn16,cn17"


def uses_cnn(runs):
    return any("NETWORK: cnn" in open(f"configs/{a.split()[1].split('/')[-1]}").read() for _, a, _ in runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", choices=sorted(SWEEPS))
    parser.add_argument("--results", default=os.path.expanduser("~/SAPIENS_jax/results"))
    parser.add_argument("--max_parallel", type=int, default=6, help="GPUs used at once (fair to other users)")
    parser.add_argument("--cluster", choices=sorted(CLUSTERS), default="real")
    parser.add_argument("--tasks", help="subset of run indices, e.g. 0-6 or 1,4,9 (default: all)")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    out = os.path.join(args.results, args.sweep)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    runs = expand(args.sweep)
    runs_file = os.path.join(out, "runs.txt")
    with open(runs_file, "w") as f:
        f.writelines(f"{name} {a}\n" for name, a, _ in runs)

    hours = max(h for _, _, h in runs)
    tasks = args.tasks or f"0-{len(runs) - 1}"
    options = list(CLUSTERS[args.cluster])
    if args.cluster == "itu" and uses_cnn(runs):
        options = [ITU_NO_V100 if o.startswith("--exclude") else o for o in options]
    cmd = ["sbatch", f"--array={tasks}%{args.max_parallel}", f"--job-name=sapiens-{args.sweep}",
           f"--time={hours}:00:00", f"--output={out}/logs/%A_%a.log", *options,
           "scripts/job.sh", runs_file]
    print(f"{len(runs)} runs written to {runs_file}")
    print(" ".join(cmd))
    if args.submit:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
