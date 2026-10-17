"""Summarises a sweep: paper metrics per run, significance tests across topologies, and figures.

    uv run python scripts/analyze.py ~/SAPIENS_jax/results/topologies

Writes summary.csv, stats.txt and figures/ inside the sweep directory. Runs that differ only in
TOPOLOGY form one comparison group (e.g. one environment at one group size).

Metrics (per seed, from greedy evaluations at every log step):
  S    group success: some agent reached SUCCESS_THRESHOLD at some point
  T+   time to first success (TOTAL_STEPS when never reached)
  T*   time until every agent succeeded at the same evaluation
  conformity  fraction of agents following the majority trajectory, averaged over training
  volatility  fraction of evaluations at which an agent's trajectory changed
  diversity   unique (s, a) pairs per buffer and in the union of the group's buffers
"""
import argparse
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
import yaml
from scipy import stats

TOPOLOGY_ORDER = ["independent", "fully", "ring", "small_world", "dynamic", "periodic"]
LABELS = {"independent": "no-sharing", "fully": "fully-connected", "ring": "ring",
          "small_world": "small-world", "dynamic": "dynamic", "periodic": "periodic"}
# fixed categorical slots (reference palette, light mode): colour follows the topology
COLORS = dict(zip(TOPOLOGY_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def load_runs(sweep_dir):
    runs = []
    for name in sorted(os.listdir(sweep_dir)):
        path = os.path.join(sweep_dir, name)
        if not os.path.exists(os.path.join(path, "metrics.npz")):
            continue
        with open(os.path.join(path, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        m = dict(onp.load(os.path.join(path, "metrics.npz")))
        group = "__".join(p for p in name.split("__") if not p.startswith("TOPOLOGY="))
        runs.append(dict(name=name, group=group, cfg=cfg, m=m))
    return runs


def first_step(mask, steps, total):
    """First logged step at which mask holds, per seed (total if never)."""
    hit = mask.any(-1)
    return onp.where(hit, steps[mask.argmax(-1)], total)


def run_metrics(run):
    m, cfg = run["m"], run["cfg"]
    ret = m["eval_return"]                       # (seeds, logs, agents)
    steps = m["step"][0]
    total = cfg["TOTAL_STEPS"]
    n_agents = ret.shape[-1]
    out = dict(final_best_return=ret[:, -1].max(-1), final_mean_return=ret[:, -1].mean(-1))
    thr = cfg.get("SUCCESS_THRESHOLD")
    if thr is not None:
        ok = ret >= thr - 1e-6
        out["S"] = ok.any(-1).any(-1).astype(float)
        out["T_first"] = first_step(ok.any(-1), steps, total)
        out["T_all"] = first_step(ok.all(-1), steps, total)

    hashes = m["traj_hash"]                      # (seeds, logs, agents)
    out["conformity"] = onp.array([[max(Counter(h.tolist()).values()) / n_agents for h in seed] for seed in hashes]).mean(-1)
    out["volatility"] = (hashes[:, 1:] != hashes[:, :-1]).mean((1, 2))
    out["diversity"] = m["diversity"].mean((1, 2))
    out["group_diversity"] = m["group_diversity"].mean(1)
    return out


def ci95(x):
    x = onp.asarray(x, float)
    return 1.96 * x.std(ddof=1) / onp.sqrt(len(x)) if len(x) > 1 else 0.0


def wilson(p, n, z=1.96):
    """95% Wilson interval for a proportion, as (lower error, upper error)."""
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * onp.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p - (centre - half), (centre + half) - p


def significance(groups, key):
    """One-way ANOVA across topologies, then Tukey HSD for pairwise comparisons."""
    names = [t for t in TOPOLOGY_ORDER if t in groups]
    samples = [onp.asarray(groups[t][key], float) for t in names]
    if len(samples) < 2 or all(onp.ptp(s) == 0 for s in samples):
        return f"  {key}: no variance, tests skipped\n"
    text = f"  {key}: ANOVA p = {stats.f_oneway(*samples).pvalue:.3g}\n"
    res = stats.tukey_hsd(*samples)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p = res.pvalue[i, j]
            if p < 0.05:
                text += f"    {LABELS[names[i]]} vs {LABELS[names[j]]}: p = {p:.3g}\n"
    return text


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)


def plot_curves(group_runs, key, ylabel, path, reduce_agents):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for run in sorted(group_runs, key=lambda r: TOPOLOGY_ORDER.index(r["cfg"]["TOPOLOGY"])):
        topo = run["cfg"]["TOPOLOGY"]
        y = run["m"][key]
        y = reduce_agents(y) if y.ndim == 3 else y      # (seeds, logs)
        x = run["m"]["step"][0]
        mean, err = y.mean(0), 1.96 * y.std(0, ddof=1) / onp.sqrt(len(y)) if len(y) > 1 else 0
        ax.plot(x, mean, color=COLORS[topo], linewidth=2, label=LABELS[topo])
        ax.fill_between(x, mean - err, mean + err, color=COLORS[topo], alpha=0.15, linewidth=0)
    _style(ax, "Environment steps per agent", ylabel)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bars(groups, key, ylabel, path):
    names = [t for t in TOPOLOGY_ORDER if t in groups]
    vals = [onp.mean(groups[t][key]) for t in names]
    if key == "S":
        errs = onp.array([wilson(v, len(groups[t][key])) for t, v in zip(names, vals)]).T
    else:
        errs = [ci95(groups[t][key]) for t in names]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(range(len(names)), vals, yerr=errs, width=0.6, color=[COLORS[t] for t in names],
           edgecolor="white", linewidth=2, error_kw=dict(ecolor=INK_2, capsize=3, linewidth=1))
    ax.set_xticks(range(len(names)), [LABELS[t] for t in names], rotation=30, ha="right")
    _style(ax, "", ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir")
    args = parser.parse_args()

    runs = load_runs(args.sweep_dir)
    fig_dir = os.path.join(args.sweep_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rows, by_group = [], defaultdict(dict)
    for run in runs:
        met = run_metrics(run)
        by_group[run["group"]][run["cfg"]["TOPOLOGY"]] = met
        row = dict(group=run["group"], topology=LABELS[run["cfg"]["TOPOLOGY"]], seeds=len(met["final_best_return"]))
        for k, v in met.items():
            row[k], row[k + "_ci95"] = float(onp.mean(v)), float(ci95(v))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.sweep_dir, "summary.csv"), index=False)

    with open(os.path.join(args.sweep_dir, "stats.txt"), "w") as f:
        for group, groups in by_group.items():
            f.write(f"{group}\n")
            for key in ("S", "T_first", "final_best_return", "group_diversity"):
                if all(key in g for g in groups.values()):
                    f.write(significance(groups, key))

    for group, groups in by_group.items():
        group_runs = [r for r in runs if r["group"] == group]
        plot_curves(group_runs, "eval_return", "Best agent's return", os.path.join(fig_dir, f"{group}__best_return.png"),
                    lambda y: y.max(-1))
        plot_curves(group_runs, "eval_return", "Mean agent return", os.path.join(fig_dir, f"{group}__mean_return.png"),
                    lambda y: y.mean(-1))
        plot_curves(group_runs, "group_diversity", "Group diversity (unique s, a)",
                    os.path.join(fig_dir, f"{group}__group_diversity.png"), None)
        if all("S" in g for g in groups.values()):
            plot_bars(groups, "S", "Group success", os.path.join(fig_dir, f"{group}__success.png"))
            plot_bars(groups, "T_first", "Time to first success", os.path.join(fig_dir, f"{group}__time_to_success.png"))
    print(pd.DataFrame(rows).to_string(index=False, columns=["group", "topology", "seeds"] +
                                       [c for c in ("S", "T_first", "final_best_return", "group_diversity") if c in rows[0]]))


if __name__ == "__main__":
    main()
