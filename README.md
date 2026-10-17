# SAPIENS (JAX)

Groups of DQN agents that share experiences from their replay buffers over a social network,
used to study how network structure affects a group's ability to escape local optima.
Follow-up to [Nisioti et al., 2022](https://arxiv.org/abs/2206.05060).

A whole experiment (all agents and all seeds of one configuration) is one jitted function,
so it runs on a single GPU.

## Setup

```bash
uv sync            # installs JAX with CUDA on Linux, CPU JAX on macOS
uv run pytest      # unit tests and a short learning check (~2 min on CPU)
```

## Running

```bash
uv run python -m sapiens.train --config configs/merging_paths.yaml \
    --set TOPOLOGY=dynamic NUM_AGENTS=10 --num_seeds 20 --output results/merging_dynamic
```

`--set KEY=VALUE` overrides any key in `DEFAULTS` in [sapiens/train.py](sapiens/train.py).
Each run writes `config.yaml`, `metrics.npz` (arrays shaped `(seeds, log steps, ...)`) and `summary.json`.

### Environments

| Config | Environment | Why |
| --- | --- | --- |
| `single_path` | Alchemy, one innovation path | no local optima |
| `merging_paths` | Alchemy, two paths that must be combined | two deceptive local optima |
| `bestoften_paths` | Alchemy, ten paths, one optimal | large search space |
| `deepsea` | bsuite DeepSea | standard hard-exploration benchmark |
| `mountaincar` | MountainCar-v0 | sparse reward, "do nothing" is a local optimum |
| `minatar_*` | MinAtar Breakout / Freeway / SpaceInvaders / Asterix | standard DQN benchmarks |

Any other [gymnax](https://github.com/RobertTLange/gymnax) environment works through `ENV_NAME`.

### Topologies (`TOPOLOGY`)

`independent` (no sharing), `fully`, `ring` (`RING_K`), `small_world` (Watts-Strogatz, `SW_K`, `SW_BETA`),
`dynamic` (groups of `GROUP_SIZE`; each agent visits another group with probability `P_VISIT` per step
for `VISIT_DURATION` steps) and `periodic` (no sharing for `T_LOW` steps, then fully connected for `T_HIGH`).

### Sharing

An agent shares at the end of each episode (`SHARE_INTERVAL=0`) or every `SHARE_INTERVAL` steps,
with probability `P_SHARE`. It then sends `SHARE_BATCH` transitions, sampled from its own buffer,
to each neighbour (a different sample for each neighbour). Every agent has one buffer of
`BUFFER_SIZE` transitions, whatever the topology.

## Cluster (REAL HPC)

Sweeps are defined in [scripts/sweep.py](scripts/sweep.py); each run is one Slurm array task on one GPU.
On the front node:

```bash
cd ~/SAPIENS_jax
python3 scripts/sweep.py topologies            # dry run: writes runs.txt, prints the sbatch command
python3 scripts/sweep.py topologies --submit
```

Results go to `~/SAPIENS_jax/results/<sweep>/<run>/`, logs to `~/SAPIENS_jax/results/<sweep>/logs/`.
To keep editing while jobs run, submit from a git worktree (see `docs/server.md`).

## Analysis

```bash
uv run python scripts/analyze.py ~/SAPIENS_jax/results/topologies
```

Writes `summary.csv` (group success, time to first success, conformity, volatility, diversity, with
95% CIs), `stats.txt` (ANOVA and Tukey HSD across topologies) and `figures/`.

## Layout

```
sapiens/train.py      training loop, config, CLI
sapiens/buffer.py     per-agent replay buffer of full (s, a, r, s', done) transitions
sapiens/topology.py   static and dynamic social networks
sapiens/envs.py       environment factory (alchemy tasks + gymnax)
envs/tiny_alchemy/    the innovation tasks
scripts/              sweeps, Slurm job, analysis; legacy/ holds the previous implementation
tests/                unit tests
```
