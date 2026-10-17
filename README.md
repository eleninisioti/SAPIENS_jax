# SAPIENS (JAX)

Groups of RL agents that share experiences over a social network, used to study how network
structure affects a group's ability to escape local optima. Two learners:

* `ALGO=dqn`: DQN agents that send each other transitions from their replay buffers.
* `ALGO=vtrace`: actor-critic agents that send each other trajectories, corrected with V-trace (as in IMPALA).
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
| `craftax_classic` | Craftax-Classic (Crafter in JAX), V-trace | achievement tree from wood to diamonds |

Any other [gymnax](https://github.com/RobertTLange/gymnax) environment works through `ENV_NAME`.

### Topologies (`TOPOLOGY`)

`independent` (no sharing), `fully`, `ring` (`RING_K`), `small_world` (Watts-Strogatz, `SW_K`, `SW_BETA`),
`dynamic` (groups of `GROUP_SIZE`; each agent visits another group with probability `P_VISIT` per step
for `VISIT_DURATION` steps), `periodic` (no sharing for `T_LOW` steps, then fully connected for `T_HIGH`)
and `centralized`, the usual distributed-RL setup: all agents act with one network trained on the pooled
data of all of them (Ape-X style with DQN, synchronous IMPALA with V-trace).

All step counts are per agent, summed over its `NUM_ENVS` parallel environments.

### Sharing

DQN: an agent shares at the end of each episode (`SHARE_INTERVAL=0`) or every `SHARE_INTERVAL` steps,
with probability `P_SHARE`. It then sends `SHARE_BATCH` transitions, sampled from its own buffer,
to each neighbour (a different sample for each neighbour). Every agent has one buffer of
`BUFFER_SIZE` transitions, whatever the topology.

V-trace: after every rollout an agent sends, with probability `P_SHARE`, `SHARE_BATCH` of its
`NUM_ENVS` trajectories to each neighbour, and learns from its own and the received trajectories.

## Cluster (REAL HPC)

Sweeps are defined in [scripts/sweep.py](scripts/sweep.py); each run is one Slurm array task on one GPU.
Submit every sweep from its own git worktree, so queued tasks keep running the commit they were
submitted with while the main checkout moves on:

```bash
cd ~/SAPIENS_jax && git pull
RUN=.worktrees/$(date +%Y%m%d-%H%M%S) && git worktree add --detach "$RUN" && cd "$RUN"
python3 scripts/sweep.py topologies            # dry run: writes runs.txt, prints the sbatch command
python3 scripts/sweep.py topologies --submit
```

Results go to `~/SAPIENS_jax/results/<sweep>/<run>/`, logs to `~/SAPIENS_jax/results/<sweep>/logs/`;
each log starts with the commit it ran. Remove the worktree when the sweep has finished.

## Analysis

```bash
uv run python scripts/analyze.py ~/SAPIENS_jax/results/topologies
```

Writes `summary.csv` (group success, time to first success, conformity, volatility, diversity, with
95% CIs), `stats.txt` (ANOVA and Tukey HSD across topologies) and `figures/`.

## Layout

```
sapiens/train.py      config, CLI
sapiens/dqn.py        DQN learner with replay sharing
sapiens/vtrace.py     V-trace actor-critic learner with trajectory sharing
sapiens/common.py     networks, evaluation, diversity metrics
sapiens/buffer.py     per-agent replay buffer of full (s, a, r, s', done) transitions
sapiens/topology.py   static and dynamic social networks
sapiens/envs.py       environment factory (alchemy tasks, Craftax, gymnax)
envs/tiny_alchemy/    the innovation tasks
scripts/              sweeps, Slurm job, analysis; legacy/ holds the previous implementation
tests/                unit tests
```
