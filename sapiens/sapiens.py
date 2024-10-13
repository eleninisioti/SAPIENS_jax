"""
PureJaxRL version of CleanRL's DQN: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py
"""
import os


import jax
import jax.numpy as jnp

import chex
import flax
import wandb
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import gymnax
import flashbax as fbx
import matplotlib.pyplot as plt
import numpy as onp
import argparse
import pickle
from gymnax.visualize import Visualizer
class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int
    buffer_diversity: float
    neighbors: jnp.array
    keep_neighbors: jnp.array
    visiting: int


def make_train(config):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"]

    # we create the group

    basic_env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        tx = optax.adam(learning_rate=lr)

        network = QNetwork(action_dim=env.action_space(env_params).n)

        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batch_size=1+config["NUM_NEIGHBORS"]*config["SHARED_BATCH_SIZE"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        def init_agent(rng, agent_id):
            rng, _rng = jax.random.split(rng)

            init_obs, env_state = env.reset(_rng)

            #init_obs = init_obs[0,...]

            # INIT BUFFER

            rng = jax.random.PRNGKey(0)  # use a dummy rng here
            _action = basic_env.action_space().sample(rng)
            _, _env_state = env.reset(rng, env_params)
            _obs, _, _reward, _done, _ = env.step(rng, _env_state, _action, env_params)
            _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
            buffer_state = buffer.init(_timestep)

            # INIT NETWORK AND OPTIMIZER
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(env.observation_space(env_params).shape)
            network_params = network.init(_rng, init_x)

            neighbors = config["initial_graph"][agent_id]


            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                target_network_params=jax.tree_map(lambda x: jnp.copy(x), network_params),
                tx=tx,
                timesteps=0,
                visiting = 0,
                keep_neighbors=jnp.zeros_like(neighbors),
                n_updates=0,
                buffer_diversity=0.0,
                neighbors=neighbors,
            )
            return train_state, env_state, init_obs, buffer_state

        # INIT ENV
        rng, _rng = jax.random.split(rng)

        agent_keys = jax.random.split(_rng, config["NUM_AGENTS"])
        agent_ids = jnp.arange(config["NUM_AGENTS"])

        train_states, env_state, init_obs, buffer_states = jax.vmap(init_agent)(agent_keys, agent_ids)

        # epsilon-greedy exploration
        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(
                rng, 2
            )  # a key for sampling random actions and one for picking
            eps = jnp.clip(  # get epsilon
                (
                    (config["EPSILON_FINISH"] - config["EPSILON_START"])
                    / config["EPSILON_ANNEAL_TIME"]
                )
                * t
                + config["EPSILON_START"],
                config["EPSILON_FINISH"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
            chosed_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape)
                < eps,  # pick the actions that should be random
                jax.random.randint(
                    rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
                ),  # sample random actions,
                greedy_actions,
            )

            return chosed_actions

        # TRAINING LOOP
        def _update_step(runner_state, unused):


            train_state, buffer_state, env_state, last_obs, rng = runner_state

            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = jax.vmap(network.apply)(train_state.params, last_obs)
            rng_as = jax.random.split(rng_a, config["NUM_AGENTS"])
            action = jax.vmap(eps_greedy_exploration)(rng_as, q_vals, train_state.timesteps)
            # explore with epsilon greedy_exploration
            rng_ss = jax.random.split(rng_s, config["NUM_AGENTS"])
            #env_state = jax.tree_map(lambda x: x[:, 0], env_state)
            action = action

            obs, env_state, reward, done, info = jax.vmap(env.step)(rng_ss, env_state, action)


            train_state = train_state.replace(
                timesteps=train_state.timesteps + 1
            )  # update timesteps count

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)

            timestep = jax.tree_map(lambda x: jnp.expand_dims(x,1), timestep)

            # add the shared experiencees
            def sample_buffer(buffer_state, rng):
                keys = jax.random.split(rng, config["NUM_NEIGHBORS"])
                exp = jax.vmap(buffer.sample, in_axes=(None, 0))(buffer_state,
                                                                 keys)  # actually we need n_agents samples
                exp = exp.experience.first
                exp = jax.tree_map(lambda x: x[:,:config["SHARED_BATCH_SIZE"],...], exp)
                return exp



            shared_exp = jax.vmap(sample_buffer)(buffer_state, agent_keys)


            shared_exp = jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])), shared_exp)
            total_exp = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1),timestep, shared_exp  )
            #total_exp = jax.tree_map(lambda x: jnp.expand_dims(x,axis=2),total_exp)
            #total_exp = timestep + shared_exp #TODO: just concatenate here
            #total_exp = timestep
            buffer_state = jax.vmap(buffer.add)(buffer_state, total_exp)

            buffer_obs = buffer_state.experience.obs

            def get_diversity(array):
                array = array.reshape(-1, array.shape[-1])
                return jnp.mean(jnp.var(array, axis=0))


            def _compute_diversity(train_state):
                diversity =jax.vmap(get_diversity)(buffer_obs)

                train_state = train_state.replace(buffer_diversity=diversity)
                return train_state


            def _is_diversity_time(buffer_state, train_state):
                value = (
                        (buffer.can_sample(buffer_state))
                        &
                        (  # enough experience in buffer
                                train_state.timesteps > config["LEARNING_STARTS"]
                        )
                        & (  # pure exploration phase ended
                                train_state.timesteps % config["DIVERSITY_INTERVAL"] == 0
                        )  # training interval
                )
                #value = True
                return value

            rng, _rng = jax.random.split(rng)

            is_diversity_time = jax.vmap(_is_diversity_time)(buffer_state, train_state)
            is_diversity_time = is_diversity_time[0]

            rng_group = jax.random.split(rng, config["NUM_AGENTS"])
            _rng_group = jax.random.split(_rng, config["NUM_AGENTS"])

            train_state = jax.lax.cond(
                is_diversity_time,
                lambda train_state, rng: _compute_diversity(train_state),
                lambda train_state, rng: train_state,  # do nothing
                train_state,
                _rng_group,
            )


            # NETWORKS UPDATE
            def _learn_phase(train_state, buffer_state, rng):

                learn_batch = buffer.sample(buffer_state, rng).experience

                q_next_target = network.apply(
                    train_state.target_network_params, learn_batch.second.obs
                )  # (batch_size, num_actions)
                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
                target = (
                    learn_batch.first.reward
                    + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target
                )

                def _loss_fn(params):

                    q_vals = network.apply(
                        params, learn_batch.first.obs
                    )  # (batch_size, num_actions)
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)

                    loss = jnp.mean((chosen_action_qvals - target) ** 2)

                    return loss

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, loss

            def is_learn_time(buffer_state, train_state):
                value = (
                        (buffer.can_sample(buffer_state))
                        & (  # enough experience in buffer
                                train_state.timesteps > config["LEARNING_STARTS"]
                        )
                        & (  # pure exploration phase ended
                                train_state.timesteps % config["TRAINING_INTERVAL"] == 0
                        )  # training interval
                )
                return  value


            def is_visit_time(buffer_state, train_state, key):

                value = (
                        (buffer.can_sample(buffer_state))
                        & (  # enough experience in buffer
                                train_state.timesteps > config["LEARNING_STARTS"]
                        )
                        & (  # pure exploration phase ended
                               jax.random.uniform(key) < config["PROB_VISIT"]
                        )  # training interval
                        & (  # pure exploration phase ended
                                jnp.logical_not(train_state.visiting)
                        )  # training interval
                        & (  # pure exploration phase ended
                            config["CONNECTIVITY"] == "dynamic"
                        )  # training interval

                )
                return  value

            def is_return_time(buffer_state, train_state, key):
                # START HERE
                value = (
                        (buffer.can_sample(buffer_state))
                        & (  # enough experience in buffer
                                train_state.timesteps > config["LEARNING_STARTS"]
                        )
                        & (  # pure exploration phase ended
                            train_state.timesteps == train_state.visiting + config["VISIT_DURATION"]
                        )  # training interval
                        & (  # pure exploration phase ended
                                config["CONNECTIVITY"] == "dynamic"
                        )  # training interval
                )
                return  value



            rng, _rng = jax.random.split(rng)
            is_learn_time = jax.vmap(is_learn_time)(buffer_state, train_state)
            is_learn_time = is_learn_time[0]

            rng_group = jax.random.split(rng, config["NUM_AGENTS"])
            _rng_group = jax.random.split(_rng, config["NUM_AGENTS"])

            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: jax.vmap(_learn_phase)(train_state, buffer_state, rng_group),
                lambda train_state, rng: (train_state, jnp.array([0.0]*config["NUM_AGENTS"])),  # do nothing
                train_state,
                _rng_group,
            )

            def _implement_visit(train_state, agent_id, to_visit):
                # choose another subgroup

                # current neighbors of agent about to take a visit
                prev_neighbors = train_state.neighbors # just for keeping

                # new neighbors of agent about to a visit
                second_neighbor = train_state.neighbors[to_visit][0]
                new_neighbors = jnp.concatenate([jnp.array([second_neighbor]), jnp.array([to_visit])], axis=0)
                updated_neighbors = prev_neighbors.at[agent_id].set(new_neighbors)

                # new neighbors of agents receiving the agent
                update_neighbors = jnp.concatenate([jnp.array([second_neighbor]), jnp.array([agent_id]) ],axis=0)
                updated_neighbors = updated_neighbors.at[to_visit].set(update_neighbors)
                update_neighbors = jnp.concatenate([jnp.array([to_visit]), jnp.array([agent_id]) ],axis=0)
                updated_neighbors = updated_neighbors.at[second_neighbor].set(update_neighbors)

                train_state = train_state.replace(neighbors=updated_neighbors, keep_neighbors=prev_neighbors, visiting=train_state.timesteps)
                return train_state

            def _check_visit(is_visit_time, train_state, key, agent_id):
                train_state = jax.lax.cond(is_visit_time,
                                           lambda train_state, key, agent_id: _implement_visit(train_state, key, agent_id),
                                           lambda train_state, _, agent_id: train_state, train_state, key, agent_id)
                return train_state


            def _implement_return(train_state):
                train_state = train_state.replace(neighbors=train_state.keep_neighbors, visiting=jnp.zeros_like(train_state.visiting))
                return train_state


            def _return_visit(is_return_time, train_state):
                train_state = jax.lax.cond(is_return_time,
                                           lambda train_state: _implement_return(train_state),
                                           lambda train_state: train_state, train_state)
                return train_state


            # implement visits
            if config["CONNECTIVITY"] == "dynamic":
                is_visit_time = jax.vmap(is_visit_time)(buffer_state, train_state, _rng_group)
                agents = jnp.arange(config["NUM_AGENTS"])
                _rng, visit_key = jax.random.split(_rng)
                to_visit = jax.random.choice(visit_key, agents)
                _rng, visit_key = jax.random.split(_rng)
                visitor_id = jax.random.choice(visit_key, agents)
                train_state = _check_visit(is_visit_time[0], train_state, visitor_id, to_visit)
                #train_state = train_state.replace(visiting=is_visit_time[0])

                is_return_time = jax.vmap(is_return_time)(buffer_state, train_state, _rng_group)
                train_state = _return_visit(is_return_time[0], train_state)

            #train_state = jax.vmap(_check_visit, in_axes=(0, None,0,0))(is_visit_time, train_state, _rng_group, agent_ids)

            def update_target(train_state):
                new_state = jax.lax.cond(
                    train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                    lambda train_state: train_state.replace(
                        target_network_params=optax.incremental_update(
                            train_state.params,
                            train_state.target_network_params,
                            config["TAU"],
                        )
                    ),
                    lambda train_state: train_state,
                    operand=train_state,
                )

                return new_state

            # update target network
            train_state = jax.vmap(update_target)(train_state)

            # share experience

            #"""


            #"""
            #buffer_state = jax.vmap(jax.vmap(buffer.add))(buffer_state, exp)

            metrics = {
                "timesteps": train_state.timesteps[...,0],
                "updates": train_state.n_updates[...,0],
                "loss": loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
                "loss_max": loss.max(),
                "returns_max": info["returned_episode_returns"].max(),
                "mnemonic_diversity_mean": train_state.buffer_diversity.mean(),
                "mnemonic_diversity_max": train_state.buffer_diversity.max()
            }

            # report on wandb if required
            if config.get("WANDB_MODE", "disabled") == "online":

                def callback(metrics):
                    if metrics["timesteps"] % 100 == 0:
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            #env_state = jax.tree_map(lambda x: jnp.expand_dims(x, -1),env_state)

            runner_state = (train_state, buffer_state, env_state, obs, rng)

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)

        #init_obs = jax.tree_map(lambda x: x[0, ...], init_obs)
        runner_state = (train_states, buffer_states, env_state, init_obs, _rng)

        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])


        return {"runner_state": runner_state, "metrics": metrics}

    return train


def init_connectivity(config):
    if config["CONNECTIVITY"] == "fully":
        config["NUM_NEIGHBORS"] = config["NUM_AGENTS"]-1

        neighbors = [onp.arange(config["NUM_AGENTS"]).tolist() for _ in range(config["NUM_AGENTS"])]
        initial_graph = []
        for idx, el in enumerate(neighbors):
            el.remove(idx)
            initial_graph.append(el )

    elif config["CONNECTIVITY"] == "dynamic":
        config["NUM_NEIGHBORS"] = 2 # start with one neighbor but due to visits the maximum is two

        group_id = 0
        neighbors = []
        for i in range(config["NUM_AGENTS"]):
            neighbors.append([group_id, group_id+1])

            if i%2 == 1:
                group_id +=2

        initial_graph = []
        for idx, el in enumerate(neighbors):
            el.remove(idx)
            el.append(-1)
            initial_graph.append(el) # -1 means it is an empty neighbor spot

        """
        for 


        def connect_fully(agent_id):
            mask = jnp.arange(config["NUM_NEIGHBORS"]) != agent_id  # Create a mask for all indices except `idx`
            neighbors= jnp.arange(config["NUM_NEIGHBORS"])
            neighbors = neighbors[mask]
            return neighbors
        agent_ids = jnp.arange(config["NUM_AGENTS"])
        initial_graph = jax.vmap(connect_fully)(agent_ids)
        """
    config["initial_graph"] = jnp.array(initial_graph)

    return config


def evaluate(train_state, config, train_seed):
    """ Evaluates a trained policy
    """
    if config["local_mode"]:
        top_dir = "projects/"
    else:
        top_dir = "/lustre/fsn1/projects/rech/imi/utw61ti/sapiens_log/projects/"


    mean_rewards = []
    max_rewards = []
    for trial in range(config["num_eval_trials"]):



        key = jax.random.PRNGKey(trial)

        basic_env, env_params = gymnax.make(config["ENV_NAME"])
        env = FlattenObservationWrapper(basic_env)
        env = LogWrapper(env)

        network = QNetwork(action_dim=env.action_space(env_params).n)
        agent_rewards = []

        for agent in range(config["NUM_AGENTS"]):

            save_dir = top_dir + config["project_name"] + "/visuals/train_seed_" + str(train_seed) + "/trial_" + str(trial) + "/agent_" + str(agent)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            key, agent_key = jax.random.split(key)

            agent_params = jax.tree_map(lambda x: x[agent], train_state.params)
            last_obs, env_state = env.reset(agent_key)

            done = False
            ep_reward = []
            state_seq = []
            while not done:
                state_seq.append(env_state.env_state)

                q_vals = network.apply(agent_params, last_obs)
                action = jnp.argmax(q_vals, axis=-1)  # get the greedy actions

                key, key_act = jax.random.split(key)
                last_obs, env_state, reward, done, info = env.step(key_act, env_state, action)
                ep_reward.append(float(reward))
            agent_rewards.append(onp.sum(ep_reward))

            with open(save_dir + "/rewards.txt", "a") as f:
                f.write(str(ep_reward) + ",\n")

            with open(save_dir + "/traj.pkl", "wb") as f:
                pickle.dump([env, env_params, state_seq, ep_reward], f)

            if config["local_mode"]:
                vis = Visualizer(env, env_params, state_seq, ep_reward)
                vis.animate(save_dir + "/anim.gif")

        mean_rewards.append(onp.mean(agent_rewards))
        max_rewards.append(onp.max(agent_rewards))

    with open(save_dir + "/mean_mean_rewards.txt", "a") as f:
        f.write(str(onp.mean(mean_rewards)) + ",\n")

    with open(save_dir + "/var_mean_rewards.txt", "a") as f:
        f.write( str(onp.var(mean_rewards))+ ",\n")

    with open(save_dir + "/mean_max_rewards.txt", "a") as f:
        f.write(str(onp.mean(max_rewards)) + ",\n")

    with open(save_dir + "/var_max_rewards.txt", "a") as f:
        f.write(str(onp.var(max_rewards)) + ",\n")

    return onp.mean(mean_rewards), onp.var(mean_rewards), onp.mean(max_rewards), onp.var(max_rewards)



def main(env_name , num_agents, connectivity,trial, local_mode=False):

    project_name = "sapiens_env" + env_name + "_conn_" + str(connectivity) + "_n_" + str(num_agents) + "_trial_" + str(trial)


    total_timesteps = {"CartPole-v1": 8e5,
                       "MountainCar-v0": 8e5,
                       "Freeway-MinAtar": 8e6,

                       }

    config = {
        "NUM_AGENTS": num_agents,
        "BUFFER_SIZE": 10000,
        "BUFFER_BATCH_SIZE": 128,
        "SHARED_BATCH_SIZE": 5,
        "CONNECTIVITY": connectivity,
        "TOTAL_TIMESTEPS": total_timesteps[env_name],
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 25e4,
        "TARGET_UPDATE_INTERVAL": 500,
        "LR": 2.5e-4,
        "LEARNING_STARTS": 10000,
        "TRAINING_INTERVAL": 10,
        "DIVERSITY_INTERVAL": 100,
        "PROB_VISIT": 0.2,
        "VISIT_DURATION": 0.2,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": env_name,
        "SEED": trial,
        "NUM_SEEDS": 15,
        "WANDB_MODE": "offline",  # set to online to activate wandb
        "ENTITY": "eleni",
        "PROJECT": "sapiens",
        "project_name": project_name,
        "num_eval_trials": 100,
        "local_mode": local_mode # if True, evaluation data will be saved locally, otherwise under server SCRATCH
    }

    config = init_connectivity(config)

    if local_mode:
        wandb_mode = "online"

    else:
        wandb_mode = "offline"
        os.environ['WANDB_DIR'] = "/lustre/fsn1/projects/rech/imi/utw61ti/sapiens_log/wandb"


    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["sapiens", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=project_name,
        config=config,
        mode=wandb_mode
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))

    eval_info =[]
    for train_seed in range(config["NUM_SEEDS"]):
        train_info = jax.tree_map(lambda x: x[train_seed], outs["runner_state"][0])

        eval_info.append(evaluate(train_info, config, train_seed=train_seed))

    mean_mean = onp.mean([el[0] for el in eval_info])
    var_mean = onp.var([el[1] for el in eval_info])
    mean_max = onp.mean([el[2] for el in eval_info])
    var_max = onp.var([el[3] for el in eval_info])


    with open(config["project_dir"] + "/info.txt", "w") as f:

        f.write("Mean of mean group performance " + str(mean_mean))
        f.write("Var of mean group performance " + str(var_mean))
        f.write("Mean of max group performance " + str(mean_max))
        f.write("Var of max group performance " + str(var_max))




    #with open(project_dir + "/policy.pkl", "wb") as f:
    #    pickle.dump(outs["runner_state"][0], f)
    wandb.finish()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script demonstrates how to use argparse")
    parser.add_argument("--env", type=str, help="Name of the environment",default="CartPole-v1")
    parser.add_argument("--n_agents", type=int, help="Number of agents",default=10)
    parser.add_argument("--connectivity", type=str, help="Connectivity",default="fully")
    parser.add_argument("--trial", type=int, help="Trial",default=0)
    parser.add_argument("--local_mode", action='store_true')


    args = parser.parse_args()

    main(env_name=args.env, num_agents=args.n_agents, connectivity=args.connectivity, trial=args.trial, local_mode=args.local_mode)



