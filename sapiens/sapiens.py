"""
PureJaxRL version of CleanRL's DQN: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py
"""
import os
from datetime import datetime
import sys
sys.path.append(".")
from collections import defaultdict
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
import yaml
from envs.tiny_alchemy import envs as alchemy_envs
from jax.experimental import io_callback
from collections import Counter
import seaborn as sns
class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        #x = nn.Dense(64)(x)
        #x = nn.relu(x)
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
    buffer_diversity_proper: float
    neighbors: jnp.array
    keep_neighbors: jnp.array
    visiting: int


def make_train(config):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"]

    # we create the group

    if "alchemy" in config["ENV_NAME"]:

        # we create the group
        basic_env = alchemy_envs.get_environment(config["ENV_NAME"], key=config["FIXED_KEY"])
        env_params = basic_env.default_params

    else:
        basic_env, env_params = gymnax.make(config["ENV_NAME"])

    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

            if config["ENV_TYPE"] == "alchemy":
                init_obs, env_state = env.reset(config["FIXED_KEY"])
            else:

                init_obs, env_state = env.reset(_rng)

            #init_obs = init_obs[0,...]

            # INIT BUFFER

            #rng = jax.random.PRNGKey(0)  # use a dummy rng here
            _action = basic_env.action_space(env_params).sample(rng)
            #_, _env_state = env.reset(rng, env_params)
            _obs, _, _reward, _done, _ = env.step(rng, env_state, _action, env_params)
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
                buffer_diversity_proper=0.0,
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


            """
            eps = jnp.clip(  # get epsilon
                (
                    (config["EPSILON_FINISH"] - config["l_START"])
                    / config["EPSILON_ANNEAL_TIME"]
                )
                * t
                + config["EPSILON_START"],
                config["EPSILON_FINISH"],
            )
            """
            progress_remaining = (config["TOTAL_TIMESTEPS"] - t) / config["TOTAL_TIMESTEPS"]
            eps = jnp.where((1 - progress_remaining) > config["EPSILON_FRACTION"], config["EPSILON_END"], config["EPSILON_START"] + (1 - progress_remaining) * (config["EPSILON_END"]- config["EPSILON_START"]) / config["EPSILON_FRACTION"])
            #eps = 0.1
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
            #rng_ss = jax.random.split(rng_s, config["NUM_AGENTS"]) # TODO this is wrong, we need the same seed for environemnts across agents
            #env_state = jax.tree_map(lambda x: x[:, 0], env_state)
            rng_ss = jnp.array([rng_s for el in range(config["NUM_AGENTS"] )])
            action = action

            obs, env_state, reward, done, info = jax.vmap(env.step)(rng_ss, env_state, action)
            """
            if config["ENV_TYPE"] == "alchemy":

                rng_temp = jnp.array([config["FIXED_KEY"] for el in range(config["NUM_AGENTS"] )])
                _, env_state_reset = jax.vmap(env.reset)(rng_temp)
            else:
                _, env_state_reset = jax.vmap(env.reset)(rng_ss)

            def expand_done_to_match(array, done):
                # Reshape `done` to match the number of leading dimensions of `array`
                expanded_done = jnp.reshape(done, done.shape + (1,) * (array.ndim - 1))
                # Broadcast `done` to the shape of `array`
                return jnp.broadcast_to(expanded_done, array.shape)

            new_env_state = jax.tree_util.tree_map(lambda a,b: jnp.where(expand_done_to_match(a, done), a,b), env_state_reset.env_state, env_state.env_state)
            #env_state = env_state.replace(env_state=new_env_state)
            """

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
            def _is_metrics_time(buffer_state, train_state):
                value = (
                        (buffer.can_sample(buffer_state))
                        &
                        (  # enough experience in buffer
                                train_state.timesteps > config["LEARNING_STARTS"]
                        )
                        & (  # pure exploration phase ended
                                train_state.timesteps % config["METRICS_INTERVAL"] == 0
                        )  # training interval
                )
                #value = True
                return value


            def _compute_metrics(train_state):

                def get_proper_diversity(buffer):
                    buffer = buffer.reshape(-1, buffer.shape[-1])

                    unique = jnp.unique(buffer, axis=0,size=config["MAX_DIVERSITY"], fill_value=0)
                    unique = jnp.sum(unique, axis=1)
                    diversity = jnp.sum(jnp.where(unique, 1, 0))

                    return diversity.astype(jnp.float32)

                diversity = jax.vmap(get_proper_diversity)(buffer_obs)
                train_state = train_state.replace(buffer_diversity_proper=diversity)
                return train_state

            is_metrics_time = jax.vmap(_is_metrics_time)(buffer_state, train_state)[0]

            rng_group = jax.random.split(rng, config["NUM_AGENTS"])
            _rng_group = jax.random.split(_rng, config["NUM_AGENTS"])

            train_state = jax.lax.cond(
                is_metrics_time,
                lambda train_state, rng: _compute_metrics(train_state),
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
                        #(buffer.can_sample(buffer_state))
                        #&
                        (  # enough experience in buffer
                                train_state.timesteps[0] > config["LEARNING_STARTS"]
                        )

                        & (  # pure exploration phase ended
                               jax.random.uniform(key) < config["PROB_VISIT"]
                        )  # training interval
                        & (  # pure exploration phase ended
                                jnp.prod(jnp.logical_not(train_state.visiting))
                        )  # training interval
                        & (  # pure exploration phase ended
                            config["CONNECTIVITY"] == "dynamic"
                        )  # training interval

                )
                return value

            def is_return_time( train_state):
                # START HERE
                value = (
                    (train_state.visiting > 0)
                     &
                         (  # pure exploration phase ended
                            train_state.timesteps > (train_state.visiting + config["VISIT_DURATION"])
                        )  # training interval

                )

                #value = True
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

                # new neighbors of agent about to take a visit
                second_neighbor = train_state.neighbors[to_visit][0]
                new_neighbors = jnp.concatenate([jnp.array([second_neighbor]), jnp.array([to_visit])], axis=0)
                updated_neighbors = prev_neighbors.at[agent_id].set(new_neighbors)

                # new neighbors of agents receiving the agent
                update_neighbors = jnp.concatenate([jnp.array([second_neighbor]), jnp.array([agent_id]) ],axis=0)
                updated_neighbors = updated_neighbors.at[to_visit].set(update_neighbors)
                update_neighbors = jnp.concatenate([jnp.array([to_visit]), jnp.array([agent_id]) ],axis=0)
                updated_neighbors = updated_neighbors.at[second_neighbor].set(update_neighbors)

                #  neighbors losing their current neighbor
                current_neighbors = train_state.neighbors[agent_id][0]
                updated_neighbors = updated_neighbors.at[current_neighbors].set(-1)

                new_visiting = train_state.visiting.at[agent_id].set(train_state.timesteps[agent_id])
                train_state = train_state.replace(neighbors=updated_neighbors, keep_neighbors=prev_neighbors, visiting=new_visiting)
                return train_state

            def _check_visit(is_visit_time, train_state, key, agent_id):


                train_state = jax.lax.cond(is_visit_time,
                                           lambda train_state, key, agent_id: _implement_visit(train_state, key, agent_id),
                                           lambda train_state, _, agent_id: train_state, train_state, key, agent_id)
                return train_state


            def _implement_return(train_state):
                temp = 0*jnp.ones_like(train_state.visiting)
                train_state = train_state.replace(neighbors=train_state.keep_neighbors,visiting=temp)
                return train_state


            def _return_visit(is_return_time, train_state):
                train_state = jax.lax.cond(is_return_time,
                                           lambda train_state: _implement_return(train_state),
                                           lambda train_state: train_state, train_state)
                return train_state


            # implement visits
            if config["CONNECTIVITY"] == "dynamic":
                is_visit_time = jax.vmap(is_visit_time, in_axes=(0, None, 0))(buffer_state, train_state, _rng_group)
                is_visit_time = jnp.sum(is_visit_time)
                #is_visit_time = 0
                agents = jnp.arange(config["NUM_AGENTS"])
                _rng, visit_key = jax.random.split(_rng)
                to_visit = jax.random.choice(visit_key, agents)
                _rng, visit_key = jax.random.split(_rng)
                visitor_id = jax.random.choice(visit_key, agents)
                #visitor_id = 0
                #to_visit= 2
                train_state = _check_visit(is_visit_time, train_state, visitor_id, to_visit)
                #train_state = train_state.replace(visiting=is_visit_time[0])
                temp = jax.vmap(is_return_time)( train_state)
                is_return_time = jnp.sum(temp)

                #is_return_time = 0
                train_state = _return_visit(is_return_time, train_state)

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
                "diversity_mean": train_state.buffer_diversity.mean(),
                "diversity_max": train_state.buffer_diversity.max(),
                "diversity_proper_mean": train_state.buffer_diversity_proper.mean(),
                "diversity_proper_max": train_state.buffer_diversity_proper.max(),

            }

            # report on wandb if required
            #if config.get("WANDB_MODE", "disabled") == "online":

            def callback(metrics, neighbors, visiting):
                if metrics["timesteps"] % 100 == 0:
                    wandb.log(metrics)

                    print("current step " + str(metrics["timesteps"]))
                    print(metrics["returns_max"])

                    save_dir = config["project_dir"] + "/neighbors"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    with open(save_dir + "/step_" + str(metrics["timesteps"]) + ".pkl", "wb") as f:
                        pickle.dump(neighbors, f)

                    save_dir = config["project_dir"] + "/visiting"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    with open(save_dir + "/step_" + str(metrics["timesteps"]) + ".pkl", "wb") as f:
                        pickle.dump(visiting, f)


                    #wandb.log({"neighbors": wandb.Image(})




            jax.debug.callback(callback, metrics, train_state.neighbors, train_state.visiting)

            #env_state = jax.tree_map(lambda x: jnp.expand_dims(x, -1),env_state)

            runner_state = (train_state, buffer_state, env_state, obs, rng)

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)

        #init_obs = jax.tree_map(lambda x: x[0, ...], init_obs)
        runner_state = (train_states, buffer_states, env_state, init_obs, _rng)

        keep_train_states = []

        for i in range(config["NUM_CHECKPOINTS"]):

            runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, int(config["NUM_UPDATES"]))
            keep_train_states.append(runner_state[0])

        return {"runner_state": runner_state, "metrics": metrics, "keep_train_states": keep_train_states}

    return train


def init_connectivity(config):
    if config["NUM_AGENTS"] >1:
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
        else:
            config["NUM_NEIGHBORS"] = 0  # start with one neighbor but due to visits the maximum is two

            initial_graph = [[]]

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
    else:
        config["NUM_NEIGHBORS"] = 0  # start with one neighbor but due to visits the maximum is two

        initial_graph = [[] ]
    config["initial_graph"] = jnp.array(initial_graph)

    return config


def evaluate(train_state, config):
    """ Evaluates a trained policy
    """
    if config["local_mode"]:
        top_dir = "projects/"
    else:
        top_dir = "/lustre/fsn1/projects/rech/imi/utw61ti/sapiens_log/projects/"

    eval_metrics = {
                    "action_conformism": [],
                    "path_conformism": [],
                    "volatility": []
    }

    eval_perf = {"mean_rewards": [],
                    "max_rewards": [],

                    }



    def get_trajectory_metrics(trajectories, recipe_book, num_steps):

        paths = list(set(recipe_book[...,4].tolist()))
        num_paths = len(paths)
        num_agents = len(trajectories)
        # get action conformism

        traj_metrics = {"action_conformism": [],
                        "path_conformism": [],
                        "volatility": []}

        agent_paths = {path: 0 for path in paths}
        agent_paths[-999] = 0
        for step in range(num_steps):
            actions = []
            for agent, traj in trajectories.items():
                actions.append(float(traj[step]["action"]))

            # Count occurrences of each element
            counts = Counter(actions)
            # Find the element with the maximum count
            majority = max(counts, key=counts.get)
            traj_metrics["action_conformism"].append(onp.sum([1 if el==majority else 0 for el in actions ])/config["NUM_AGENTS"])

            # path_conformism
            #paths = {el: 0 for el in range(num_paths)} # which paths is the agent exploring
            for agent, traj in trajectories.items():
                inventory = traj[step]["inventory"]
                current_paths = []
                for el, exists in enumerate(inventory):
                    if exists:
                        for recipe_item in recipe_book:
                            if recipe_item[2] == el:
                                paths[int(recipe_item[4])] += 1
                                current_paths.append(recipe_item[4])

                                agent_paths[int(recipe_item[4])] += 1

                if not current_paths:
                    agent_paths[-999] += 1

            majority_path = max(agent_paths, key=agent_paths.get)
            if majority_path ==-999:
                traj_metrics["path_conformism"].append(1)
            else:
                traj_metrics["path_conformism"].append(agent_paths[majority_path]/num_agents)



        # volatility
        volatilities = []
        for agent, traj in trajectories.items():
            changes = 0
            volatility = [0]

            agent_paths = []

            for step in range(num_steps):
                inventory = traj[step]["inventory"]
                current_paths = []
                for el, exists in enumerate(inventory):
                    if exists:
                        for recipe_item in recipe_book:
                            if recipe_item[2] == el:
                                paths[int(recipe_item[4])] += 1
                                current_paths.append(int(recipe_item[4]))

                    if not current_paths:
                        current_paths= [-999]

                agent_paths.append(current_paths)

            for step in range(1, num_steps):

                if agent_paths[step] != agent_paths[step-1]:
                    changes += 1
                volatility.append(changes)

            #volatility.append(changes)
            volatilities.append(volatility)

        traj_metrics["volatility"] = onp.mean(onp.array(volatilities),axis=0).tolist()

        return traj_metrics



    for trial in range(config["num_eval_trials"]):

        key = jax.random.PRNGKey(trial)
        if "alchemy" in config["ENV_NAME"]:
            # we create the group
            basic_env = alchemy_envs.get_environment(config["ENV_NAME"], key=config["FIXED_KEY"])
            env_params = basic_env.default_params

        else:
            basic_env, env_params = gymnax.make(config["ENV_NAME"])
        env = FlattenObservationWrapper(basic_env)
        env = LogWrapper(env)

        network = QNetwork(action_dim=env.action_space(env_params).n)
        agent_rewards = []
        trajectories = {"agent_" + str(el): [] for el in range(config["NUM_AGENTS"])}
        for agent in range(config["NUM_AGENTS"]):

            save_dir = top_dir + config["project_name"] + "/visuals/trial_" + str(trial) + "/agent_" + str(agent)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            key, agent_key = jax.random.split(key)

            agent_params = jax.tree_map(lambda x: x[agent], train_state.params)

            if config["ENV_TYPE"] == "alchemy":
                last_obs, env_state = env.reset(config["FIXED_KEY"])
            else:
                last_obs, env_state = env.reset(agent_key)

            done = False
            ep_reward = []
            state_seq = []
            trajectory_steps = []
            while not done:
                state_seq.append(env_state.env_state)
                q_vals = network.apply(agent_params, last_obs)

                action = jnp.argmax(q_vals, axis=-1)  # get the greedy actions

                trajectory_steps.append({"obs": last_obs, "action": action, "inventory": env_state.env_state.items})

                key, key_act = jax.random.split(key)
                last_obs, env_state, reward, done, info = env.step(key_act, env_state, action)
                ep_reward.append(float(reward))
            agent_rewards.append(onp.sum(ep_reward))
            trajectories["agent_" + str(agent)] = trajectory_steps

            with open(save_dir + "/rewards_" + str(trial) +  ".txt", "a") as f:
                f.write(str(ep_reward) + ",\n")

            with open(save_dir + "/traj_" + str(trial) +  ".pkl", "wb") as f:
                pickle.dump([env, env_params, state_seq, ep_reward], f)

            #if config["local_mode"]:
            #    vis = Visualizer(env, env_params, state_seq, ep_reward)
            #    vis.animate(save_dir + "/anim.gif")

        eval_perf["mean_rewards"].append(onp.mean(agent_rewards))
        eval_perf["max_rewards"].append(onp.max(agent_rewards))

        traj_metrics = get_trajectory_metrics(trajectories, env_state.env_state.recipe_book, basic_env.episode_length)
        for key, val in traj_metrics.items():
            eval_metrics[key].append(val)

    final_eval_perf = {}
    for key, val in eval_perf.items():
        final_eval_perf[key + "_mean"] = onp.mean(val)
        final_eval_perf[key + "_var"] = onp.var(val)

    final_eval_metrics = defaultdict(list)
    for step in range(basic_env.episode_length):
        for key, val in eval_metrics.items():
            final_eval_metrics[key + "_mean"].append(onp.mean([el[step] for el in val]))
            final_eval_metrics[key + "_var"].append(onp.var([el[step] for el in val]))

        wandb.log({key: value[-1] for key, value in final_eval_metrics.items()})

    return final_eval_perf, final_eval_metrics



def main(env_name , num_agents, connectivity, shared_batch_size, prob_visit, visit_duration, trial, learning_rate, local_mode=False):
    project_name =  "/sapiens_env" + env_name + "_conn_" + str(connectivity) + "_shared_batch_" + str(shared_batch_size) + "_prob_visit_" + str(prob_visit) + "_visit_dur_" + str(visit_duration) + "_n_" + str(
        num_agents) + "_trial_" + str(trial) + "_lr_" + str(learning_rate) + "_rew_8"
    e_start = 1.0
    e_end = 0.05

    wandb.login(key="575600e429b7b9e69b36d7f1584e727775d3fcfa")


    total_timesteps = {"CartPole-v1": 8e5,
                       "MountainCar-v0": 8e6,
                       "Freeway-MinAtar": 8e6,
                       "Single-path-alchemy": 1e6,
                       "Merging-paths-alchemy": 2e6,
                       "Bestoften-paths-alchemy": 8e7
                       }


    buffer_size = 25_000
    if connectivity == "fully":
        buffer_scale = num_agents
    else:
        buffer_scale = 1 # maybe here I want to scale dynamic by 2?

    config = {
        "NUM_AGENTS": num_agents,
        "BUFFER_SIZE": buffer_size*buffer_scale,
        "BUFFER_BATCH_SIZE": 256, # 64
        "SHARED_BATCH_SIZE": shared_batch_size,
        "CONNECTIVITY": connectivity,
        "TOTAL_TIMESTEPS": total_timesteps[env_name],
        "NUM_CHECKPOINTS": 20, # we will save intermediate states for computing metrics during evaluation (eg conformity)
        "EPSILON_START": e_start,
        "EPSILON_END": e_end,
        "EPSILON_FRACTION": 0.2,
        "TARGET_UPDATE_INTERVAL": 10000,
        "LR": learning_rate,
        "LEARNING_STARTS": 10000,
        "TRAINING_INTERVAL": 4,
        "DIVERSITY_INTERVAL": 100,
        "MAX_DIVERSITY": 5000,
        "METRICS_INTERVAL": 100,
        "PROB_VISIT": prob_visit,
        "VISIT_DURATION": visit_duration,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": env_name,
        "SEED": trial,
        "NUM_SEEDS": 1,
        #"WANDB_MODE": "online",  # set to online to activate wandb
        "ENTITY": "eleni",
        "PROJECT": "sapiens",
        "FIXED_KEY": jax.random.PRNGKey(trial),
        "project_name": project_name,
        "num_eval_trials": 10,
        "local_mode": local_mode # if True, evaluation data will be saved locally, otherwise under server SCRATCH
    }
    num_updates = config["TOTAL_TIMESTEPS"]/config["NUM_CHECKPOINTS"]
    config["TOTAL_TIMESTEPS"]= num_updates
    config["NUM_UPDATES"] = num_updates


    if config["local_mode"]:
        top_dir = "projects/"
    else:
        top_dir = "/lustre/fsn1/projects/rech/imi/utw61ti/sapiens_log/projects/"

    if "alchemy" in env_name:
        config["ENV_TYPE"] = "alchemy"
    else:
        config["ENV_TYPE"] = "gymnax"
    project_dir  = top_dir +  datetime.today().strftime(
        '%Y_%m_%d') + "/" + project_name

    if not os.path.exists(project_dir+ "/neighbors"):
        os.makedirs(project_dir + "/neighbors")
        os.makedirs(project_dir + "/visiting")
        os.makedirs(project_dir + "/metrics")



    config["project_dir"] = project_dir

    config = init_connectivity(config)

    with open(config["project_dir"] + "/config.yaml", "w") as f:
        yaml.dump(config, f)

    if local_mode:
        wandb_mode = "online"
        wandb_dir = "."

    else:
        wandb_mode = "offline"
        wandb_dir= "/lustre/fsn1/projects/rech/imi/utw61ti/sapiens_log/wandb"

    print(wandb_mode)


    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["sapiens", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=project_name,
        config=config,
        mode=wandb_mode,
        dir=wandb_dir
    )


    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))

    # find convergence time
    found_optimal = jnp.where(outs["metrics"]["returns_max"][0]==1, 0, 1)
    first_done = jnp.argmax(found_optimal)
    if first_done:
        convergence_time = first_done

    else:
        convergence_time = config["TOTAL_TIMESTEPS"]
    outs["metrics"]["convergence_time"] = convergence_time

    with open(config["project_dir"] + "/train_outs.pkl", "wb") as f:
        pickle.dump(outs["metrics"], f)
    print("evaluating")

    eval_info =[]

    for checkpoint in range(config["NUM_CHECKPOINTS"]):
        train_info = jax.tree_map(lambda x: x[0], outs["keep_train_states"][checkpoint]) # for picking the single train seed
        eval_perf, eval_metrics = evaluate(train_info, config)

    #def viz_eval_metrics(eval_metrics):




    #viz_eval_metrics(eval_metrics)

    with open(config["project_dir"] + "/info.txt", "w") as f:
        f.write("Evaluation performance " + str(eval_perf))

    with open(config["project_dir"] + "/eval_metrics.pkl", "wb") as f:
        pickle.dump(eval_metrics, f)



    #with open(project_dir + "/policy.pkl", "wb") as f:
    #    pickle.dump(outs["runner_state"][0], f)
    wandb.finish()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script demonstrates how to use argparse")
    parser.add_argument("--env", type=str, help="Name of the environment",default="CartPole-v1")
    parser.add_argument("--n_agents", type=int, help="Number of agents",default=10)
    parser.add_argument("--trial", type=int, help="Number of agents",default=1)
    parser.add_argument("--shared_batch_size", type=int, help="NUmber of exps shared at each step",default=1)
    parser.add_argument("--prob_visit", type=float, help="Probability of visit in dynamic networks",default=0.2)
    parser.add_argument("--visit_duration", type=int, help="Duration of visit in dynamic networks",default=10)
    parser.add_argument("--connectivity", type=str, help="Connectivity",default="fully")
    parser.add_argument("--local_mode", action='store_true')
    parser.add_argument("--learning_rate", type=float, help="Probability of visit in dynamic networks",default=0.2)

    args = parser.parse_args()

    main(env_name=args.env, num_agents=args.n_agents, connectivity=args.connectivity, shared_batch_size=args.shared_batch_size,
         prob_visit= args.prob_visit, visit_duration=args.visit_duration, trial=args.trial,  local_mode=args.local_mode, learning_rate=args.learning_rate)



