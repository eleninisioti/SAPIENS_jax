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
import aim
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
from aim import Run
from sapiens.utils import preprocess_dict
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
    group_buffer_diversity: float
    neighbors: jnp.array
    group_indexes: jnp.array
    visiting: int


def make_train(config, logger_run):

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
            group_indexes = config["initial_group_indexes"][agent_id]


            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                target_network_params=jax.tree_map(lambda x: jnp.copy(x), network_params),
                tx=tx,
                timesteps=0,
                visiting = 0,
                group_indexes=group_indexes,
                n_updates=0,
                buffer_diversity=0.0,
                buffer_diversity_proper=0.0,
                group_buffer_diversity=0.0,
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


            train_state = train_state.replace(
                timesteps=train_state.timesteps + 1
            )  # update timesteps count

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)

            timestep = jax.tree_map(lambda x: jnp.expand_dims(x,1), timestep)


            # add the shared experiencees
            def sample_buffer(buffer_state, rng):
                keys = jax.random.split(rng, config["NUM_AGENTS"])
                exp = jax.vmap(buffer.sample, in_axes=(None, 0))(buffer_state,
                                                                 keys)  # actually we need n_agents samples
                exp = exp.experience.first
                exp = jax.tree_map(lambda x: x[:,:config["SHARED_BATCH_SIZE"],...], exp)
                return exp



            # each agent samples an experience for all agents

            if config["CONNECTIVITY"] != "independent":
                group_shared_exp = jax.vmap(sample_buffer)(buffer_state, agent_keys)
                # we distribute the experiences based on neighborhood
                def get_exps_for_agent(group_shared_exp, receiver_id):
                    agent_neighbors = jnp.take(train_state.neighbors, receiver_id, axis=0)
                    #fixed_neighbor = receiver_id # ah here I assumed that the first neighbor is correct which is not the case with the new dynamic

                    #dummy_exp = jax.tree_map(lambda x: jnp.take(jnp.take(x, fixed_neighbor, axis=0), receiver_id, axis=0),
                    #                         group_shared_exp)
                    dummy_exp = jax.tree_map(lambda x: x[receiver_id, ...], timestep)

                    def get_exp_from_neighbor(neighbor_id):
                        received_exp = jax.tree_map(
                            lambda x: jnp.take(jnp.take(x, neighbor_id, axis=0), receiver_id, axis=0),
                            group_shared_exp)
                        received_exp = jax.lax.cond(neighbor_id == -1, lambda x: dummy_exp, lambda x: x, received_exp)
                        return received_exp

                    received_exp = jax.vmap(get_exp_from_neighbor)(agent_neighbors)

                    return received_exp

                shared_exp = jax.vmap(get_exps_for_agent, in_axes=(None, 0))(group_shared_exp, jnp.arange(config["NUM_AGENTS"]))
                shared_exp = jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])), shared_exp)
                total_exp = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1),timestep, shared_exp  )

                buffer_state = jax.vmap(buffer.add)(buffer_state, total_exp)
            else:
                buffer_state = jax.vmap(buffer.add)(buffer_state, timestep)


            buffer_obs = buffer_state.experience.obs

            def get_diversity(array):
                array = array.reshape(-1, array.shape[-1])
                return jnp.mean(jnp.var(array, axis=0))


            def _compute_diversity(train_state):
                diversity =jax.vmap(get_diversity)(buffer_obs)

                train_state = train_state.replace(buffer_diversity=diversity)
                return train_state


            def _compute_group_diversity(train_state):
                new_arr = buffer_obs.reshape(-1, *buffer_obs.shape[2:])

                diversity =get_diversity(new_arr)

                train_state = train_state.replace(group_buffer_diversity=jnp.repeat(diversity, config["NUM_AGENTS"]))
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

            train_state = jax.lax.cond(
                is_diversity_time,
                lambda train_state, rng: _compute_group_diversity(train_state),
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

            
            def is_visit_time(train_state, key, visiting_agent):

                value = (
                        #(buffer.can_sample(buffer_state))
                        #&
                        #(  # enough experience in buffer
                        #        train_state.timesteps[visiting_agent]> config["LEARNING_STARTS"]
                        #)

                         (  # pure exploration phase ended
                               jax.random.uniform(key) < config["PROB_VISIT"]
                        )  # training interval
                        & (  # pure exploration phase ended
                            config["CONNECTIVITY"] == "dynamic"
                        )  # training interval
                        & (  # pure exploration phase ended
                            train_state.visiting[visiting_agent] == 0
                        )  # training interval

                )
                return value

            def is_return_time(train_state):
                # START HERE
                value = (
                    (train_state.visiting > 0)
                     &
                         (  # pure exploration phase ended
                            train_state.timesteps > (train_state.visiting + config["VISIT_DURATION"])
                        )  # training interval

                )

                #value = False
                #value = True
                return  value
                
            def _implement_visit(train_state, visiting_agent, receiving_group):

                old_neighbors = train_state.neighbors[visiting_agent, ...]

                new_neighbors = jnp.where(train_state.group_indexes == receiving_group, agents, -1)

                # inform new neighbors that you came

                def update_neighbors(neighbor_idx, agent_idx):
                    new_neighbors = train_state.neighbors[neighbor_idx,...]
                    new_neighbors = new_neighbors.at[visiting_agent].set(visiting_agent)
                    new_neighbors = jnp.where(neighbor_idx != -1, new_neighbors, train_state.neighbors[agent_idx,...])
                    return new_neighbors
                temp_neighbors = jax.vmap(update_neighbors)(new_neighbors, jnp.arange(config["NUM_AGENTS"]))
                train_state = train_state.replace(neighbors=temp_neighbors)


                def update_old_neighbors(neighbor_idx, agent_idx):
                    new_neighbors = train_state.neighbors[neighbor_idx,...]
                    new_neighbors = new_neighbors.at[visiting_agent].set(-1)
                    new_neighbors = jnp.where(neighbor_idx != -1, new_neighbors, train_state.neighbors[agent_idx,...])

                    return new_neighbors
                temp_neighbors = jax.vmap(update_old_neighbors)(old_neighbors, jnp.arange(config["NUM_AGENTS"]))
                train_state = train_state.replace(neighbors=temp_neighbors)

                train_state = train_state.replace(
                    neighbors=train_state.neighbors.at[visiting_agent, ...].set(new_neighbors))
                train_state = train_state.replace(
                    visiting=train_state.visiting.at[visiting_agent].set(train_state.timesteps[visiting_agent]))

                train_state = train_state.replace(group_indexes=train_state.group_indexes.at[visiting_agent].set(receiving_group))

                return train_state

            def _check_visit(train_state, is_visit_time,  key, visiting_agent):

                weights = jnp.where(jnp.arange(config["NUM_GROUPS"]) == train_state.group_indexes[visiting_agent], 0, 1)
                receiving_group= jax.random.choice(key, jnp.arange(config["NUM_GROUPS"]), p=weights)  # ensuring that an agent cannot visit itself
                #receiving_group = 0

                train_state = jax.lax.cond(is_visit_time,
                                           lambda train_state, key, agent_id: _implement_visit(train_state, visiting_agent, receiving_group),
                                           lambda train_state, _, agent_id: train_state, train_state, key, visiting_agent)
                return train_state


            def _implement_return(train_state, returning_agent):

                init_group = config["initial_group_indexes"][returning_agent]

                # inform current neighbors that you are leaving
                old_neighbors = jnp.where(train_state.group_indexes == train_state.group_indexes[returning_agent],
                                          jnp.arange(config["NUM_AGENTS"]),-1 )
                def update_neighbors(neighbor_idx, agent_idx):
                    new_neighbors = train_state.neighbors[neighbor_idx, ...]
                    new_neighbors = new_neighbors.at[returning_agent].set(-1)
                    new_neighbors = jnp.where(neighbor_idx != -1, new_neighbors,
                                              train_state.neighbors[agent_idx, ...]) # dont understand this
                    return new_neighbors

                old_neighbors = jax.vmap(update_neighbors)(old_neighbors, jnp.arange(config["NUM_AGENTS"]))
                train_state = train_state.replace(neighbors=old_neighbors)

                # inform neighbors that you are returning
                old_neighbors = jnp.where(train_state.group_indexes == init_group,jnp.arange(config["NUM_AGENTS"]),-1 )
                def update_neighbors(neighbor_idx, agent_idx):
                    new_neighbors = train_state.neighbors[neighbor_idx, ...]
                    new_neighbors = new_neighbors.at[returning_agent].set(returning_agent)
                    new_neighbors = jnp.where(neighbor_idx != -1, new_neighbors,
                                              train_state.neighbors[agent_idx, ...]) # dont understand this
                    return new_neighbors

                temp_neighbors = jax.vmap(update_neighbors)(old_neighbors, jnp.arange(config["NUM_AGENTS"]))
                train_state = train_state.replace(neighbors=temp_neighbors)

                new_neighbors = jnp.where(jnp.arange(config["NUM_AGENTS"])==returning_agent, -1, old_neighbors)

                train_state = train_state.replace(
                    neighbors=train_state.neighbors.at[returning_agent, ...].set(new_neighbors))
                train_state = train_state.replace(
                    visiting=train_state.visiting.at[returning_agent].set(0))
                train_state = train_state.replace(
                    group_indexes=train_state.group_indexes.at[returning_agent].set(init_group))
                """
                new_neighbors = jnp.where(train_state.group_indexes == group, agents, -1)

                # inform new neighbors that you came
                def update_neighbors(neighbor_idx, agent_idx):
                    new_neighbors = train_state.neighbors[neighbor_idx, ...]
                    new_neighbors = new_neighbors.at[visiting_agent].set(visiting_agent)
                    new_neighbors = jnp.where(neighbor_idx != -1, new_neighbors,
                                              train_state.neighbors[agent_idx, ...])
                    return new_neighbors

                temp_neighbors = jax.vmap(update_neighbors)(new_neighbors, jnp.arange(config["NUM_AGENTS"]))
                train_state = train_state.replace(neighbors=temp_neighbors)


                # inform old neighbors that you left
                def update_neighbors(neighbor_idx, agent_idx):
                    new_neighbors = train_state.neighbors[neighbor_idx, ...]
                    new_neighbors = new_neighbors.at[visiting_agent].set(-1)
                    new_neighbors = jnp.where(neighbor_idx != -1, new_neighbors,
                                              train_state.neighbors[agent_idx, ...])
                    return new_neighbors
                old_neighbors = train_state.neighbors[returning_agent]
                temp_neighbors = jax.vmap(update_neighbors)(old_neighbors, jnp.arange(config["NUM_AGENTS"]))
                train_state = train_state.replace(neighbors=temp_neighbors)

                new_neighbors = jnp.where(jnp.arange(config["NUM_AGENTS"])==returning_agent, -1, new_neighbors)
                
                train_state = train_state.replace(
                    neighbors=train_state.neighbors.at[returning_agent, ...].set(new_neighbors))
                train_state = train_state.replace(
                    visiting=train_state.visiting.at[returning_agent].set(0))
                """

                return train_state


            def _return_visit(train_state, is_return_time,  returning_agent):
                train_state = jax.lax.cond(is_return_time,
                                           lambda train_state: _implement_return(train_state, returning_agent=returning_agent),
                                           lambda train_state: train_state, train_state)
                return train_state


            # implement visits

            if config["CONNECTIVITY"] == "dynamic":

                agents = jnp.arange(config["NUM_AGENTS"])
                _rng, current_rng = jax.random.split(_rng)

                visiting_agent =jax.random.choice(current_rng, agents)
                #visiting_agent = 2

                _rng, current_rng = jax.random.split(_rng)

                is_visit_time = is_visit_time(train_state, current_rng, visiting_agent)
                _rng, current_rng = jax.random.split(_rng)

                train_state = _check_visit(train_state, is_visit_time,current_rng, visiting_agent)
                is_return_time = jax.vmap(is_return_time)(train_state)
                weights = jnp.where(is_return_time, 1, 0)
                returning_agent = jax.random.choice(_rng, agents, shape=(), p=weights)
                train_state = _return_visit(train_state, is_return_time[returning_agent], returning_agent)

            


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
                "returns_max": info["returned_episode_returns"].max(),

                "loss_max": loss.max(),
                "diversity_mean": train_state.buffer_diversity.mean(),
                "diversity_max": train_state.buffer_diversity.max(),
                "group_diversity_mean": train_state.group_buffer_diversity.mean(),
                "diversity_proper_mean": train_state.buffer_diversity_proper.mean(),
                "diversity_proper_max": train_state.buffer_diversity_proper.max(),

            }

            # report on wandb if required
            #if config.get("WANDB_MODE", "disabled") == "online":

            def callback(metrics, neighbors, visiting, group_indexes):
                if metrics["timesteps"] % 100 == 0:


                    for key, value in metrics.items():
                        logger_run.track(value, name=key)

                    print("current step " + str(metrics["timesteps"]))

                    print(metrics["returns_max"])

                    """
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

                    save_dir = config["project_dir"] + "/groups"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    with open(save_dir + "/step_" + str(metrics["timesteps"]) + ".pkl", "wb") as f:
                        pickle.dump(group_indexes, f)
                    """


                    #wandb.log({"neighbors": wandb.Image(})




            jax.debug.callback(callback, metrics, train_state.neighbors, train_state.visiting, train_state.group_indexes)

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
            group_indexes = [0]*config["NUM_AGENTS"]

        elif config["CONNECTIVITY"] == "dynamic":
            config["NUM_NEIGHBORS"] = 10 # start with one neighbor but due to visits the maximum is two

            group_id = 0
            group_indexes = []
            neighbors = []
            current_group = 0
            for i in range(config["NUM_AGENTS"]):
                current_neighbors = [-1]*config["NUM_NEIGHBORS"]
                current_neighbors[group_id] = group_id
                current_neighbors[group_id+1] = group_id+1
                neighbors.append(current_neighbors)

                #neighbors.append([group_id, group_id+1] + [-1]*8)



                if i%2 == 1:
                    group_id +=2

                if i%2 == 0 and i:

                    current_group += 1

                group_indexes.append(current_group)

            initial_graph = []
            for idx, el in enumerate(neighbors):
                el[idx] = -1
                initial_graph.append(el) # -1 means it is an empty neighbor spot
        else:
            config["NUM_NEIGHBORS"] = 0  # start with one neighbor but due to visits the maximum is two
            group_indexes = range(config["NUM_AGENTS"])
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
    config["initial_group_indexes"] = jnp.array(group_indexes)

    return config


def evaluate(train_state, config, logger_run, checkpoint):
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





    trajectories = {"agent_" + str(el): [] for el in range(config["NUM_AGENTS"])}
    save_dir = config["project_dir"] + "/eval_data/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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

        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        network = QNetwork(action_dim=env.action_space(env_params).n)
        agent_rewards = []
        for agent in range(config["NUM_AGENTS"]):

            key, agent_key = jax.random.split(key)

            agent_params = jax.tree_map(lambda x: x[agent], train_state.params)

            if config["ENV_TYPE"] == "alchemy":
                last_obs, env_state = jit_reset(config["FIXED_KEY"])
            else:
                last_obs, env_state = jit_reset(agent_key)

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
                last_obs, env_state, reward, done, info = jit_step(key_act, env_state, action)
                ep_reward.append(float(reward))
            agent_rewards.append(onp.sum(ep_reward))
            trajectories["agent_" + str(agent)].append( trajectory_steps)

            #if config["local_mode"]:
            #    vis = Visualizer(env, env_params, state_seq, ep_reward)
            #    vis.animate(save_dir + "/anim.gif")

        eval_perf["mean_rewards"].append(onp.mean(agent_rewards))
        eval_perf["max_rewards"].append(onp.max(agent_rewards))

       #traj_metrics = get_trajectory_metrics(trajectories, env_state.env_state.recipe_book, basic_env.episode_length)
        #for key, val in traj_metrics.items():
        #    eval_metrics[key].append(val)

    with open(save_dir + "/trajectories" + str(checkpoint) + ".pkl", "wb") as f:
        pickle.dump([trajectories], f)

    final_eval_perf = {}
    for key, val in eval_perf.items():
        final_eval_perf[key + "_mean"] = onp.mean(val)
        final_eval_perf[key + "_var"] = onp.var(val)

    final_eval_metrics = defaultdict(list)
    for step in range(basic_env.episode_length):
        for key, val in eval_metrics.items():
            final_eval_metrics[key + "_mean"].append(onp.mean([el[step] for el in val]))
            final_eval_metrics[key + "_var"].append(onp.var([el[step] for el in val]))

        log_dict = {key: value[-1] for key, value in final_eval_metrics.items()}

        for key, value in log_dict.items():
            logger_run.track(value, name=key)
    return final_eval_perf, final_eval_metrics



def main(env_name , num_agents, connectivity, shared_batch_size, prob_visit, visit_duration, trial, learning_rate, local_mode=False):
    project_name =  "/sapiens_env" + env_name + "_conn_" + str(connectivity) + "_shared_batch_" + str(shared_batch_size) + "_prob_visit_" + str(prob_visit) + "_visit_dur_" + str(visit_duration) + "_n_" + str(
        num_agents) + "_lr_" + str(learning_rate) + "_rew_8" +  "/trial_" + str(trial)

    e_start = 1.0
    e_end = 0.05

    params = {"num_agents": num_agents,
              "connectivity": connectivity,
              "shared_batch_size": shared_batch_size,
              "prob_visit": prob_visit,
              "visit_duration": visit_duration,
              "trial": trial,
              "learning_rate": learning_rate,}

    logger_run = Run(experiment=project_name)
    logger_run['hparams'] = params

    #wandb.login(key="575600e429b7b9e69b36d7f1584e727775d3fcfa")


    total_timesteps = {"CartPole-v1": 8e5,
                       "MountainCar-v0": 8e6,
                       "Freeway-MinAtar": 8e6,
                       "Single-path-alchemy": 1e6,
                       "Merging-paths-alchemy": 2e6,
                       "Bestoften-paths-alchemy": 8e7
                       }


    buffer_size = 25_000
    #buffer_size = 5_000
    #if connectivity == "fully":
    #    buffer_scale = num_agents
    #else:
    #   buffer_scale = 1 # maybe here I want to scale dynamic by 2?
    buffer_scale = 1
    config = {
        "NUM_AGENTS": num_agents,
        "NUM_GROUPS": num_agents/2, # only for dynamic
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
        "num_eval_trials": 2,
        "local_mode": local_mode # if True, evaluation data will be saved locally, otherwise under server SCRATCH
    }
    num_updates = config["TOTAL_TIMESTEPS"]/config["NUM_CHECKPOINTS"]
    config["TOTAL_TIMESTEPS"]= num_updates
    config["NUM_UPDATES"] = num_updates
    config["aim_hash"] = logger_run.hash


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
        yaml.dump(preprocess_dict(config), f)


    if local_mode:
        wandb_mode = "online"
        wandb_dir = "."

    else:
        wandb_mode = "offline"
        wandb_dir= "/lustre/fsn1/projects/rech/imi/utw61ti/sapiens_log/wandb"



    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, logger_run)))
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
        eval_perf, eval_metrics = evaluate(train_info, config, logger_run, checkpoint)

    #def viz_eval_metrics(eval_metrics):




    #viz_eval_metrics(eval_metrics)

    with open(config["project_dir"] + "/info.txt", "w") as f:
        f.write("Evaluation performance " + str(eval_perf))

    with open(config["project_dir"] + "/eval_metrics.pkl", "wb") as f:
        pickle.dump(eval_metrics, f)



    #with open(project_dir + "/policy.pkl", "wb") as f:
    #    pickle.dump(outs["runner_state"][0], f)





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
    parser.add_argument("--learning_rate", type=float, help="Probability of visit in dynamic networks",default=1e-4)

    args = parser.parse_args()

    main(env_name=args.env, num_agents=args.n_agents, connectivity=args.connectivity, shared_batch_size=args.shared_batch_size,
         prob_visit= args.prob_visit, visit_duration=args.visit_duration, trial=args.trial,  local_mode=args.local_mode, learning_rate=args.learning_rate)



