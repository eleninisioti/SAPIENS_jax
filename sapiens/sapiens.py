"""
PureJaxRL version of CleanRL's DQN: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer
import matplotlib.pyplot as plt
import numpy as onp
os.environ["AX_TRACEBACK_FILTERING"] = "off"



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







def make_train(config):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"]

    # we create the group

    basic_env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
                keep_neighbors=None,
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

            #exp = jax.vmap(sample_buffer)(buffer_state)

            def buffer_add_batch(i, carry):
                buffer_state, train_state, exp = carry
                i = train_state.neighbors[i]
                exp_current = jax.tree_map(lambda x: x[i, ...], exp)
                exp_current = jax.tree_map(lambda x: jnp.expand_dims(x, axis=1), exp_current)

                def add_to_buffer(i, carry):
                    current_exp = jax.tree_map(lambda x: x[i, ...], exp_current)
                    return buffer.add(carry, current_exp)

                buffer_state = jax.lax.fori_loop(lower=0, upper=config["BUFFER_SHARE_BATCH_SIZE"],
                                                 body_fun=add_to_buffer,
                                                 init_val=buffer_state)
                return (buffer_state, train_state, exp)
                # return jax.vmap(buffer.add, in_axes=(None,0))(buffer_state, exp)

            def add_buffer_agent(buffer_state, train_state, exp):
                # buffest_state =
                # buffer_state, _, _ = jax.lax.fori_loop(lower=0, upper=config["NUM_NEIGHBORS"], body_fun=buffer_add_batch,
                #                           init_val=(buffer_state, train_state, exp))
                # return buffer_state
                return jax.vmap(buffer_add_batch, in_axes=(None, 0))(buffer_state, exp)

            shared_exp = jax.vmap(sample_buffer)(buffer_state, agent_keys)
            temp = shared_exp.action.shape[0], shared_exp.action.shape[1] * shared_exp.action.shape[2], shared_exp.action.shape[3:]
            temp2 = shared_exp.obs.shape[0], shared_exp.obs.shape[1] * shared_exp.obs.shape[2], shared_exp.obs.shape[3:]

            shared_exp = jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])), shared_exp)
            total_exp = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1),timestep, shared_exp  )
            #total_exp = jax.tree_map(lambda x: jnp.expand_dims(x,axis=2),total_exp)
            #total_exp = timestep + shared_exp #TODO: just concatenate here
            #total_exp = timestep
            buffer_state = jax.vmap(buffer.add)(buffer_state, total_exp)


            #buffer_state = jax.vmap(buffer.add, in_axes=(0,0))(buffer_state, timestep)

            buffer_states = buffer_state.experience.obs

            #buffer_states = buffer_obs.reshape(buffer_obs.shape[0], buffer_obs.shape[2],buffer_obs.shape[3])

            def get_diversity(array):
                return jnp.mean(jnp.var(array, axis=0))
                #temp = jnp.unique(array, axis=0)
                #return jnp.array([temp.shape[0]])

            result_shape = jax.ShapeDtypeStruct((1,), jnp.int32)



            #buffer_diversity = pure_callback_with_shape(buffer_states)


            """
            def process_slice(i, carry):

                #carry_value, intermediate_values = carry
                # Example operation (sum along the second axis)
                result = carry + jax.pure_callback(get_diversity,  result_shape, buffer_states[i])
                return result

            # Initial carry value (not used in this example)
            init_carry = jnp.array([0], dtype=jnp.float32)
            # Use jax.lax.scan to iterate through the first dimension
            result = jax.lax.fori_loop(lower=0, upper=config["NUM_AGENTS"], body_fun=process_slice, init_val=init_carry)
            """




            def _compute_diversity(train_state):
                diversity =jax.vmap(get_diversity)(buffer_states)

                train_state = train_state.replace(buffer_diversity=diversity)
                return train_state


            def _is_diversity_time(buffer_state, train_state):
                value = (
                        (buffer.can_sample(buffer_state))
                        & (  # enough experience in buffer
                                train_state.timesteps > config["LEARNING_STARTS"]
                        )
                        & (  # pure exploration phase ended
                                train_state.timesteps % config["DIVERSITY_INTERVAL"] == 0
                        )  # training interval
                )
                return  value

            rng, _rng = jax.random.split(rng)
            """
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
            """


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
                               jax.random.uniform(key) < config["PROB_VISIT"]
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

            def _implement_visit(train_state, key, agent_id):
                # choose another subgroup
                agents = jnp.arange(config["NUM_AGENTS"])
                to_visit = jax.random.choice(key, agents)

                neighbors = train_state.neighbors # just for keeping

                new_neighbors = jnp.concatenate([train_state.neighbors[to_visit], jnp.array([to_visit])], axis=0)
                update_neighbors = jnp.concatenate([train_state.neighbors, agent_id ])

                prev_neighbors = train_state.neighbors
                updated_neighbors = prev_neighbors.at[agent_id].set(new_neighbors)
                updated_neighbors = updated_neighbors.at[to_visit].set(update_neighbors)
                updated_neighbors = updated_neighbors.at[train_state.neighbors[to_visit]].set(update_neighbors)



                train_state = train_state._replace(neighbors=updated_neighbors, keep_neighbors=neighbors)
                return train_state

            def _check_visit(is_visit_time, train_state, key, agent_id):
                train_state = jax.lax.cond(is_visit_time,
                                           lambda train_state, key: _implement_visit(train_state, key, agent_id),
                                           lambda train_state, key: (train_state, jnp.array([0.0] * config["NUM_AGENTS"])))
                return train_state

            agent_ids = jnp.arange(config["NUM_AGENTS"])

            #is_visit_time = jax.vmap(is_visit_time)(buffer_state, train_state, _rng_group, agent_ids)

            #train_state = jax.vmap(_check_visit)(is_visit_time, train_state, buffer_state, _rng_group)

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
            initial_graph.append(el)

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
            initial_graph.append(el)

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



def main(env_name,num_agents):





    config = {

        "NUM_AGENTS": num_agents,
        "BUFFER_SIZE": 10000,
        "BUFFER_BATCH_SIZE": 128,
        "SHARED_BATCH_SIZE": 32,
        "CONNECTIVITY": "fully",
        "TOTAL_TIMESTEPS": 8e5,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 25e4,
        "TARGET_UPDATE_INTERVAL": 500,
        "LR": 2.5e-4,
        "LEARNING_STARTS": 10000,
        "TRAINING_INTERVAL": 10,
        "DIVERSITY_INTERVAL": 100,
        "PROB_VISIT": 0.2,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": env_name,
        "SEED": 1,
        "NUM_SEEDS": 15,
        "WANDB_MODE": "online",  # set to online to activate wandb
        "ENTITY": "eleni",
        "PROJECT": "sapiens",
    }

    config = init_connectivity(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["DQN", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=f'sapiens_sharing_{config["ENV_NAME"]}_{config["CONNECTIVITY"]}_{config["NUM_AGENTS"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))

    save_dir = "projects/dqn/cartpole"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    temp = outs["metrics"]["timesteps"][0].tolist()
    temp2 = outs["metrics"]["returns"][0].tolist()
    for i in range(1):
        plt.plot(outs["metrics"]["timesteps"][i].tolist(),outs["metrics"]["returns"][i].tolist() )
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(save_dir + "/returns.png")
    plt.clf()
    wandb.finish()


def run_all():
    #env_name ="CartPole-v1"
    env_name ="Freeway-MinAtar"
    #env_name ="MountainCar-v0"

    envs = ["CartPole-v1", "Freeway-MinAtar","MountainCar-v0" ]
    num_agents_values = [1,5, 10, 20]
    for env_name in envs:
        for num_agents in num_agents_values:
            main(env_name, num_agents)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    run_all()
