
def make_train(config):


    # we create the group

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
            add_batch_size=1 + config["NUM_NEIGHBORS"] * config["SHARED_BATCH_SIZE"],
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

            # init_obs = init_obs[0,...]

            # INIT BUFFER

            # rng = jax.random.PRNGKey(0)  # use a dummy rng here
            _action = basic_env.action_space(env_params).sample(rng)
            # _, _env_state = env.reset(rng, env_params)
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
                visiting=0,
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

            progress_remaining = (config["TOTAL_TIMESTEPS"] - t) / config["TOTAL_TIMESTEPS"]
            eps = jnp.where((1 - progress_remaining) > config["EPSILON_FRACTION"], config["EPSILON_END"],
                            config["EPSILON_START"] + (1 - progress_remaining) * (
                                        config["EPSILON_END"] - config["EPSILON_START"]) / config["EPSILON_FRACTION"])
            # eps = 0.1
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
            # rng_ss = jax.random.split(rng_s, config["NUM_AGENTS"]) # TODO this is wrong, we need the same seed for environemnts across agents
            # env_state = jax.tree_map(lambda x: x[:, 0], env_state)
            rng_ss = jnp.array([rng_s for el in range(config["NUM_AGENTS"])])
            action = action

            obs, env_state, reward, done, info = jax.vmap(env.step)(rng_ss, env_state, action)

            train_state = train_state.replace(
                timesteps=train_state.timesteps + 1
            )  # update timesteps count

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)

            timestep = jax.tree_map(lambda x: jnp.expand_dims(x, 1), timestep)

            # add the shared experiencees
            def sample_buffer(buffer_state, rng):
                keys = jax.random.split(rng, config["NUM_AGENTS"])
                exp = jax.vmap(buffer.sample, in_axes=(None, 0))(buffer_state,
                                                                 keys)  # actually we need n_agents samples
                exp = exp.experience.first
                exp = jax.tree_map(lambda x: x[:, :config["SHARED_BATCH_SIZE"], ...], exp)
                return exp

            # each agent samples an experience for all agents
            group_shared_exp = jax.vmap(sample_buffer)(buffer_state, agent_keys)

            # we distribute the experiences based on neighborhood
            def get_exps_for_agent(group_shared_exp, receiver_id):
                agent_neighbors = jnp.take(train_state.neighbors, receiver_id, axis=0)
                fixed_neighbor = agent_neighbors[0]

                dummy_exp = jax.tree_map(lambda x: jnp.take(jnp.take(x, fixed_neighbor, axis=0), receiver_id, axis=0),
                                         group_shared_exp)

                def get_exp_from_neighbor(neighbor_id):
                    received_exp = jax.tree_map(
                        lambda x: jnp.take(jnp.take(x, neighbor_id, axis=0), receiver_id, axis=0),
                        group_shared_exp)
                    received_exp = jax.lax.cond(neighbor_id == -1, lambda x: dummy_exp, lambda x: x, received_exp)
                    return received_exp

                received_exp = jax.vmap(get_exp_from_neighbor)(agent_neighbors)

                return received_exp

            shared_exp = jax.vmap(get_exps_for_agent, in_axes=(None, 0))(group_shared_exp, jnp.arange(config["NUM_AGENTS"]))
            shared_exp = jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])),
                                      shared_exp)
            total_exp = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1), timestep, shared_exp)

            buffer_state = jax.vmap(buffer.add)(buffer_state, total_exp)

            buffer_obs = buffer_state.experience.obs

            def get_diversity(array):
                array = array.reshape(-1, array.shape[-1])
                return jnp.mean(jnp.var(array, axis=0))

            def _compute_diversity(train_state):
                diversity = jax.vmap(get_diversity)(buffer_obs)

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
                # value = True
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
                # value = True
                return value

            def _compute_metrics(train_state):

                def get_proper_diversity(buffer):
                    buffer = buffer.reshape(-1, buffer.shape[-1])

                    unique = jnp.unique(buffer, axis=0, size=config["MAX_DIVERSITY"], fill_value=0)
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
                return value



            rng, _rng = jax.random.split(rng)
            is_learn_time = jax.vmap(is_learn_time)(buffer_state, train_state)
            is_learn_time = is_learn_time[0]

            rng_group = jax.random.split(rng, config["NUM_AGENTS"])
            _rng_group = jax.random.split(_rng, config["NUM_AGENTS"])

            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: jax.vmap(_learn_phase)(train_state, buffer_state, rng_group),
                lambda train_state, rng: (train_state, jnp.array([0.0] * config["NUM_AGENTS"])),  # do nothing
                train_state,
                _rng_group,
            )


            # train_state = jax.vmap(_check_visit, in_axes=(0, None,0,0))(is_visit_time, train_state, _rng_group, agent_ids)

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

            # """

            # """
            # buffer_state = jax.vmap(jax.vmap(buffer.add))(buffer_state, exp)

            metrics = {
                "timesteps": train_state.timesteps[..., 0],
                "updates": train_state.n_updates[..., 0],
                "loss": loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
                "loss_max": loss.max(),
                "returns_max": info["returned_episode_returns"].max(),
                "diversity_mean": train_state.buffer_diversity.mean(),
                "diversity_max": train_state.buffer_diversity.max(),
                "diversity_proper_mean": train_state.buffer_diversity_proper.mean(),
                "diversity_proper_max": train_state.buffer_diversity_proper.max(),

            }


            runner_state = (train_state, buffer_state, env_state, obs, rng)

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)

        # init_obs = jax.tree_map(lambda x: x[0, ...], init_obs)
        runner_state = (train_states, buffer_states, env_state, init_obs, _rng)

        keep_train_states = []

        for i in range(config["NUM_CHECKPOINTS"]):
            runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, int(config["NUM_UPDATES"]))
            keep_train_states.append(runner_state[0])

        return {"runner_state": runner_state, "metrics": metrics, "keep_train_states": keep_train_states}


    return train