""" This script can be used to play with an environment.
 You are the fish robot, having two actions, which are speed and angle.
"""

import sys
import os
sys.path.append('.')
sys.path.append("envs") # to be able to import our brax
import gymnax
from envs.tiny_alchemy import envs as alchemy_envs
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

import jax.numpy as jnp

import curses

from jax import random
import jax
import math
import time
import numpy as onp
def input_char(message):
    try:
        win = curses.initscr()
        win.addstr(0, 0, message)
        while True:
            ch = win.getch()
            if ch == ord('a'):
                return [1,0,0,0] # move left
            elif ch == ord('d'):
                return [0,1,0,0] # move right
            elif ch == ord('w'):
                return [0,0,1,0] # move forward
            elif ch == ord('s'):
                return [0,0,0,1] # move back

    finally:
        curses.endwin()


def display_state(state, step):
    print("Step: " + str(step))
    print("Reward: " + str(state.reward))
    print("------------------------------------------")


def get_best_actions(env_name):
    if env_name == "Single-path-alchemy":
        actions = [2, 0, 3, 0, 4,2,5,2, 6,1,7,1,8,1,9,2] +[2, 0, 3, 0, 4,2,5,2, 6,1,7,1,8,1,9,2]
    elif env_name == "Merging-paths-alchemy":
        actions = [2, 0, 3, 0, 11,12,14,13,4,15, 22,0,23,13,24,0,25,2,26,6,27,13,28,1]
        actions = [2, 0, 3, 0, 4,2,5,2, 6,1,7,1,8,1,9,2] +[2, 0, 3, 0, 4,2,5,2, 6,1,7,1,8,1,9,2]

    return actions


def play_episode(game, episode_idx):

    # initialize environment
    env_name = game
    episode_length = 1000
    env = alchemy_envs.get_environment(env_name, key=jax.random.PRNGKey(0))

    states = []
    jit_reset =jax.jit(env.reset)
    jit_reset =env.reset
    key =random.PRNGKey(episode_idx+1)
    obs, state = jit_reset(key)

    cum_reward = 0
    action = jnp.array([0, 0 , 1, 0])
    jit_step =jax.jit(env.step)
    jit_step =env.step
    params = env.default_params

    print(state.recipe_book)
    actions = get_best_actions(env_name)

    #actions = [3, 0]

    #actions = [2,0,3,0,4,2,5,2,6,0,7,2,8,1,9,1,12,12,14,12]
    #actions = [28, 0]
    for step in range(episode_length+1):
        #print(state)
        #display_state(state, step)


        #action = jnp.array(input_char('Choose your move:'))
        action = jnp.array(actions[step])

        start = time.time()

        key, current_key = jax.random.split(key)
        print(obs)
        obs, state, reward, done, info =  jit_step(key, state, action, params)
        cum_reward += reward
        print("cum " + str(cum_reward) + " " + str(step))


        #print(action, obs, reward)

        states.append(state)
        if done:
            print("resetting")
            print(cum_reward)
            break

    print(step)


    saving_dir = "projects/human/game_" + str(game) + "/episode_" + str(episode_idx) + "/visuals/"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir, exist_ok=True)
    with open(saving_dir + "/traj_" + str(episode_idx) +"_" + str(cum_reward) + ".html", "w") as f:
        f.write(render)

    print("Episode ended. Total reward: " + str(cum_reward))


def play_game(n_episodes, game):

    for episode in range(n_episodes):
        print("Episode " + str(episode) + " is starting.")

        results = play_episode(game, episode)
        #quit()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("You are the discrete fish robot. \n"
          "You have only four actions: moving left ('a'), moving right ('d'), moving forward ('w') and moving backward ('s').")

    play_game(n_episodes=1, game="Merging-paths-alchemy")