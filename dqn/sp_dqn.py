#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""


"""
Stevens and Pradhan  Deep Reinforcement Learning
"""

import numpy as np
from dqn.heuristic_solver import HeuristicSolver
from dqn import config


EPSILON_DELTA = 9 * 10**-7
MIN_EPSILON = 0.1
UPDATE_MODEL_ROUND = 1000


def play_one_game(total_round, epsilon, env, agent):
    """
    Play one game.
    """
    episode_reward = 0
    solver = HeuristicSolver()

    # Reset environment and get initial state
    _, _, current_piece, raw_current_state, current_state = env.reset()
    current_state = agent.reshape_input(current_state)

    # Reset flag and start iterating until episode ends
    last_round = False

    while not last_round:
        # Explore other actions with probability epsilon
        if np.random.random() <= epsilon:
            action = solver.action(current_piece, raw_current_state)
        else:
            q_values = agent.q_values_for_state(current_state)
            # Choose best action
            action = np.argmax(q_values)

        last_round, reward, next_piece, raw_next_state, next_state = env.step(action)
        reward = adjust_reward(reward)
        next_state = agent.reshape_input(next_state)

        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        # Every step update replay memory and train NN model
        transition = config.Transition(current_state, action, reward, next_state, last_round)
        agent.update_replay_memory(transition)
        agent.train(last_round)

        current_piece = next_piece
        current_state = next_state
        raw_current_state = raw_next_state

        epsilon = adjust_epsilon(epsilon)

        # Update Q' model (this prevent instability when training)
        if total_round % UPDATE_MODEL_ROUND == 0:
            agent.update_caching_model()

        total_round += 1

    return total_round, episode_reward, epsilon


def adjust_reward(reward):
    """
    Adjust reward - lines_cleared**2
    """
    return reward**2


def adjust_epsilon(epsilon):
    """
    Decay epsilon.
    """
    if epsilon > MIN_EPSILON:
        epsilon -= EPSILON_DELTA

    return epsilon
