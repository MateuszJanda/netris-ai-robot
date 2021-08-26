#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""


"""
Reinforcement learning - Deep Q-Network/Learning.
"""

import numpy as np
from dqn import config
from dqn import utils


EPSILON_DECAY = 0.99995     # Decay epsilon. Smarter NN is, then less random action should be taken
MIN_EPSILON = 0.02          # Epsilon shouldn't less than this. We always want to check something new
UPDATE_MODEL_ROUND = 400


def play_one_game(total_rounds, epsilon, env, agent, enable_learning):
    """
    Play one game.
    """
    episode_reward = 0
    episode_lines = 0
    prev_lines = 0

    # Reset environment and get initial state
    _, _, _, _, current_state = env.reset()

    # Reset flag and start iterating until episode ends
    last_round = False

    while not last_round:
        # Explore other actions with probability epsilon
        if enable_learning and np.random.random() <= epsilon:
            action = np.random.randint(0, config.ACTION_SPACE_SIZE)
        else:
            q_values = agent.q_values_for_state(current_state)
            # Choose best action
            action = np.argmax(q_values)

        last_round, lines, _, raw_next_state, next_state = env.step(action)
        episode_lines += lines
        prev_lines = lines

        # Transform new continuous state to new discrete state and count reward
        reward = adjust_reward(raw_next_state, prev_lines, lines)
        episode_reward += reward

        # Every step update replay memory and train NN model
        if enable_learning:
            transition = config.Transition(current_state, action, reward, next_state, last_round)
            agent.update_replay_memory(transition)
            agent.train(last_round)

            epsilon = adjust_epsilon(epsilon)

        # Update Q' model (this prevent instability when training)
        if enable_learning and total_rounds % UPDATE_MODEL_ROUND == 0:
            agent.update_caching_model()

        total_rounds += 1
        current_state = next_state

    return total_rounds, episode_reward, episode_lines, epsilon


def adjust_reward(board, prev_lines, lines):
    """
    Adjust reward.
    """
    if prev_lines > 0 and lines == 0:
        return - 0.36 * utils.holes(board) - 0.18 * utils.bumpiness(board)
    else:
        return -0.51 * utils.aggregate_height(board) + 0.76 * lines \
            - 0.36 * utils.holes(board) - 0.18 * utils.bumpiness(board)


def adjust_epsilon(epsilon):
    """
    Decay epsilon.
    """
    if epsilon > MIN_EPSILON:
        epsilon = epsilon * EPSILON_DECAY

    return epsilon
